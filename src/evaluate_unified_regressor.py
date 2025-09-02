"""
Unified evaluation script for trained spectrum emulation regressors.

This script can evaluate both autoencoder and PCA-based regressors using
a common evaluation pipeline that produces comprehensive metrics and figures.

Usage:
    python evaluate_unified_regressor.py <method> <latent_model_path> <regressor_path> <grid_dir> <grid_name> [options]
    
Examples:
    # Evaluate autoencoder-based regressor
    python evaluate_unified_regressor.py autoencoder models/autoencoder_simple_dense.msgpack models/unified_regressor_autoencoder.msgpack /path/to/grids bc03.hdf5
    
    # Evaluate PCA-based regressor
    python evaluate_unified_regressor.py pca ../pca_jax/pca_models/pca_model.h5 models/unified_regressor_pca.msgpack /path/to/grids bc03.hdf5
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Tuple
import argparse

from latent_representations import LatentRepresentation
from train_unified_regressor import create_latent_representation, load_data_unified
from train_regressor import load_regressor, RegressorMLP


def load_trained_regressor(regressor_path: str) -> Tuple[RegressorMLP, Dict]:
    """Load a trained regressor model.
    
    Args:
        regressor_path: Path to the saved regressor model
        
    Returns:
        Tuple of (model, params)
    """
    return load_regressor(regressor_path)


def evaluate_reconstruction_quality(
    true_spectra: jnp.ndarray, 
    pred_spectra: jnp.ndarray,
    wavelengths: jnp.ndarray = None
) -> Dict[str, Any]:
    """Evaluate reconstruction quality metrics.
    
    Args:
        true_spectra: Ground truth spectra
        pred_spectra: Predicted spectra  
        wavelengths: Wavelength array (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Convert to linear space for meaningful fractional errors
    true_linear = 10**true_spectra
    pred_linear = 10**pred_spectra
    
    # Fractional error: (pred - true) / true
    fractional_error = (pred_linear - true_linear) / (true_linear + 1e-9)
    
    # Compute various metrics
    metrics = {
        'mean_abs_frac_error': np.mean(np.abs(fractional_error), axis=1),
        'rms_frac_error': np.sqrt(np.mean(fractional_error**2, axis=1)),
        'max_abs_frac_error': np.max(np.abs(fractional_error), axis=1),
        
        # Log-space metrics
        'log_mse': np.mean((pred_spectra - true_spectra)**2, axis=1),
        'log_mae': np.mean(np.abs(pred_spectra - true_spectra), axis=1),
        
        # Overall statistics
        'overall_mean_abs_frac_error': np.mean(np.abs(fractional_error)),
        'overall_rms_frac_error': np.sqrt(np.mean(fractional_error**2)),
        'overall_log_mse': np.mean((pred_spectra - true_spectra)**2),
        'overall_log_mae': np.mean(np.abs(pred_spectra - true_spectra)),
    }
    
    # Wavelength-dependent metrics if wavelengths provided
    if wavelengths is not None:
        metrics['frac_error_vs_wavelength'] = np.mean(np.abs(fractional_error), axis=0)
        metrics['log_error_vs_wavelength'] = np.mean(np.abs(pred_spectra - true_spectra), axis=0)
        metrics['wavelengths'] = wavelengths
    
    return metrics


def unnormalize_spectra(normalized_spectra: jnp.ndarray, norm_params: Dict[str, Any], 
                       conditions: jnp.ndarray = None) -> jnp.ndarray:
    """Unnormalize spectra back to original log-flux space.
    
    Args:
        normalized_spectra: Normalized spectra from latent decoder
        norm_params: Normalization parameters from latent representation
        conditions: Physical conditions (needed for per-spectra normalization prediction)
        
    Returns:
        Unnormalized spectra in original log-flux space
    """
    if not norm_params:
        # No normalization was applied
        return normalized_spectra
    
    spec_mean = norm_params.get('spec_mean')
    spec_std = norm_params.get('spec_std')
    
    if spec_mean is None or spec_std is None:
        # No normalization parameters available
        print("Warning: No normalization parameters found, returning normalized spectra")
        return normalized_spectra
    
    # Convert to numpy arrays for easier handling
    spec_mean_arr = jnp.asarray(spec_mean)
    spec_std_arr = jnp.asarray(spec_std)
    
    # Check if normalization is per-spectrum or global
    if spec_mean_arr.ndim == 0:  # Global normalization (scalar)
        print(f"Applying global unnormalization: mean={spec_mean_arr:.4f}, std={spec_std_arr:.4f}")
        unnormalized = normalized_spectra * spec_std_arr + spec_mean_arr
    else:  # Per-spectrum normalization 
        print(f"Warning: Per-spectrum normalization detected but no normalization MLP available!")
        print(f"Using training-set statistics for unnormalization (may not be accurate)")
        print(f"Mean shape: {spec_mean_arr.shape}, Std shape: {spec_std_arr.shape}")
        
        # For per-spectrum normalization during evaluation, we would need a normalization MLP
        # to predict mean/std from conditions. Since we don't have this in the unified approach,
        # we'll use average statistics as an approximation
        if len(spec_mean_arr) == len(normalized_spectra):
            # Direct mapping case - use individual spectrum normalization
            spec_mean_batch = spec_mean_arr[:, None]  
            spec_std_batch = spec_std_arr[:, None]   
            print(f"Using direct mapping for per-spectrum normalization")
        else:
            # Use average normalization statistics for all spectra
            avg_mean = jnp.mean(spec_mean_arr)
            avg_std = jnp.mean(spec_std_arr)
            print(f"Using average statistics: mean={avg_mean:.4f}, std={avg_std:.4f}")
            spec_mean_batch = avg_mean
            spec_std_batch = avg_std
        
        unnormalized = normalized_spectra * spec_std_batch + spec_mean_batch
    
    return unnormalized


def predict_spectra_end_to_end(
    latent_repr: LatentRepresentation,
    regressor_model: RegressorMLP,
    regressor_params: Dict,
    conditions: jnp.ndarray,
    batch_size: int = 64
) -> jnp.ndarray:
    """Perform end-to-end spectrum prediction with proper unnormalization.
    
    Args:
        latent_repr: Latent representation instance
        regressor_model: Trained regressor model
        regressor_params: Regressor model parameters
        conditions: Physical conditions (age, metallicity)
        batch_size: Batch size for processing
        
    Returns:
        Predicted spectra in original log-flux space (unnormalized)
    """
    predicted_spectra = []
    
    for i in range(0, len(conditions), batch_size):
        end_idx = min(i + batch_size, len(conditions))
        batch_conditions = conditions[i:end_idx]
        
        # Predict latent vectors from conditions
        pred_latents = regressor_model.apply(
            {'params': regressor_params}, batch_conditions, training=False
        )
        
        # Decode latent vectors to normalized spectra
        pred_normalized_spectra = latent_repr.decode_latents(pred_latents)
        
        # Unnormalize spectra back to original log-flux space
        norm_params = latent_repr.get_normalization_params()
        pred_spectra_batch = unnormalize_spectra(
            pred_normalized_spectra, norm_params, batch_conditions
        )
        
        predicted_spectra.append(pred_spectra_batch)
    
    return jnp.concatenate(predicted_spectra) if predicted_spectra else jnp.array([])


def plot_reconstruction_examples(
    true_spectra: np.ndarray,
    pred_spectra: np.ndarray, 
    wavelengths: np.ndarray,
    conditions: np.ndarray,
    method_name: str,
    save_dir: str,
    n_examples: int = 5
):
    """Plot reconstruction examples with properly unnormalized spectra.
    
    Args:
        true_spectra: Ground truth spectra (should be unnormalized)
        pred_spectra: Predicted spectra (should be unnormalized) 
        wavelengths: Wavelength array
        conditions: Physical conditions (should be denormalized for display)
        method_name: Name of the method (for plot titles)
        save_dir: Directory to save plots
        n_examples: Number of examples to plot
    """
    print(f"Plotting {n_examples} reconstruction examples...")
    print(f"True spectra range in plots: [{true_spectra.min():.3f}, {true_spectra.max():.3f}]")
    print(f"Pred spectra range in plots: [{pred_spectra.min():.3f}, {pred_spectra.max():.3f}]")
    print(f"Conditions range in plots: Age [{conditions[:, 0].min():.3f}, {conditions[:, 0].max():.3f}] Gyr, Z [{conditions[:, 1].min():.4f}, {conditions[:, 1].max():.4f}]")
    os.makedirs(save_dir, exist_ok=True)
    
    # Select random examples
    n_spectra = len(true_spectra)
    indices = np.random.choice(n_spectra, min(n_examples, n_spectra), replace=False)
    
    for i, idx in enumerate(indices):
        plt.figure(figsize=(12, 8))
        
        # Full spectrum comparison
        plt.subplot(2, 1, 1)
        plt.plot(wavelengths, true_spectra[idx], label='True (unnormalized)', alpha=0.8, linewidth=1.5)
        plt.plot(wavelengths, pred_spectra[idx], label='Predicted (unnormalized)', alpha=0.8, linewidth=1.5, linestyle='--')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Log Flux (unnormalized)')
        plt.legend()
        plt.title(f'{method_name.capitalize()} Reconstruction - Age={conditions[idx, 0]:.2f} Gyr, Z={conditions[idx, 1]:.4f}')
        plt.grid(True, alpha=0.3)
        
        # Fractional error
        plt.subplot(2, 1, 2)
        true_linear = 10**true_spectra[idx]
        pred_linear = 10**pred_spectra[idx]
        frac_error = (pred_linear - true_linear) / (true_linear + 1e-9)
        plt.plot(wavelengths, frac_error * 100, color='red', alpha=0.7)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Fractional Error (%)')
        plt.title('Fractional Error')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{method_name}_reconstruction_example_{i}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_error_distributions(metrics: Dict[str, Any], method_name: str, save_dir: str):
    """Plot error distribution statistics.
    
    Args:
        metrics: Dictionary of computed metrics
        method_name: Name of the method
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mean absolute fractional error distribution
    axes[0, 0].hist(metrics['mean_abs_frac_error'] * 100, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Mean Abs. Fractional Error (%)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Mean Abs. Fractional Errors')
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMS fractional error distribution  
    axes[0, 1].hist(metrics['rms_frac_error'] * 100, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('RMS Fractional Error (%)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of RMS Fractional Errors')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log MSE distribution
    axes[1, 0].hist(metrics['log_mse'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Log MSE')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of Log-space MSE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error vs wavelength
    if 'wavelengths' in metrics:
        axes[1, 1].plot(metrics['wavelengths'], metrics['frac_error_vs_wavelength'] * 100)
        axes[1, 1].set_xlabel('Wavelength (Å)')
        axes[1, 1].set_ylabel('Mean Abs. Fractional Error (%)')
        axes[1, 1].set_title('Error vs Wavelength')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Wavelength data\nnot available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Error vs Wavelength')
    
    plt.suptitle(f'{method_name.capitalize()} Reconstruction Quality', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{method_name}_error_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_fractional_error_vs_wavelength(
    true_log_spectra: np.ndarray,
    pred_log_spectra: np.ndarray,
    wavelengths: np.ndarray,
    save_path: str,
    wl_min: float = 2000.0,
    wl_max: float = 10000.0,
):
    """Plot median and percentile bands of fractional error vs wavelength.

    Fractional error is defined in linear space as (pred - true) / (true + 1e-9).

    Args:
        true_log_spectra: Ground truth spectra in log10 flux space, shape (N, W)
        pred_log_spectra: Predicted spectra in log10 flux space, shape (N, W)
        wavelengths: Wavelength array, shape (W,)
        save_path: Output filepath for the PNG plot
        wl_min: Minimum wavelength for plotting
        wl_max: Maximum wavelength for plotting
    """
    # Convert to linear flux for fractional errors
    true_linear = 10 ** true_log_spectra
    pred_linear = 10 ** pred_log_spectra

    # Apply wavelength mask
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    wl = wavelengths[mask]
    true_sel = true_linear[:, mask]
    pred_sel = pred_linear[:, mask]

    frac_err = (pred_sel - true_sel) / (true_sel + 1e-9)

    # Percentiles across the test set
    median = np.median(frac_err, axis=0)
    p16, p84 = np.percentile(frac_err, [16, 84], axis=0)
    p2_5, p97_5 = np.percentile(frac_err, [2.5, 97.5], axis=0)
    p0_15, p99_85 = np.percentile(frac_err, [0.15, 99.85], axis=0)

    plt.figure(figsize=(14, 7))
    # 3σ in grey, 2σ and 1σ in red; convert to %
    plt.fill_between(wl, p0_15 * 100, p99_85 * 100, color='grey', alpha=0.15, label='3σ (99.7%)')
    plt.fill_between(wl, p2_5 * 100, p97_5 * 100, color='red', alpha=0.15, label='2σ (95%)')
    plt.fill_between(wl, p16 * 100, p84 * 100, color='red', alpha=0.30, label='1σ (68%)')
    plt.plot(wl, median * 100, color='black', lw=2, label='Median')
    # Reference lines
    plt.axhline(0, color='black', linestyle='--', lw=1)
    plt.axhline(1, color='blue', linestyle='--', lw=1, label='+1%')
    plt.axhline(-1, color='blue', linestyle='--', lw=1, label='-1%')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Fractional Error (%)')
    plt.title('Reconstruction Fractional Error vs. Wavelength')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(wl_min, wl_max)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def print_summary_metrics(metrics: Dict[str, Any], method_name: str):
    """Print summary of evaluation metrics.
    
    Args:
        metrics: Dictionary of computed metrics
        method_name: Name of the method
    """
    print(f"\n{method_name.upper()} RECONSTRUCTION QUALITY SUMMARY")
    print("=" * 50)
    print(f"Overall Mean Abs. Fractional Error: {metrics['overall_mean_abs_frac_error']*100:.4f}%")
    print(f"Overall RMS Fractional Error:       {metrics['overall_rms_frac_error']*100:.4f}%")
    print(f"Overall Log MSE:                    {metrics['overall_log_mse']:.6f}")
    print(f"Overall Log MAE:                    {metrics['overall_log_mae']:.6f}")
    print()
    print("Per-spectrum statistics:")
    print(f"  Median Mean Abs. Frac. Error:    {np.median(metrics['mean_abs_frac_error'])*100:.4f}%")
    print(f"  90th percentile Mean Abs. Frac. Error: {np.percentile(metrics['mean_abs_frac_error'], 90)*100:.4f}%")
    print(f"  Max Mean Abs. Frac. Error:       {np.max(metrics['mean_abs_frac_error'])*100:.4f}%")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate unified spectrum emulation regressor')
    
    # Positional arguments
    parser.add_argument('method', choices=['autoencoder', 'pca'],
                       help='Latent representation method')
    parser.add_argument('latent_model_path', type=str,
                       help='Path to latent model (autoencoder .msgpack or PCA .h5)')
    parser.add_argument('regressor_path', type=str,
                       help='Path to trained regressor model (.msgpack)')
    parser.add_argument('grid_dir', type=str,
                       help='Directory containing spectral grids')
    parser.add_argument('grid_name', type=str,
                       help='Name of the spectral grid file')
    
    # Optional arguments
    parser.add_argument('--samples', type=int, default=2000,
                       help='Number of test samples (default: 2000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for test data (default: 42)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation (default: 64)')
    parser.add_argument('--save-dir', type=str, default='figures/unified_evaluation',
                       help='Directory to save evaluation figures')
    parser.add_argument('--n-examples', type=int, default=5,
                       help='Number of reconstruction examples to plot (default: 5)')
    parser.add_argument('--wl-min', type=float,
                       help='Minimum wavelength for plots (Å, default: 2000.0)')
    parser.add_argument('--wl-max', type=float,
                       help='Maximum wavelength for plots (Å, default: 10000.0)')
    
    # PCA-specific arguments
    parser.add_argument('--pca-group', type=str, default=None,
                       help='PCA group name to load (default: latest)')
    parser.add_argument('--no-whitening', action='store_true',
                       help='Disable PCA whitening transformation')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    print(f"JAX devices: {jax.devices()}")
    print(f"Evaluating {args.method} regressor with unified pipeline")
    
    # Create latent representation
    print(f"Loading {args.method} model from {args.latent_model_path}")
    
    latent_repr_kwargs = {}
    if args.method == 'pca':
        latent_repr_kwargs['pca_group'] = args.pca_group
        latent_repr_kwargs['whitened'] = not args.no_whitening
    
    latent_repr = create_latent_representation(
        method=args.method,
        model_path=args.latent_model_path,
        **latent_repr_kwargs
    )
    
    # Load trained regressor
    print(f"Loading regressor from {args.regressor_path}")
    regressor_model, regressor_state = load_trained_regressor(args.regressor_path)
    regressor_params = regressor_state['params']
    
    print(f"Loaded models - Latent dim: {latent_repr.get_latent_dim()}, Regressor hidden dims: {regressor_model.hidden_dims}")
    
    # Determine wavelength limits from latent model if available and CLI not provided
    latent_norm = latent_repr.get_normalization_params()
    model_wavelength = latent_norm.get('wavelength') if latent_norm else None
    wl_min = args.wl_min if args.wl_min is not None else (float(model_wavelength.min()) if model_wavelength is not None else None)
    wl_max = args.wl_max if args.wl_max is not None else (float(model_wavelength.max()) if model_wavelength is not None else None)

    # Load test dataset
    print("Loading test dataset...")
    train_dataset, val_dataset, test_dataset = load_data_unified(
        grid_dir=args.grid_dir,
        grid_name=args.grid_name,
        n_samples=args.samples,
        latent_repr=latent_repr,
        seed=args.seed,
        wl_min=wl_min,
        wl_max=wl_max,
    )
    
    # Use test dataset for evaluation
    eval_dataset = test_dataset
    print(f"Evaluating on {len(eval_dataset)} test samples")
    
    # Get conditions and wavelengths
    normalized_conditions = np.array(eval_dataset.conditions)
    # Prefer wavelength grid from model if present; else use dataset
    wavelengths = None
    if model_wavelength is not None:
        wavelengths = np.array(model_wavelength)
    elif hasattr(eval_dataset, 'wavelength'):
        wavelengths = np.array(eval_dataset.wavelength)
    
    # Denormalize conditions for proper display (age, metallicity)
    # The eval_dataset has normalization parameters for the physical parameters
    denormalized_conditions = np.copy(normalized_conditions)
    if hasattr(eval_dataset, 'age_mean') and hasattr(eval_dataset, 'age_std'):
        denormalized_conditions[:, 0] = normalized_conditions[:, 0] * eval_dataset.age_std + eval_dataset.age_mean  # Age
    if hasattr(eval_dataset, 'met_mean') and hasattr(eval_dataset, 'met_std'):  
        denormalized_conditions[:, 1] = normalized_conditions[:, 1] * eval_dataset.met_std + eval_dataset.met_mean  # Metallicity
    
    print(f"Normalized conditions range: Age [{normalized_conditions[:, 0].min():.3f}, {normalized_conditions[:, 0].max():.3f}], Z [{normalized_conditions[:, 1].min():.3f}, {normalized_conditions[:, 1].max():.3f}]")
    print(f"Denormalized conditions range: Age [{denormalized_conditions[:, 0].min():.3f}, {denormalized_conditions[:, 0].max():.3f}], Z [{denormalized_conditions[:, 1].min():.4f}, {denormalized_conditions[:, 1].max():.4f}]")
    
    # Get TRUE UNNORMALIZED spectra for proper comparison
    # The eval_dataset.spectra are normalized, so we need to unnormalize them
    normalized_true_spectra = np.array(eval_dataset.spectra)
    norm_params = latent_repr.get_normalization_params()
    true_spectra = unnormalize_spectra(normalized_true_spectra, norm_params, normalized_conditions)
    true_spectra = np.array(true_spectra)
    
    print(f"True spectra shape: {true_spectra.shape} (unnormalized)")
    print(f"Normalized true spectra range: [{normalized_true_spectra.min():.3f}, {normalized_true_spectra.max():.3f}]")  
    print(f"Unnormalized true spectra range: [{true_spectra.min():.3f}, {true_spectra.max():.3f}]")
    
    # Perform end-to-end prediction
    print("Performing end-to-end spectrum prediction...")
    pred_spectra = predict_spectra_end_to_end(
        latent_repr=latent_repr,
        regressor_model=regressor_model,
        regressor_params=regressor_params,
        conditions=normalized_conditions,  # Use normalized conditions for prediction
        batch_size=args.batch_size
    )
    
    pred_spectra = np.array(pred_spectra)
    print(f"Predicted spectra shape: {pred_spectra.shape}")
    print(f"Predicted spectra range: [{pred_spectra.min():.3f}, {pred_spectra.max():.3f}]")
    
    # Sanity check: both true and predicted should be in the same (unnormalized) space
    print(f"Normalization parameters used: {list(norm_params.keys())}")
    spec_mean = norm_params.get('spec_mean')
    spec_std = norm_params.get('spec_std')
    if spec_mean is not None:
        print(f"Spec mean type: {type(spec_mean)}, shape/value: {getattr(spec_mean, 'shape', spec_mean)}")
    if spec_std is not None:  
        print(f"Spec std type: {type(spec_std)}, shape/value: {getattr(spec_std, 'shape', spec_std)}")
    
    # Evaluate reconstruction quality
    print("Computing reconstruction quality metrics...")
    metrics = evaluate_reconstruction_quality(
        true_spectra=true_spectra,
        pred_spectra=pred_spectra,
        wavelengths=wavelengths
    )
    
    # Print summary
    print_summary_metrics(metrics, args.method)
    
    # Create evaluation plots
    print(f"Creating evaluation plots in {args.save_dir}")
    
    # Plot reconstruction examples
    plot_reconstruction_examples(
        true_spectra=true_spectra,
        pred_spectra=pred_spectra,
        wavelengths=wavelengths,
        conditions=denormalized_conditions,  # Use denormalized conditions for proper titles
        method_name=args.method,
        save_dir=args.save_dir,
        n_examples=args.n_examples
    )
    
    # Plot error distributions
    plot_error_distributions(metrics, args.method, args.save_dir)

    # Plot fractional error vs wavelength across the test set
    frac_err_plot_path = os.path.join(args.save_dir, 'fractional_error_vs_wavelength.png')
    plot_fractional_error_vs_wavelength(
        true_log_spectra=true_spectra,
        pred_log_spectra=pred_spectra,
        wavelengths=wavelengths,
        save_path=frac_err_plot_path,
        wl_min=wl_min if wl_min is not None else 2000.0,
        wl_max=wl_max if wl_max is not None else 10000.0,
    )

    print(f"\nEvaluation complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
