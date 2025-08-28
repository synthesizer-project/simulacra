"""
Evaluate a trained autoencoder and generate figures.

Description:
- Default mode: loads a bundled autoencoder and a normalization MLP, samples a
  dataset from the specified grid, reconstructs final log10 spectra, and
  computes metrics.
- Global normalization mode (--global-norm): bypasses the MLP entirely and
  uses dataset-wide global mean/std for (de)normalization of spectra.
- Produces figures: metric distributions, example reconstructions, latent space
  PCA, normalization diagnostics, and fractional error vs. wavelength.

CLI:
  python evaluate_autoencoder.py <grid_dir> <grid_name> <autoencoder_model_path>
                                 [<norm_mlp_model_path>] [--global-norm] [--no-norm]

Arguments:
- grid_dir: Directory containing the spectral grid HDF5 files.
- grid_name: File name of the grid to load.
- autoencoder_model_path: Path to bundled AE msgpack produced by train_autoencoder.py.
- norm_mlp_model_path: Path to bundled NormalizationMLP msgpack produced by train_norm_mlp.py.
  Required unless --global-norm or --no-norm is provided.
- --global-norm: Use dataset-wide global mean/std (no MLP required).
- --no-norm: Disable normalization entirely; evaluate directly in log10 space
  (no MLP required).

Outputs:
- Figures saved to figures/autoencoder_evaluation/.

Notes:
- Evaluates up to 2000 randomly chosen spectra for speed.
"""

import sys
sys.path.append('..')

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

from grids import SpectralDatasetSynthesizer
from train_autoencoder import load_model as load_autoencoder_model
from train_norm_mlp import load_norm_mlp_model


def compute_reconstruction_metrics(true_log_spectra, pred_log_spectra):
    """Compute mean absolute fractional error on linear flux spectra."""
    # Convert log spectra to linear flux for meaningful metrics
    true_linear = 10**true_log_spectra
    pred_linear = 10**pred_log_spectra
    
    # Fractional error: (pred - true) / true
    fractional_error = (pred_linear - true_linear) / (true_linear + 1e-9)
    mean_abs_frac_error = np.mean(np.abs(fractional_error), axis=1)
    
    return {'mean_abs_frac_error': mean_abs_frac_error}


def plot_reconstruction(true_log_spectrum, pred_log_spectrum, wavelengths, save_path):
    """Plot a single spectrum reconstruction."""
    plt.figure(figsize=(12, 6))
    
    # Plot full spectrum
    plt.subplot(2, 1, 1)
    plt.plot(wavelengths, true_log_spectrum, label='True', alpha=0.7)
    plt.plot(wavelengths, pred_log_spectrum, label='Reconstructed', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Log Flux')
    plt.legend()
    plt.title('Full Spectrum')
    
    # Plot zoomed region (e.g., around 4000-5000 Å)
    plt.subplot(2, 1, 2)
    mask = (wavelengths >= 4000) & (wavelengths <= 5000)
    plt.plot(wavelengths[mask], true_log_spectrum[mask], label='True', color='blue', alpha=0.7)
    plt.plot(wavelengths[mask], pred_log_spectrum[mask], label='Reconstructed', color='orange', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Log Flux')
    plt.legend()
    plt.title('Zoomed Region (4000-5000 Å)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_final_reconstruction(true_log_spec, pred_log_spec, wavelengths, save_path,
                              true_mean, pred_mean, true_std, pred_std):
    """Plots the final, un-normalized reconstruction with normalization param info."""
    plt.figure(figsize=(14, 7))
    plt.plot(wavelengths, true_log_spec, label='True', alpha=0.8)
    plt.plot(wavelengths, pred_log_spec, label='Final Reconstruction', alpha=0.8, linestyle='--')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Log Flux')
    
    # Handle both per-spectrum (scalar) and per-wavelength (array) normalization
    # Convert to numpy arrays for easier handling
    true_mean_arr = np.asarray(true_mean)
    pred_mean_arr = np.asarray(pred_mean)
    true_std_arr = np.asarray(true_std)
    pred_std_arr = np.asarray(pred_std)
    
    # Handle mixed normalization types (true is per-spectrum, pred could be per-wavelength)
    # Show mean values for arrays, individual values for scalars
    
    def format_param(param_arr, param_name):
        if param_arr.size == 1:
            return f'{param_name}: {float(param_arr.item()):.2f}'
        else:
            return f'{param_name} (avg): {float(np.mean(param_arr)):.2f}'
    
    title_text = (
        f'Final Reconstruction\n'
        f'True {format_param(true_mean_arr, "Mean")}, Pred {format_param(pred_mean_arr, "Mean")}\n'
        f'True {format_param(true_std_arr, "Std")}, Pred {format_param(pred_std_arr, "Std")}'
    )
    plt.title(title_text)
    
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_diagnostic_reconstruction(
    true_norm_spec, pred_norm_spec, wavelengths, save_path
):
    """
    Plots a diagnostic grid comparing the autoencoder's reconstruction
    in the model's native normalized space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    # 1. Full Spectrum (Normalized Space)
    axes[0].plot(wavelengths, true_norm_spec, label='True Normalized', alpha=0.7)
    axes[0].plot(wavelengths, pred_norm_spec, label='AE Predicted Normalized', alpha=0.7, linestyle='--')
    axes[0].set_xlabel('Wavelength (Å)')
    axes[0].set_ylabel('Normalized Value')
    axes[0].set_title('Autoencoder Reconstruction in Normalized Space')
    axes[0].legend()

    # 2. Zoomed Spectrum (Normalized Space)
    mask = (wavelengths >= 4000) & (wavelengths <= 5000)
    axes[1].plot(wavelengths[mask], true_norm_spec[mask], label='True Normalized', alpha=0.7)
    axes[1].plot(wavelengths[mask], pred_norm_spec[mask], label='AE Predicted Normalized', alpha=0.7, linestyle='--')
    axes[1].set_xlabel('Wavelength (Å)')
    axes[1].set_ylabel('Normalized Value')
    axes[1].set_title('Zoomed Region (Normalized Space)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_normalization_diagnostics(true_means, pred_means, true_stds, pred_stds, save_path):
    """
    Plots diagnostic comparisons for the predicted normalization parameters.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Flatten inputs
    true_means = true_means.flatten()
    pred_means = pred_means.flatten()
    true_stds = true_stds.flatten()
    pred_stds = pred_stds.flatten()

    # --- Mean Diagnostics ---
    # Scatter plot of predicted vs. true mean
    axes[0, 0].scatter(true_means, pred_means, alpha=0.3)
    axes[0, 0].plot([true_means.min(), true_means.max()], [true_means.min(), true_means.max()], 'r--', label='y=x')
    axes[0, 0].set_xlabel("True Mean")
    axes[0, 0].set_ylabel("Predicted Mean")
    axes[0, 0].set_title("Mean Prediction")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Histogram of mean residuals
    mean_residuals = pred_means - true_means
    axes[0, 1].hist(mean_residuals, bins=50, alpha=0.7)
    axes[0, 1].set_xlabel("Predicted Mean - True Mean")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Mean Prediction Residuals")
    axes[0, 1].axvline(0, color='r', linestyle='--')
    axes[0, 1].grid(True)

    # --- Std Dev Diagnostics ---
    # Scatter plot of predicted vs. true std
    axes[1, 0].scatter(true_stds, pred_stds, alpha=0.3)
    axes[1, 0].plot([true_stds.min(), true_stds.max()], [true_stds.min(), true_stds.max()], 'r--', label='y=x')
    axes[1, 0].set_xlabel("True Std Dev")
    axes[1, 0].set_ylabel("Predicted Std Dev")
    axes[1, 0].set_title("Std Dev Prediction")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Histogram of std dev residuals
    std_residuals = pred_stds - true_stds
    axes[1, 1].hist(std_residuals, bins=50, alpha=0.7)
    axes[1, 1].set_xlabel("Predicted Std Dev - True Std Dev")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Std Dev Prediction Residuals")
    axes[1, 1].axvline(0, color='r', linestyle='--')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_distribution(metrics, save_dir):
    """Plot distribution of mean absolute fractional error."""
    plt.figure(figsize=(8, 6))
    plt.hist(metrics['mean_abs_frac_error'] * 100, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Mean Absolute Fractional Error (%)')
    plt.ylabel('Count')
    plt.title('Mean Absolute Fractional Error Distribution\n(Computed on Linear Flux)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/reconstruction_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_latent_space(latent_vectors, ages, metallicities, save_path):
    """Plot 2D projection of latent space with age/metallicity coloring."""
    from sklearn.decomposition import PCA
    
    # Project to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot age coloring
    scatter1 = ax1.scatter(latent_2d[:, 0], latent_2d[:, 1], c=ages, cmap='viridis')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Latent Space (colored by age)')
    plt.colorbar(scatter1, ax=ax1, label='Age (Gyr)')
    
    # Plot metallicity coloring
    scatter2 = ax2.scatter(latent_2d[:, 0], latent_2d[:, 1], c=metallicities, cmap='plasma')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Latent Space (colored by metallicity)')
    plt.colorbar(scatter2, ax=ax2, label='Metallicity (Z/Z_sun)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_fractional_error_vs_wavelength(true_spectra, pred_spectra, wavelengths, save_path, wl_min=2000.0, wl_max=10000.0):
    """
    Computes and plots the median fractional error and percentile ranges vs. wavelength.
    Fractional error is computed on linear flux: (pred_linear - true_linear) / (true_linear + 1e-9).
    """
    # Convert log spectra to linear flux for fractional error calculation
    true_linear = 10**true_spectra
    pred_linear = 10**pred_spectra
    
    # Calculate fractional error
    fractional_error = (pred_linear - true_linear) / (true_linear + 1e-9)
    
    # Compute percentiles for sigma bands
    median_error = np.median(fractional_error, axis=0)
    
    # Define percentiles for sigma bands  
    p16, p84 = np.percentile(fractional_error, [16, 84], axis=0)      # 1-sigma
    p2_5, p97_5 = np.percentile(fractional_error, [2.5, 97.5], axis=0) # 2-sigma
    p0_15, p99_85 = np.percentile(fractional_error, [0.15, 99.85], axis=0) # 3-sigma

    plt.figure(figsize=(14, 7))

    # Plot sigma bands: 1σ and 2σ in red, 3σ in grey
    plt.fill_between(wavelengths, p0_15 * 100, p99_85 * 100, color='grey', alpha=0.15, label='3σ (99.7%)')
    plt.fill_between(wavelengths, p2_5 * 100, p97_5 * 100, color='red', alpha=0.15, label='2σ (95%)')
    plt.fill_between(wavelengths, p16 * 100, p84 * 100, color='red', alpha=0.30, label='1σ (68%)')

    # Plot the median error on top
    plt.plot(wavelengths, median_error * 100, label='Median Fractional Error', color='black', lw=2)

    # Zero and ±1% reference lines
    plt.axhline(0, color='black', linestyle='--', lw=1)
    plt.axhline(1, color='blue', linestyle='--', lw=1, label='+1%')
    plt.axhline(-1, color='blue', linestyle='--', lw=1, label='-1%')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Fractional Error (%)')
    plt.title('Fractional Error vs. Wavelength')
    # Set x-limits first, then compute y-limits from data within that range
    plt.xlim(wl_min, wl_max)

    # Compute dynamic y-limits based on the fractional error distribution within x-lims
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    fe_window = fractional_error[:, mask]
    # Guard against empty masks
    if fe_window.size > 0:
        lower = np.percentile(fe_window, 0.5)
        upper = np.percentile(fe_window, 99.5)
        span = max(upper - lower, 1e-6)
        pad = 0.10 * span
        y_min = (lower - pad) * 100.0
        y_max = (upper + pad) * 100.0
        # Ensure zero is visible and bounds are ordered
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 0.0)
        plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Manual CLI parsing to allow optional flag
    args = sys.argv[1:]
    use_global_norm = False
    use_no_norm = False
    if "--global-norm" in args:
        args.remove("--global-norm")
        use_global_norm = True
    if "--no-norm" in args:
        args.remove("--no-norm")
        use_no_norm = True

    if (use_no_norm and len(args) != 3) or ((not use_no_norm) and use_global_norm and len(args) != 3) or ((not use_no_norm) and (not use_global_norm) and len(args) != 4):
        print("Usage: python evaluate_autoencoder.py <grid_dir> <grid_name> <autoencoder_model_path> [<norm_mlp_model_path>] [--global-norm] [--no-norm]")
        sys.exit(1)

    grid_dir, grid_name, ae_model_path = args[0], args[1], args[2]
    mlp_model_path = None if (use_global_norm or use_no_norm) else args[3]
    
    # Create output directory
    output_dir = f'figures/autoencoder_evaluation/'
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset for evaluation
    N_samples = int(1e4)
    dataset = SpectralDatasetSynthesizer(
        grid_dir=grid_dir,
        grid_name=grid_name,
        num_samples=N_samples,
        norm=(None if use_no_norm else ('global' if use_global_norm else 'per-spectra')),
    )
    
    # Load models
    print(f"Loading autoencoder from {ae_model_path}...")
    ae_model, ae_state, _ = load_autoencoder_model(ae_model_path)
    ae_variables = {'params': ae_state.params, 'batch_stats': ae_state.batch_stats}
    
    if use_no_norm:
        print("Using no normalization (direct log-space, no MLP).")
        mlp_model, mlp_params, mlp_variables = None, None, None
    elif use_global_norm:
        print("Using global normalization (no MLP).")
        mlp_model, mlp_params, mlp_variables = None, None, None
    else:
        print(f"Loading normalization MLP from {mlp_model_path}...")
        mlp_model, mlp_params = load_norm_mlp_model(mlp_model_path)
        mlp_variables = {'params': mlp_params}

    # --- Evaluation ---
    test_size = min(2000, len(dataset))
    test_indices = np.random.choice(len(dataset), test_size, replace=False)
    
    # Get ground truth data
    conditions = dataset.conditions[test_indices]
    norm_true_spectra = dataset.spectra[test_indices]
    if not use_no_norm:
        # Handle global scalar mean/std by expanding to per-sample arrays for convenience
        if np.isscalar(dataset.true_spec_mean) or (np.asarray(dataset.true_spec_mean).ndim == 0):
            true_means = np.full((len(test_indices), 1), float(np.asarray(dataset.true_spec_mean)))
            true_stds = np.full((len(test_indices), 1), float(np.asarray(dataset.true_spec_std)))
        else:
            true_means = dataset.true_spec_mean[test_indices]
            true_stds = dataset.true_spec_std[test_indices]
    ages = dataset.ages[test_indices]
    metallicities = dataset.metallicities[test_indices]
    
    print(f"Evaluating system on {test_size} spectra...")
    
    # 1. Get Autoencoder predictions for the normalized shape
    pred_norm_spectra = ae_model.apply(ae_variables, norm_true_spectra, training=False)
    latent_vectors = ae_model.apply(ae_variables, norm_true_spectra, method=ae_model.encode, training=False)
    
    # 2. Get normalization parameters or bypass
    if use_no_norm:
        pred_means, pred_stds = None, None
    elif use_global_norm:
        pred_means, pred_stds = true_means, true_stds
    else:
        pred_means, pred_stds = mlp_model.apply(mlp_variables, conditions)

    # 3. Combine predictions to get final log-flux spectra
    if use_no_norm:
        pred_log_spectra = pred_norm_spectra
        true_log_spectra = norm_true_spectra
    else:
        pred_log_spectra = (pred_norm_spectra * pred_stds) + pred_means
        true_log_spectra = (norm_true_spectra * true_stds) + true_means
    
    # --- Compute and Plot Metrics on Final Reconstruction ---
    print("Computing metrics on final reconstruction...")
    metrics = compute_reconstruction_metrics(true_log_spectra, pred_log_spectra)
    
    print("\nReconstruction Metrics Summary:")
    print(f"Mean Absolute Fractional Error: {np.mean(metrics['mean_abs_frac_error']):.4f} ± {np.std(metrics['mean_abs_frac_error']):.4f} ({np.mean(metrics['mean_abs_frac_error'])*100:.2f}%)")
    print(f"Median Absolute Fractional Error: {np.median(metrics['mean_abs_frac_error'])*100:.2f}%")
    
    plot_metrics_distribution(metrics, output_dir)
    
    # --- Plot Example Reconstructions ---
    print("Plotting example reconstructions...")
    num_examples_to_plot = 5
    for i in range(num_examples_to_plot):
        # Plot the final combined reconstruction
        if use_no_norm:
            plot_reconstruction(
                true_log_spectra[i],
                pred_log_spectra[i],
                dataset.wavelength,
                save_path=f'{output_dir}/final_reconstruction_example_{i}.png'
            )
        else:
            plot_final_reconstruction(
                true_log_spectra[i],
                pred_log_spectra[i],
                dataset.wavelength,
                save_path=f'{output_dir}/final_reconstruction_example_{i}.png',
                true_mean=true_means[i],
                pred_mean=pred_means[i],
                true_std=true_stds[i],
                pred_std=pred_stds[i]
            )
        # Plot the diagnostic for the autoencoder's normalized output
        plot_diagnostic_reconstruction(
            norm_true_spectra[i],
            pred_norm_spectra[i],
            dataset.wavelength,
            save_path=f'{output_dir}/diagnostic_ae_reconstruction_example_{i}.png'
        )

    # --- Plot Latent Space ---
    print("Plotting autoencoder latent space...")
    plot_latent_space(latent_vectors, ages, metallicities, save_path=f'{output_dir}/latent_space.png')

    # --- Plot Normalization Parameter Diagnostics ---
    if not (use_global_norm or use_no_norm):
        print("Plotting normalization MLP diagnostics...")
        plot_normalization_diagnostics(
            true_means,
            pred_means,
            true_stds,
            pred_stds,
            save_path=f'{output_dir}/normalization_diagnostics.png'
        )

    # --- Evaluate Autoencoder Performance in Isolation ---
    print("Evaluating autoencoder performance (using true normalization)...")
    if use_no_norm:
        # Already in final space
        ae_pred_log_spectra = pred_norm_spectra
        ae_true_log_spectra = norm_true_spectra
    else:
        # Use true normalization parameters to isolate autoencoder performance
        ae_pred_log_spectra = (pred_norm_spectra * true_stds) + true_means
        ae_true_log_spectra = (norm_true_spectra * true_stds) + true_means
    
    # Compute metrics for autoencoder-only performance
    ae_metrics = compute_reconstruction_metrics(ae_true_log_spectra, ae_pred_log_spectra)
    print("Autoencoder-Only Performance (using true normalization):")
    print(f"  Mean Absolute Fractional Error: {np.mean(ae_metrics['mean_abs_frac_error']):.4f} ± {np.std(ae_metrics['mean_abs_frac_error']):.4f} ({np.mean(ae_metrics['mean_abs_frac_error'])*100:.2f}%)")
    print(f"  Median Absolute Fractional Error: {np.median(ae_metrics['mean_abs_frac_error'])*100:.2f}%")
    
    # Plot autoencoder-only fractional error vs wavelength
    print("Plotting autoencoder-only fractional error vs wavelength...")
    plot_fractional_error_vs_wavelength(
        ae_true_log_spectra,
        ae_pred_log_spectra,
        dataset.wavelength,
        save_path=f'{output_dir}/autoencoder_only_fractional_error_vs_wavelength.png'
    )

    # --- Plot Fractional Error vs. Wavelength on Final Spectra ---
    print("Plotting fractional error of final reconstruction...")
    plot_fractional_error_vs_wavelength(
        true_log_spectra,
        pred_log_spectra,
        dataset.wavelength,
        save_path=f'{output_dir}/fractional_error_vs_wavelength.png'
    )

    print(f"\nEvaluation complete. Figures saved to {output_dir}")


if __name__ == '__main__':
    main() 
