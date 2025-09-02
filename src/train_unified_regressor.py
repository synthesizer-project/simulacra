"""
Unified training script for spectrum emulation regressors.

This script can train MLP regressors for both autoencoder and PCA-based
latent representations using a common training pipeline.

Usage:
    python train_unified_regressor.py <method> <model_path> <grid_dir> <grid_name> [options]
    
    Methods:
        autoencoder - Use autoencoder latent space
        pca - Use PCA component space
    
Examples:
    # Train autoencoder-based regressor
    python train_unified_regressor.py autoencoder models/autoencoder_simple_dense.msgpack /path/to/grids bc03-2003-padova00_chabrier03-0.1,100.hdf5
    
    # Train PCA-based regressor 
    python train_unified_regressor.py pca ../pca_jax/pca_models/pca_model_bc03-2003-padova00_chabrier03-0.1,100.h5 /path/to/grids bc03-2003-padova00_chabrier03-0.1,100.hdf5
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
from typing import Dict, Any
import argparse

from latent_representations import LatentRepresentation
from autoencoder_latent import AutoencoderLatentRepresentation  
from pca_latent import PCALatentRepresentation

from train_regressor import (
    RegressorMLP, train_and_evaluate as original_train_and_evaluate,
    plot_training_history
)
from grids import SpectralDatasetSynthesizer


def create_latent_representation(method: str, model_path: str, **kwargs) -> LatentRepresentation:
    """Factory function to create latent representation based on method.
    
    Args:
        method: Either 'autoencoder' or 'pca'
        model_path: Path to the trained model file
        **kwargs: Additional method-specific arguments
        
    Returns:
        LatentRepresentation instance
    """
    if method.lower() == 'autoencoder':
        return AutoencoderLatentRepresentation.load_model(model_path)
    elif method.lower() == 'pca':
        pca_group = kwargs.get('pca_group')
        whitened = kwargs.get('whitened', True)
        return PCALatentRepresentation.load_model(model_path, pca_group, whitened)
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: 'autoencoder', 'pca'")


def prepare_training_data(latent_repr: LatentRepresentation, dataset, batch_size: int = 64) -> jnp.ndarray:
    """Prepare training data by encoding spectra to latent space.
    
    Args:
        latent_repr: Latent representation instance
        dataset: Dataset containing spectra
        batch_size: Batch size for encoding (to manage memory)
        
    Returns:
        Encoded latent vectors
    """
    latent_vectors = []
    
    # Process in batches to avoid memory issues
    for i in range(0, len(dataset.spectra), batch_size):
        end_idx = min(i + batch_size, len(dataset.spectra))
        batch_spectra = dataset.spectra[i:end_idx]
        
        latent_batch = latent_repr.encode_spectra(batch_spectra)
        latent_vectors.append(latent_batch)
    
    return jnp.concatenate(latent_vectors) if latent_vectors else jnp.array([])


def load_data_unified(grid_dir: str, grid_name: str, n_samples: int, 
                     latent_repr: LatentRepresentation, seed: int = 0,
                     wl_min: float | None = None, wl_max: float | None = None):
    """Load and preprocess data for unified regressor training.
    
    Args:
        grid_dir: Directory containing spectral grids
        grid_name: Name of the grid file
        n_samples: Number of samples to generate
        latent_repr: Latent representation for normalization parameters
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Get normalization parameters from the latent representation
    norm_params = latent_repr.get_normalization_params()
    
    # Attempt to derive wavelength limits from latent representation if not provided
    if wl_min is None or wl_max is None:
        wl_arr = norm_params.get('wavelength') if norm_params else None
        if wl_arr is not None and len(wl_arr) > 0:
            wl_min = float(wl_arr.min()) if wl_min is None else wl_min
            wl_max = float(wl_arr.max()) if wl_max is None else wl_max

    # For PCA, we need to ensure exact compatibility with original PCA training
    if latent_repr.method_name == 'pca':
        # PCA always uses global normalization with specific parameters
        norm_mode = 'global'
        
        # Ensure we have the required normalization parameters
        if 'true_spec_mean' not in norm_params or 'true_spec_std' not in norm_params:
            raise ValueError(f"PCA latent representation missing required normalization parameters: {list(norm_params.keys())}")
        
        print(f"Using PCA-compatible normalization:")
        print(f"  spec_mean: {norm_params.get('true_spec_mean')}")
        print(f"  spec_std: {norm_params.get('true_spec_std')}")
        
        # Create dataset with exact PCA normalization
        dataset = SpectralDatasetSynthesizer(
            grid_dir=grid_dir,
            grid_name=grid_name,
            num_samples=n_samples,
            norm='global',
            seed=seed,
            true_spec_mean=norm_params['true_spec_mean'],
            true_spec_std=norm_params['true_spec_std'],
            wl_min=wl_min,
            wl_max=wl_max,
        )
    else:
        # Autoencoder approach - determine normalization mode
        norm_mode = 'per-spectra'  # Default for autoencoder
        if 'spec_mean' in norm_params and 'spec_std' in norm_params:
            # Check if these are scalars (global) or arrays (per-spectra)
            mean_val = norm_params['spec_mean']
            if np.isscalar(mean_val) or (hasattr(mean_val, 'ndim') and mean_val.ndim == 0):
                norm_mode = 'global'
        
        # Create the main dataset
        dataset = SpectralDatasetSynthesizer(
            grid_dir=grid_dir,
            grid_name=grid_name,
            num_samples=n_samples,
            norm=norm_mode,
            seed=seed,
            true_spec_mean=norm_params.get('spec_mean'),
            true_spec_std=norm_params.get('spec_std'),
            wl_min=wl_min,
            wl_max=wl_max,
        )
    
    # Validate wavelength consistency for PCA
    if latent_repr.method_name == 'pca':
        try:
            expected_w = int(latent_repr.pca_input_mean.shape[0])  # type: ignore[attr-defined]
        except Exception:
            expected_w = None
        if expected_w is not None and dataset.spectra.shape[1] != expected_w:
            msg = (
                f"Wavelength dimension mismatch: dataset {dataset.spectra.shape[1]} vs PCA mean {expected_w}. "
                f"Set --wl-min/--wl-max to match the PCA model (e.g., [{wl_min}, {wl_max}])."
            )
            raise ValueError(msg)

    # Split into train/validation/test
    perm = jax.random.permutation(jax.random.PRNGKey(seed), len(dataset))
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    
    train_ds = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[:train_size])
    val_ds = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[train_size:train_size + val_size])
    test_ds = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[train_size + val_size:])
    
    return train_ds, val_ds, test_ds


def train_and_evaluate_unified(
    latent_repr: LatentRepresentation,
    train_dataset, val_dataset, 
    train_latent, val_latent,
    num_epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dims: tuple = (256, 256),
    activation_name: str = 'parametric_gated',
    dropout_rate: float = 0.1,
    rng_seed: int = 0,
    patience: int = 20,
    save_path: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Unified training function that works with any latent representation.
    
    Args:
        latent_repr: The latent representation instance
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        train_latent: Training latent vectors
        val_latent: Validation latent vectors
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        hidden_dims: Hidden layer dimensions for MLP
        activation_name: Activation function name
        dropout_rate: Dropout rate
        rng_seed: Random seed
        patience: Early stopping patience
        save_path: Path to save best model
        verbose: Whether to print training progress
        
    Returns:
        Dictionary containing training results
    """
    rng = jax.random.PRNGKey(rng_seed)
    
    # Create regressor model with appropriate latent dimension
    model = RegressorMLP(
        hidden_dims=hidden_dims,
        latent_dim=latent_repr.get_latent_dim(),
        dropout_rate=dropout_rate,
        activation_name=activation_name
    )
    
    if verbose:
        print(f"Training {latent_repr.method_name} regressor:")
        print(f"  Latent dim: {latent_repr.get_latent_dim()}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Activation: {activation_name}")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
    
    # Use existing training function
    state, train_history, val_history, best_val_loss = original_train_and_evaluate(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_latent=train_latent,
        val_latent=val_latent,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        rng=rng,
        patience=patience,
        save_path=save_path,
        verbose=verbose
    )
    
    return {
        'state': state,
        'train_history': train_history,
        'val_history': val_history,
        'best_val_loss': best_val_loss,
        'model': model
    }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train unified spectrum emulation regressor')
    
    # Positional arguments
    parser.add_argument('method', choices=['autoencoder', 'pca'], 
                       help='Latent representation method')
    parser.add_argument('model_path', type=str,
                       help='Path to trained model (autoencoder .msgpack or PCA .h5)')
    parser.add_argument('grid_dir', type=str,
                       help='Directory containing spectral grids')
    parser.add_argument('grid_name', type=str,
                       help='Name of the spectral grid file')
    
    # Optional arguments
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of training samples (default: 10000)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--hidden-dims', type=str, default='512,512',
                       help='Comma-separated hidden layer dimensions (default: 512,512)')
    parser.add_argument('--activation', type=str, default='parametric_gated',
                       choices=['relu', 'parametric_gated'],
                       help='Activation function (default: parametric_gated)')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save trained models (default: models)')
    parser.add_argument('--wl-min', type=float, default=None,
                       help='Minimum wavelength (Å) for dataset (optional)')
    parser.add_argument('--wl-max', type=float, default=None,
                       help='Maximum wavelength (Å) for dataset (optional)')
    
    # PCA-specific arguments
    parser.add_argument('--pca-group', type=str, default=None,
                       help='PCA group name to load (default: latest)')
    parser.add_argument('--no-whitening', action='store_true',
                       help='Disable PCA whitening transformation')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    print(f"JAX devices: {jax.devices()}")
    print(f"Training {args.method} regressor with unified pipeline")
    
    # Parse hidden dimensions
    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(','))
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create latent representation
    print(f"Loading {args.method} model from {args.model_path}")
    
    latent_repr_kwargs = {}
    if args.method == 'pca':
        latent_repr_kwargs['pca_group'] = args.pca_group
        latent_repr_kwargs['whitened'] = not args.no_whitening
    
    latent_repr = create_latent_representation(
        method=args.method, 
        model_path=args.model_path,
        **latent_repr_kwargs
    )
    
    print(f"Loaded {args.method} model with latent dimension: {latent_repr.get_latent_dim()}")
    
    # Load datasets
    print("Loading and preparing datasets...")
    train_dataset, val_dataset, test_dataset = load_data_unified(
        grid_dir=args.grid_dir,
        grid_name=args.grid_name,
        n_samples=args.samples,
        latent_repr=latent_repr,
        seed=args.seed,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
    )
    
    # Prepare latent training targets
    print("Encoding spectra to latent representations...")
    train_latent = prepare_training_data(latent_repr, train_dataset)
    val_latent = prepare_training_data(latent_repr, val_dataset)
    
    print(f"Training latent shape: {train_latent.shape}")
    print(f"Validation latent shape: {val_latent.shape}")
    
    # Debug: Check latent value ranges to ensure they're reasonable
    print(f"Training latent range: [{train_latent.min():.4f}, {train_latent.max():.4f}]")
    print(f"Training latent mean: {train_latent.mean():.4f}, std: {train_latent.std():.4f}")
    print(f"Validation latent range: [{val_latent.min():.4f}, {val_latent.max():.4f}]")
    
    # For PCA, expect whitened weights roughly in range [-3, +3] with ~unit variance
    if latent_repr.method_name == 'pca':
        if abs(train_latent.std() - 1.0) > 0.5:
            print(f"WARNING: PCA latent std ({train_latent.std():.4f}) far from 1.0 - check whitening!")
        if train_latent.max() > 10 or train_latent.min() < -10:
            print(f"WARNING: PCA latent range very large - check normalization consistency!")
    
    # Set up save path
    grid_basename = os.path.splitext(args.grid_name)[0]
    save_path = os.path.join(args.save_dir, f'unified_regressor_{args.method}_{grid_basename}.msgpack')
    
    # Train the model
    print("\nStarting training...")
    results = train_and_evaluate_unified(
        latent_repr=latent_repr,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_latent=train_latent,
        val_latent=val_latent,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dims=hidden_dims,
        activation_name=args.activation,
        dropout_rate=args.dropout_rate,
        rng_seed=args.seed,
        patience=args.patience,
        save_path=save_path,
        verbose=True
    )
    
    # Plot training history
    plot_save_path = os.path.join('figures', f'unified_regressor_training_{args.method}_{grid_basename}.png')
    os.makedirs('figures', exist_ok=True)
    plot_training_history(results['train_history'], results['val_history'], plot_save_path)
    
    print(f"\n{args.method.capitalize()} regressor training complete!")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Model saved to: {save_path}")
    print(f"Training plot saved to: {plot_save_path}")


if __name__ == "__main__":
    main()
