import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import serialization
from tqdm import tqdm

from autoencoder import SpectrumAutoencoder
from grids import SpectralDatasetSynthesizer


def load_model(model_path, spectrum_dim, latent_dim):
    """Load a trained autoencoder model."""
    # Create model instance
    model = SpectrumAutoencoder(
        spectrum_dim=spectrum_dim,
        latent_dim=latent_dim
    )
    
    # Load state
    with open(model_path, 'rb') as f:
        state_dict = serialization.from_bytes(model, f.read())
    
    return model, state_dict


def compute_reconstruction_metrics(true_spectra, pred_spectra):
    """Compute various reconstruction metrics."""
    # Mean Squared Error
    mse = np.mean((true_spectra - pred_spectra) ** 2, axis=1)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(true_spectra - pred_spectra), axis=1)
    
    # Mean Absolute Percentage Error
    # mape = np.mean(np.abs((true_spectra - pred_spectra) / true_spectra), axis=1) * 100
    
    # Peak Signal-to-Noise Ratio (PSNR)
    max_val = np.max(true_spectra)
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        # 'mape': mape,
        'psnr': psnr
    }


def plot_reconstruction(true_spectrum, pred_spectrum, wavelengths, save_path):
    """Plot a single spectrum reconstruction."""
    plt.figure(figsize=(12, 6))
    
    # Plot full spectrum
    plt.subplot(2, 1, 1)
    plt.plot(wavelengths, true_spectrum, label='True', alpha=0.7)
    plt.plot(wavelengths, pred_spectrum, label='Reconstructed', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.legend()
    plt.title('Full Spectrum')
    
    # Plot zoomed region (e.g., around 4000-5000 Å)
    plt.subplot(2, 1, 2)
    mask = (wavelengths >= 4000) & (wavelengths <= 5000)
    plt.plot(wavelengths[mask], true_spectrum[mask], label='True', alpha=0.7)
    plt.plot(wavelengths[mask], pred_spectrum[mask], label='Reconstructed', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.legend()
    plt.title('Zoomed Region (4000-5000 Å)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_distribution(metrics, save_dir):
    """Plot distributions of reconstruction metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MSE distribution
    axes[0, 0].hist(metrics['mse'], bins=50)
    axes[0, 0].set_xlabel('MSE')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Mean Squared Error Distribution')
    
    # MAE distribution
    axes[0, 1].hist(metrics['mae'], bins=50)
    axes[0, 1].set_xlabel('MAE')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Mean Absolute Error Distribution')
    
    # MAPE distribution
    # axes[1, 0].hist(metrics['mape'], bins=50)
    # axes[1, 0].set_xlabel('MAPE (%)')
    # axes[1, 0].set_ylabel('Count')
    # axes[1, 0].set_title('Mean Absolute Percentage Error Distribution')
    
    # PSNR distribution
    axes[1, 1].hist(metrics['psnr'], bins=50)
    axes[1, 1].set_xlabel('PSNR (dB)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Peak Signal-to-Noise Ratio Distribution')
    
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


def main():
    # Load dataset
    grid_dir = '../../synthesizer_grids/grids/'
    dataset = SpectralDatasetSynthesizer(grid_dir=grid_dir, grid_name='bc03-2016-Miles_chabrier-0.1,100.hdf5')
    
    # Load model
    model, state_dict = load_model(
        'models/best_autoencoder.msgpack',
        spectrum_dim=dataset.n_wavelength,
        latent_dim=128
    )
    
    # Create output directory
    import os
    os.makedirs('figures/autoencoder_evaluation', exist_ok=True)
    
    # Evaluate on test set
    test_size = min(100, len(dataset))  # Limit to 1000 samples for evaluation
    test_indices = np.random.choice(len(dataset), test_size, replace=False)
    
    # Collect predictions and metrics
    true_spectra = []
    pred_spectra = []
    latent_vectors = []
    ages = []
    metallicities = []
    
    print("Evaluating model...")
    for idx in tqdm(test_indices):
        # Get true spectrum
        true_spectrum = dataset.spectra[idx]
        true_spectra.append(true_spectrum)
        
        # Get model prediction
        variables = {'params': state_dict['params'], 'batch_stats': state_dict['batch_stats']}
        pred_spectrum = model.apply(
            variables,
            true_spectrum[None, :],  # Add batch dimension
            training=False
        )[0]  # Remove batch dimension
        pred_spectra.append(pred_spectrum)
        
        # Get latent vector
        latent = model.apply(
            variables,
            true_spectrum[None, :],
            method='encode',
            training=False
        )[0]
        latent_vectors.append(latent)
        
        # Store parameters
        ages.append(dataset.ages[idx])
        metallicities.append(dataset.metallicities[idx])
    
    # Convert to numpy arrays
    true_spectra = np.array(true_spectra)
    pred_spectra = np.array(pred_spectra)
    latent_vectors = np.array(latent_vectors)
    ages = np.array(ages)
    metallicities = np.array(metallicities)
    
    # Compute metrics
    metrics = compute_reconstruction_metrics(true_spectra, pred_spectra)
    
    # Print summary statistics
    print("\nReconstruction Metrics Summary:")
    print(f"MSE: {np.mean(metrics['mse']):.4f} ± {np.std(metrics['mse']):.4f}")
    print(f"MAE: {np.mean(metrics['mae']):.4f} ± {np.std(metrics['mae']):.4f}")
    # print(f"MAPE: {np.mean(metrics['mape']):.2f}% ± {np.std(metrics['mape']):.2f}%")
    print(f"PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f} dB")
    
    # Plot metrics distributions
    plot_metrics_distribution(metrics, 'figures/autoencoder_evaluation')
    
    # Plot latent space
    plot_latent_space(
        latent_vectors,
        ages,
        metallicities,
        'figures/autoencoder_evaluation/latent_space.png'
    )
    
    # Plot example reconstructions
    for i in range(5):  # Plot 5 examples
        plot_reconstruction(
            true_spectra[i],
            pred_spectra[i],
            dataset.wavelength,
            f'figures/autoencoder_evaluation/reconstruction_{i+1}.png'
        )


if __name__ == "__main__":
    main() 