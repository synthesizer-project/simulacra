import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from grids import SpectralDatasetSynthesizer
from train_autoencoder import load_model


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
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    
    # MSE distribution
    axes[0].hist(metrics['mse'], bins=50)
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Mean Squared Error Distribution')
    
    # MAE distribution
    axes[1].hist(metrics['mae'], bins=50)
    axes[1].set_xlabel('MAE')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Mean Absolute Error Distribution')
    
    # PSNR distribution
    axes[2].hist(metrics['psnr'], bins=50)
    axes[2].set_xlabel('PSNR (dB)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Peak Signal-to-Noise Ratio Distribution')
    
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


def plot_fractional_error_vs_wavelength(true_spectra, pred_spectra, wavelengths, save_path, epsilon=1e-9):
    """
    Computes and plots the median fractional error and percentile ranges
    as a function of wavelength.
    """
    # Calculate the fractional error, avoiding division by zero
    fractional_error = (pred_spectra - true_spectra) / (true_spectra + epsilon)
    
    # Compute the median and percentile statistics across the test set for each wavelength
    median_error = np.median(fractional_error, axis=0)
    p16_error = np.percentile(fractional_error, 16, axis=0)
    p84_error = np.percentile(fractional_error, 84, axis=0)
    
    plt.figure(figsize=(14, 6))
    
    # Plot the median fractional error
    plt.plot(wavelengths, median_error, label='Median Fractional Error', color='crimson')
    
    # Plot the 1-sigma equivalent percentile range as a shaded region
    plt.fill_between(wavelengths, p16_error, p84_error, color='crimson', alpha=0.3,
                     label='16th-84th Percentile Range')
    
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Fractional Error (pred - true) / true')
    plt.title('Fractional Error Distribution vs. Wavelength')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(-0.5, 0.5)  # Set a reasonable y-axis limit
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    if len(sys.argv) != 4:
        print("Usage: python evaluate_autoencoder.py <grid_dir> <grid_name> <model_path>")
        sys.exit(1)
        
    grid_dir, grid_name, model_path = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # Load dataset for evaluation
    N_samples = int(1e4)
    dataset = SpectralDatasetSynthesizer(
        grid_dir=grid_dir,
        grid_name=grid_name,
        num_samples=N_samples,
    )
    
    # Load model and state
    print(f"Loading model from {model_path}...")
    model, state = load_model(model_path)
    
    # Create output directory
    os.makedirs('figures/autoencoder_evaluation', exist_ok=True)
    
    # Evaluate on a subset of the dataset
    test_size = min(100, len(dataset))
    test_indices = np.random.choice(len(dataset), test_size, replace=False)
    
    true_spectra, pred_spectra, latent_vectors = [], [], []
    ages, metallicities = [], []
    
    print("Evaluating model...")
    for idx in tqdm(test_indices):
        true_spectrum = dataset.spectra[idx]
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        
        # Reconstruct spectrum using the default __call__ method
        reconstructed_spectrum = model.apply(
            variables,
            true_spectrum[None, :],
            training=False
        )[0]
        
        # Encode to get latent vector using the 'encode' method
        latent = model.apply(
            variables,
            true_spectrum[None, :],
            method='encode',
            training=False
        )[0]
        
        true_spectra.append(true_spectrum)
        pred_spectra.append(reconstructed_spectrum)
        latent_vectors.append(latent)
        ages.append(dataset.ages[idx])
        metallicities.append(dataset.metallicities[idx])
    
    # Convert to numpy arrays
    true_spectra = np.array(true_spectra)
    pred_spectra = np.array(pred_spectra)
    latent_vectors = np.array(latent_vectors)
    ages = np.array(ages)
    metallicities = np.array(metallicities)
    
    # Compute and print metrics
    metrics = compute_reconstruction_metrics(true_spectra, pred_spectra)
    print("\nReconstruction Metrics Summary:")
    print(f"MSE: {np.mean(metrics['mse']):.4f} ± {np.std(metrics['mse']):.4f}")
    print(f"MAE: {np.mean(metrics['mae']):.4f} ± {np.std(metrics['mae']):.4f}")
    print(f"PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f} dB")
    
    # Plot metrics and latent space
    plot_metrics_distribution(metrics, 'figures/autoencoder_evaluation')
    plot_latent_space(latent_vectors, ages, metallicities, 'figures/autoencoder_evaluation/latent_space.png')
    
    print("Plotting fractional error vs wavelength distribution...")
    plot_fractional_error_vs_wavelength(
        true_spectra, pred_spectra, dataset.wavelength, 
        'figures/autoencoder_evaluation/fractional_error_vs_wavelength.png'
    )
    
    # Plot example reconstructions
    print("Saving example reconstruction plots...")
    for i in range(min(5, test_size)):
        plot_reconstruction(
            true_spectra[i],
            pred_spectra[i],
            dataset.wavelength,
            f'figures/autoencoder_evaluation/reconstruction_{i+1}.png'
        )


if __name__ == "__main__":
    main() 