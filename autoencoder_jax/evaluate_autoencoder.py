import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from grids import SpectralDatasetSynthesizer
from train_autoencoder import load_model


def unnormalize_spectrum(norm_spectrum, norm_params):
    """Un-normalizes a spectrum to log-space using the provided parameters."""
    if norm_params is None:
        return norm_spectrum
    mean = norm_params['spec_mean']
    std = norm_params['spec_std']
    return (norm_spectrum * std) + mean


def compute_reconstruction_metrics(true_spectra, pred_spectra):
    """Compute various reconstruction metrics on log-spectra."""
    # Mean Squared Error
    mse = np.mean((true_spectra - pred_spectra) ** 2, axis=1)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(true_spectra - pred_spectra), axis=1)
    
    # Signal-to-Noise Ratio (calculated on log-spectra)
    signal_var = np.var(true_spectra, axis=1)
    noise_var = np.var(true_spectra - pred_spectra, axis=1)
    # Add a small epsilon to avoid division by zero
    snr = 10 * np.log10(signal_var / (noise_var + 1e-9))

    return {
        'mse': mse,
        'mae': mae,
        'snr': snr
    }


def plot_reconstruction(true_spectrum, pred_spectrum, wavelengths, save_path):
    """Plot a single spectrum reconstruction."""
    plt.figure(figsize=(12, 6))
    
    # Plot full spectrum
    plt.subplot(2, 1, 1)
    plt.plot(wavelengths, true_spectrum, label='True', alpha=0.7)
    plt.plot(wavelengths, pred_spectrum, label='Reconstructed', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Log Flux')
    plt.legend()
    plt.title('Full Spectrum')
    
    # Plot zoomed region (e.g., around 4000-5000 Å)
    plt.subplot(2, 1, 2)
    mask = (wavelengths >= 4000) & (wavelengths <= 5000)
    plt.plot(wavelengths[mask], true_spectrum[mask], label='True', alpha=0.7)
    plt.plot(wavelengths[mask], pred_spectrum[mask], label='Reconstructed', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Log Flux')
    plt.legend()
    plt.title('Zoomed Region (4000-5000 Å)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_diagnostic_reconstruction(
    true_log_spec, pred_log_spec, true_norm_spec, pred_norm_spec, wavelengths, save_path
):
    """
    Plots a 2x2 diagnostic grid comparing spectrum reconstructions in both
    log-flux space and the model's native normalized space.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Full Spectrum (Log-Flux Space)
    axes[0, 0].plot(wavelengths, true_log_spec, label='True', alpha=0.7)
    axes[0, 0].plot(wavelengths, pred_log_spec, label='Reconstructed', alpha=0.7, linestyle='--')
    axes[0, 0].set_xlabel('Wavelength (Å)')
    axes[0, 0].set_ylabel('Log Flux')
    axes[0, 0].set_title('Reconstruction in Log-Flux Space')
    axes[0, 0].legend()

    # 2. Zoomed Spectrum (Log-Flux Space)
    mask = (wavelengths >= 4000) & (wavelengths <= 5000)
    axes[0, 1].plot(wavelengths[mask], true_log_spec[mask], label='True', alpha=0.7)
    axes[0, 1].plot(wavelengths[mask], pred_log_spec[mask], label='Reconstructed', alpha=0.7, linestyle='--')
    axes[0, 1].set_xlabel('Wavelength (Å)')
    axes[0, 1].set_ylabel('Log Flux')
    axes[0, 1].set_title('Zoomed Region (Log-Flux Space)')
    axes[0, 1].legend()

    # 3. Full Spectrum (Normalized Space)
    axes[1, 0].plot(wavelengths, true_norm_spec, label='True', alpha=0.7)
    axes[1, 0].plot(wavelengths, pred_norm_spec, label='Reconstructed', alpha=0.7, linestyle='--')
    axes[1, 0].set_xlabel('Wavelength (Å)')
    axes[1, 0].set_ylabel('Normalized Value')
    axes[1, 0].set_title('Reconstruction in Model\'s Normalized Space')
    axes[1, 0].legend()

    # 4. Zoomed Spectrum (Normalized Space)
    axes[1, 1].plot(wavelengths[mask], true_norm_spec[mask], label='True', alpha=0.7)
    axes[1, 1].plot(wavelengths[mask], pred_norm_spec[mask], label='Reconstructed', alpha=0.7, linestyle='--')
    axes[1, 1].set_xlabel('Wavelength (Å)')
    axes[1, 1].set_ylabel('Normalized Value')
    axes[1, 1].set_title('Zoomed Region (Normalized Space)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_distribution(metrics, save_dir):
    """Plot distributions of reconstruction metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
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
    
    # SNR distribution
    axes[2].hist(metrics['snr'], bins=50)
    axes[2].set_xlabel('SNR (dB)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Signal-to-Noise Ratio Distribution')
    
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
    model, state, norm_params = load_model(model_path)
    
    # Create output directory
    os.makedirs('figures/autoencoder_evaluation', exist_ok=True)
    
    # Evaluate on a subset of the dataset
    test_size = min(100, len(dataset))
    test_indices = np.random.choice(len(dataset), test_size, replace=False)
    
    true_spectra, pred_spectra, latent_vectors = [], [], []
    ages, metallicities = [], []
    diagnostic_plot_data = [] # Initialize a dedicated list for plot data
    
    print("Evaluating model...")
    for idx in tqdm(test_indices):
        norm_true_spectrum = dataset.spectra[idx]
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        
        # Reconstruct spectrum using the default __call__ method
        norm_reconstructed_spectrum = model.apply(
            variables,
            norm_true_spectrum[None, :],
            training=False
        )[0]
        
        # Un-normalize for metrics and plotting
        true_spectrum = unnormalize_spectrum(norm_true_spectrum, norm_params)
        reconstructed_spectrum = unnormalize_spectrum(norm_reconstructed_spectrum, norm_params)
        
        # Encode to get latent vector using the 'encode' method
        latent = model.apply(
            variables,
            norm_true_spectrum[None, :],
            method='encode',
            training=False
        )[0]
        
        true_spectra.append(true_spectrum)
        pred_spectra.append(reconstructed_spectrum)
        latent_vectors.append(latent)
        ages.append(dataset.ages[idx])
        metallicities.append(dataset.metallicities[idx])

        # Store data for diagnostic plot
        if len(diagnostic_plot_data) < 5:
            diagnostic_plot_data.append({
                "true_log_spec": true_spectrum,
                "pred_log_spec": reconstructed_spectrum,
                "true_norm_spec": norm_true_spectrum,
                "pred_norm_spec": norm_reconstructed_spectrum
            })

    
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
    print(f"SNR: {np.mean(metrics['snr']):.2f} ± {np.std(metrics['snr']):.2f} dB")
    
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
    
    print("Saving diagnostic reconstruction plots...")
    for i, data in enumerate(diagnostic_plot_data):
        plot_diagnostic_reconstruction(
            true_log_spec=data["true_log_spec"],
            pred_log_spec=data["pred_log_spec"],
            true_norm_spec=data["true_norm_spec"],
            pred_norm_spec=data["pred_norm_spec"],
            wavelengths=dataset.wavelength,
            save_path=f'figures/autoencoder_evaluation/diagnostic_reconstruction_{i+1}.png'
        )


if __name__ == "__main__":
    main() 