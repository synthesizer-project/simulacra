import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from tqdm import tqdm
import os
import jax

from train_autoencoder import load_model as load_autoencoder
from train_regressor import load_regressor, load_data_regressor


def unnormalize_spectrum(norm_spectrum, norm_params):
    """Un-normalizes a spectrum to log-space using the provided parameters."""
    if norm_params is None:
        return norm_spectrum
    mean = norm_params['spec_mean']
    std = norm_params['spec_std']
    return (norm_spectrum * std) + mean


def generate_spectrum(regressor, regressor_state, autoencoder, autoencoder_state, age, metallicity):
    """Generate a spectrum for given age and metallicity."""
    conditions = jnp.array([[age, metallicity]])
    
    # Generate latent vector using regressor
    regressor_variables = {'params': regressor_state['params']}
    latent = regressor.apply(regressor_variables, conditions, training=False)
    
    # Decode latent vector to spectrum using autoencoder
    autoencoder_variables = {'params': autoencoder_state.params, 'batch_stats': autoencoder_state.batch_stats}
    spectrum = autoencoder.apply(autoencoder_variables, latent, method='decode', training=False)
    
    return spectrum[0]


def compute_metrics(true_spectra, pred_spectra):
    """Compute various reconstruction metrics on log-spectra."""
    mse = np.mean((true_spectra - pred_spectra) ** 2, axis=1)
    mae = np.mean(np.abs(true_spectra - pred_spectra), axis=1)
    
    signal_var = np.var(true_spectra, axis=1)
    noise_var = np.var(true_spectra - pred_spectra, axis=1)
    snr = 10 * np.log10(signal_var / (noise_var + 1e-9))
    
    return {'mse': mse, 'mae': mae, 'snr': snr}


def plot_spectrum_comparison(true_spectrum, pred_spectrum, wavelengths, age, metallicity, save_path):
    """Plot comparison of true and predicted spectra."""
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(wavelengths, true_spectrum, label='True', alpha=0.7)
    plt.plot(wavelengths, pred_spectrum, label='Predicted', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Log Flux')
    plt.legend()
    plt.title(f'Full Spectrum (Age: {age:.2f} Gyr, Z: {metallicity:.2f})')
    plt.subplot(2, 1, 2)
    mask = (wavelengths >= 4000) & (wavelengths <= 5000)
    plt.plot(wavelengths[mask], true_spectrum[mask], label='True', alpha=0.7)
    plt.plot(wavelengths[mask], pred_spectrum[mask], label='Predicted', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Log Flux')
    plt.legend()
    plt.title('Zoomed Region (4000-5000 Å)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_distribution(metrics, save_dir):
    """Plot distributions of reconstruction metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(metrics['mse'], bins=50)
    axes[0].set_title('Mean Squared Error Distribution')
    axes[1].hist(metrics['mae'], bins=50)
    axes[1].set_title('Mean Absolute Error Distribution')
    axes[2].hist(metrics['snr'], bins=50)
    axes[2].set_title('Signal-to-Noise Ratio Distribution')
    axes[2].set_xlabel('SNR (dB)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/regressor_metrics_dist.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_vs_wavelength(true_spectra, pred_spectra, wavelengths, save_path):
    """
    Computes and plots the median absolute error and percentile ranges
    as a function of wavelength.
    """
    # Calculate the absolute error for each spectrum at each wavelength
    abs_error = np.abs(true_spectra - pred_spectra)
    
    # Compute the median and percentile statistics across the test set for each wavelength
    median_error = np.median(abs_error, axis=0)
    p16_error = np.percentile(abs_error, 16, axis=0)
    p84_error = np.percentile(abs_error, 84, axis=0)
    
    plt.figure(figsize=(14, 6))
    
    # Plot the median error
    plt.plot(wavelengths, median_error, label='Median Absolute Error', color='blue')
    
    # Plot the 1-sigma equivalent percentile range as a shaded region
    plt.fill_between(wavelengths, p16_error, p84_error, color='blue', alpha=0.3,
                     label='16th-84th Percentile Range')
    
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Absolute Error (Log Flux)')
    plt.title('Error Distribution vs. Wavelength')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    if len(sys.argv) != 5:
        print("Usage: python evaluate_regressor.py <grid_dir> <grid_name> <autoencoder_path> <regressor_path>")
        sys.exit(1)

    grid_dir, grid_name, autoencoder_path, regressor_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    
    _, _, test_dataset = load_data_regressor(grid_dir, grid_name, n_samples=int(1e2))
    
    print(f"Loading autoencoder from {autoencoder_path}...")
    autoencoder, autoencoder_state, norm_params = load_autoencoder(autoencoder_path)
    
    print(f"Loading regressor from {regressor_path}...")
    regressor, regressor_state = load_regressor(regressor_path)
    
    output_dir = 'figures/regressor_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    true_spectra, pred_spectra, ages, metallicities = [], [], [], []
    
    print(f"Evaluating model on {len(test_dataset)} test samples...")
    for i in tqdm(range(len(test_dataset))):
        norm_true_spectrum = test_dataset.spectra[i]
        norm_age = test_dataset.conditions[i, 0]
        norm_metallicity = test_dataset.conditions[i, 1]
        
        # Generate the predicted spectrum (which is normalized)
        norm_pred_spectrum_jax = generate_spectrum(
            regressor, regressor_state, autoencoder, autoencoder_state, norm_age, norm_metallicity
        )
        
        # Un-normalize for metrics and plotting
        true_spectrum = unnormalize_spectrum(norm_true_spectrum, norm_params)
        pred_spectrum = unnormalize_spectrum(np.asarray(norm_pred_spectrum_jax), norm_params)
        
        true_spectra.append(true_spectrum)
        pred_spectra.append(pred_spectrum)

        # Un-normalize physical parameters for plotting
        ages.append(test_dataset.unnormalize_age(norm_age))
        metallicities.append(test_dataset.unnormalize_metallicity(norm_metallicity))
    
    true_spectra = np.array(true_spectra)
    pred_spectra = np.array(pred_spectra)
    
    metrics = compute_metrics(true_spectra, pred_spectra)
    print("\nReconstruction Metrics Summary:")
    print(f"MSE: {np.mean(metrics['mse']):.4f} ± {np.std(metrics['mse']):.4f}")
    print(f"MAE: {np.mean(metrics['mae']):.4f} ± {np.std(metrics['mae']):.4f}")
    print(f"SNR: {np.mean(metrics['snr']):.2f} ± {np.std(metrics['snr']):.2f} dB")
    
    plot_metrics_distribution(metrics, output_dir)
    
    print("Plotting error vs wavelength distribution...")
    plot_error_vs_wavelength(true_spectra, pred_spectra, test_dataset.wavelength, f'{output_dir}/error_vs_wavelength.png')
    
    print("Saving example spectrum comparison plots...")
    for i in range(min(5, len(test_dataset))):
        plot_spectrum_comparison(
            true_spectra[i], pred_spectra[i], test_dataset.wavelength, ages[i], metallicities[i],
            f'{output_dir}/spectrum_comparison_{i+1}.png'
        )

if __name__ == "__main__":
    main() 