import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from flax import serialization
from tqdm import tqdm

from autoencoder import SpectrumAutoencoder
from regressor import RegressorMLP
from grids import SpectralDatasetSynthesizer


def load_models(autoencoder_path, regressor_path, spectrum_dim, latent_dim):
    """Load trained autoencoder and regressor models."""
    # Load autoencoder
    autoencoder = SpectrumAutoencoder(
        spectrum_dim=spectrum_dim,
        latent_dim=latent_dim
    )
    with open(autoencoder_path, 'rb') as f:
        autoencoder_state = serialization.from_bytes(autoencoder, f.read())
    
    # Load regressor
    regressor = RegressorMLP(
        hidden_dims=[256, 512, 1024], # 512, 1024, 512, 256],
        latent_dim=latent_dim,
        dropout_rate=0.1
    )
    with open(regressor_path, 'rb') as f:
        regressor_state = serialization.from_bytes(regressor, f.read())
    
    return autoencoder, autoencoder_state, regressor, regressor_state


def generate_spectrum(regressor, regressor_state, autoencoder, autoencoder_state, age, metallicity, ages, metallicities):
    """Generate a spectrum for given age and metallicity.
    
    Args:
        regressor: Regressor model
        regressor_state: Regressor state dictionary
        autoencoder: Autoencoder model
        autoencoder_state: Autoencoder state dictionary
        age: Age in Gyr
        metallicity: Metallicity in Z/Z_sun
        ages: Array of ages
        metallicities: Array of metallicities
        
    Returns:
        Generated spectrum
    """
    # Normalize inputs
    norm_age = (age - ages.mean()) / ages.std()
    norm_met = (metallicity - metallicities.mean()) / metallicities.std()
    
    # Create input array
    conditions = jnp.array([[norm_age, norm_met]])
    
    # Generate latent vector using regressor
    variables = {'params': regressor_state['params'], 'batch_stats': regressor_state['batch_stats']}
    latent = regressor.apply(
        variables,
        conditions,
        training=False
    )
    
    # Decode latent vector to spectrum using autoencoder
    variables = {'params': autoencoder_state['params'], 'batch_stats': autoencoder_state['batch_stats']}
    spectrum = autoencoder.apply(
        variables,
        latent,
        method='decode',
        training=False
    )
    
    return spectrum[0]  # Remove batch dimension


def compute_metrics(true_spectra, pred_spectra):
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


def plot_spectrum_comparison(true_spectrum, pred_spectrum, wavelengths, age, metallicity, save_path):
    """Plot comparison of true and predicted spectra."""
    plt.figure(figsize=(12, 6))
    
    # Plot full spectrum
    plt.subplot(2, 1, 1)
    plt.plot(wavelengths, true_spectrum, label='True', alpha=0.7)
    plt.plot(wavelengths, pred_spectrum, label='Predicted', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.legend()
    plt.title(f'Full Spectrum (Age: {age:.2f} Gyr, Z: {metallicity:.2f})')
    
    # Plot zoomed region (e.g., around 4000-5000 Å)
    plt.subplot(2, 1, 2)
    mask = (wavelengths >= 4000) & (wavelengths <= 5000)
    plt.plot(wavelengths[mask], true_spectrum[mask], label='True', alpha=0.7)
    plt.plot(wavelengths[mask], pred_spectrum[mask], label='Predicted', alpha=0.7)
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
    plt.savefig(f'{save_dir}/regressor_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Load dataset
    grid_dir = '../../synthesizer_grids/grids/'
    dataset = SpectralDatasetSynthesizer(grid_dir=grid_dir, grid_name='bc03-2016-Miles_chabrier-0.1,100.hdf5')
    
    # Load models
    autoencoder, autoencoder_state, regressor, regressor_state = load_models(
        'models/best_autoencoder.msgpack',
        'models/best_regressor.msgpack',
        spectrum_dim=dataset.n_wavelength,
        latent_dim=128
    )
    
    # Evaluate on test set
    test_size = min(100, len(dataset))  # Limit to 1000 samples for evaluation
    test_indices = np.random.choice(len(dataset), test_size, replace=False)
    
    # Collect predictions and metrics
    true_spectra = []
    pred_spectra = []
    ages = []
    metallicities = []
    
    print("Evaluating model...")
    for idx in tqdm(test_indices):
        # Get true spectrum and parameters
        true_spectrum = dataset.spectra[idx]
        age = dataset.ages[idx]
        metallicity = dataset.metallicities[idx]
        
        # Generate predicted spectrum
        pred_spectrum = generate_spectrum(
            regressor,
            regressor_state,
            autoencoder,
            autoencoder_state,
            age,
            metallicity,
            dataset.ages,
            dataset.metallicities
        )
        
        true_spectra.append(true_spectrum)
        pred_spectra.append(pred_spectrum)
        ages.append(age)
        metallicities.append(metallicity)
    
    # Convert to numpy arrays
    true_spectra = np.array(true_spectra)
    pred_spectra = np.array(pred_spectra)
    ages = np.array(ages)
    metallicities = np.array(metallicities)
    
    # Compute metrics
    metrics = compute_metrics(true_spectra, pred_spectra)
    
    # Print summary statistics
    print("\nReconstruction Metrics Summary:")
    print(f"MSE: {np.mean(metrics['mse']):.4f} ± {np.std(metrics['mse']):.4f}")
    print(f"MAE: {np.mean(metrics['mae']):.4f} ± {np.std(metrics['mae']):.4f}")
    # print(f"MAPE: {np.mean(metrics['mape']):.2f}% ± {np.std(metrics['mape']):.2f}%")
    print(f"PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f} dB")
    
    # Plot metrics distributions
    plot_metrics_distribution(metrics, 'figures/regressor_evaluation')
    
    # Plot example reconstructions
    for i in range(5):  # Plot 5 examples
        plot_spectrum_comparison(
            true_spectra[i],
            pred_spectra[i],
            dataset.wavelength,
            ages[i],
            metallicities[i],
            f'figures/regressor_evaluation/spectrum_comparison_{i+1}.png'
        )


if __name__ == "__main__":
    main() 