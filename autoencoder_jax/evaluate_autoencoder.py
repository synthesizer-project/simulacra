"""
Evaluate a trained autoencoder + normalization MLP and generate figures.

Description:
- Loads a bundled autoencoder and a normalization MLP, samples a dataset from
  the specified grid, reconstructs final log10 spectra, and computes metrics.
- Produces figures: metric distributions, example reconstructions, latent space
  PCA, normalization diagnostics, and fractional error vs. wavelength.

CLI:
  python evaluate_autoencoder.py <grid_dir> <grid_name>
                                 <autoencoder_model_path>
                                 <norm_mlp_model_path>

Arguments:
- grid_dir: Directory containing the spectral grid HDF5 files.
- grid_name: File name of the grid to load.
- autoencoder_model_path: Path to bundled AE msgpack produced by train_autoencoder.py.
- norm_mlp_model_path: Path to bundled NormalizationMLP msgpack produced by train_norm_mlp.py.

Outputs:
- Figures saved to figures/autoencoder_evaluation/.

Notes:
- Evaluates up to 2000 randomly chosen spectra for speed.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import os

from grids import SpectralDatasetSynthesizer
from train_autoencoder import load_model as load_autoencoder_model
from train_norm_mlp import load_norm_mlp_model


def compute_reconstruction_metrics(true_log_spectra, pred_log_spectra):
    """Compute various reconstruction metrics on log-spectra."""
    # Mean Squared Error
    mse = np.mean((true_log_spectra - pred_log_spectra) ** 2, axis=1)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(true_log_spectra - pred_log_spectra), axis=1)
    
    # Signal-to-Noise Ratio (calculated on log-spectra)
    signal_var = np.var(true_log_spectra, axis=1)
    noise_var = np.var(true_log_spectra - pred_log_spectra, axis=1)
    # Add a small epsilon to avoid division by zero
    snr = 10 * np.log10(signal_var / (noise_var + 1e-9))

    return {
        'mse': mse,
        'mae': mae,
        'snr': snr
    }


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
    plt.plot(wavelengths[mask], true_log_spectrum[mask], label='True', alpha_color='blue', alpha=0.7)
    plt.plot(wavelengths[mask], pred_log_spectrum[mask], label='Reconstructed', alpha_color='orange', alpha=0.7)
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
    
    title_text = (
        f'Final Reconstruction\n'
        f'True Mean: {true_mean.item():.2f}, Pred Mean: {pred_mean.item():.2f}\n'
        f'True Std: {true_std.item():.2f}, Pred Std: {pred_std.item():.2f}'
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    if len(sys.argv) != 5:
        print("Usage: python evaluate_autoencoder.py <grid_dir> <grid_name> <autoencoder_model_path> <norm_mlp_model_path>")
        sys.exit(1)
        
    grid_dir, grid_name, ae_model_path, mlp_model_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    
    # Create output directory
    output_dir = f'figures/autoencoder_evaluation/'
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset for evaluation
    N_samples = int(1e4)
    dataset = SpectralDatasetSynthesizer(
        grid_dir=grid_dir,
        grid_name=grid_name,
        num_samples=N_samples,
    )
    
    # Load models
    print(f"Loading autoencoder from {ae_model_path}...")
    ae_model, ae_state, _ = load_autoencoder_model(ae_model_path)
    ae_variables = {'params': ae_state.params, 'batch_stats': ae_state.batch_stats}
    
    print(f"Loading normalization MLP from {mlp_model_path}...")
    mlp_model, mlp_params = load_norm_mlp_model(mlp_model_path)
    mlp_variables = {'params': mlp_params}

    # --- Evaluation ---
    test_size = min(2000, len(dataset))
    test_indices = np.random.choice(len(dataset), test_size, replace=False)
    
    # Get ground truth data
    conditions = dataset.conditions[test_indices]
    norm_true_spectra = dataset.spectra[test_indices]
    true_means = dataset.true_spec_mean[test_indices]
    true_stds = dataset.true_spec_std[test_indices]
    ages = dataset.ages[test_indices]
    metallicities = dataset.metallicities[test_indices]
    
    print(f"Evaluating system on {test_size} spectra...")
    
    # 1. Get Autoencoder predictions for the normalized shape
    pred_norm_spectra = ae_model.apply(ae_variables, norm_true_spectra, training=False)
    latent_vectors = ae_model.apply(ae_variables, norm_true_spectra, method=ae_model.encode, training=False)
    
    # 2. Get MLP predictions for the normalization parameters
    pred_means, pred_stds = mlp_model.apply(mlp_variables, conditions)

    # 3. Combine predictions to get final log-flux spectra
    pred_log_spectra = (pred_norm_spectra * pred_stds) + pred_means
    true_log_spectra = (norm_true_spectra * true_stds) + true_means
    
    # --- Compute and Plot Metrics on Final Reconstruction ---
    print("Computing metrics on final reconstruction...")
    metrics = compute_reconstruction_metrics(true_log_spectra, pred_log_spectra)
    plot_metrics_distribution(metrics, output_dir)
    
    # --- Plot Example Reconstructions ---
    print("Plotting example reconstructions...")
    num_examples_to_plot = 5
    for i in range(num_examples_to_plot):
        # Plot the final combined reconstruction
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
    print("Plotting normalization MLP diagnostics...")
    plot_normalization_diagnostics(
        true_means,
        pred_means,
        true_stds,
        pred_stds,
        save_path=f'{output_dir}/normalization_diagnostics.png'
    )

    # --- Plot Fractional Error vs. Wavelength on Final Spectra ---
    print("Plotting fractional error of final reconstruction...")
    true_linear_spectra = 10**true_log_spectra
    pred_linear_spectra = 10**pred_log_spectra
    plot_fractional_error_vs_wavelength(
        true_linear_spectra,
        pred_linear_spectra,
        dataset.wavelength,
        save_path=f'{output_dir}/fractional_error_vs_wavelength.png'
    )

    print(f"\nEvaluation complete. Figures saved to {output_dir}")


if __name__ == '__main__':
    main() 
