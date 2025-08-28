import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os

from train_autoencoder import load_model as load_autoencoder
from train_regressor import load_regressor, load_data_regressor
from train_norm_mlp import load_norm_mlp_model


def unnormalize_spectrum(norm_spectrum, norm_params):
    """Un-normalizes a spectrum to log-space using the provided parameters."""
    if norm_params is None:
        return norm_spectrum
    mean = norm_params['spec_mean']
    std = norm_params['spec_std']
    return (norm_spectrum * std) + mean


def generate_spectrum(regressor, regressor_state, autoencoder, autoencoder_state, age, metallicity):
    """Generate spectrum(s) for given age(s) and metallicity(ies).
    
    Args:
        age: scalar or array of ages
        metallicity: scalar or array of metallicities
        
    Returns:
        spectrum: single spectrum if inputs are scalars, batch of spectra if inputs are arrays
    """
    # Handle both scalar and vector inputs
    age = jnp.atleast_1d(age)
    metallicity = jnp.atleast_1d(metallicity)
    conditions = jnp.stack([age, metallicity], axis=1)
    
    # Generate latent vectors using regressor
    regressor_variables = {'params': regressor_state['params']}
    latent = regressor.apply(regressor_variables, conditions, training=False)
    
    # Decode latent vectors to spectra using autoencoder
    autoencoder_variables = {'params': autoencoder_state.params, 'batch_stats': autoencoder_state.batch_stats}
    spectra = autoencoder.apply(autoencoder_variables, latent, method='decode', training=False)
    
    # Return single spectrum if original inputs were scalars, otherwise return batch
    if spectra.shape[0] == 1:
        return spectra[0]
    else:
        return spectra


def compute_metrics(true_spectra, pred_spectra):
    """Compute mean absolute fractional error on linear flux spectra."""
    # Convert log spectra to linear flux for meaningful metrics
    true_linear = 10**true_spectra
    pred_linear = 10**pred_spectra
    
    # Fractional error: (pred - true) / true
    fractional_error = (pred_linear - true_linear) / (true_linear + 1e-9)
    mean_abs_frac_error = np.mean(np.abs(fractional_error), axis=1)
    
    return {'mean_abs_frac_error': mean_abs_frac_error}


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
    """Plot distribution of mean absolute fractional error."""
    plt.figure(figsize=(8, 6))
    plt.hist(metrics['mean_abs_frac_error'] * 100, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Mean Absolute Fractional Error (%)')
    plt.ylabel('Count')
    plt.title('Mean Absolute Fractional Error Distribution\n(Computed on Linear Flux)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/regressor_metrics_dist.png', dpi=200, bbox_inches='tight')
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
    plt.xlim(wl_min, wl_max)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Manual CLI to allow optional --global-norm and --no-norm flags
    args = sys.argv[1:]
    use_global_norm = False
    use_no_norm = False
    if "--global-norm" in args:
        args.remove("--global-norm")
        use_global_norm = True
    if "--no-norm" in args:
        args.remove("--no-norm")
        use_no_norm = True

    if (use_no_norm and len(args) != 4) or ((not use_no_norm) and use_global_norm and len(args) != 4) or ((not use_no_norm) and (not use_global_norm) and len(args) != 5):
        print("Usage: python evaluate_regressor.py <grid_dir> <grid_name> <autoencoder_path> <regressor_path> [<norm_mlp_path>] [--global-norm] [--no-norm]")
        sys.exit(1)

    grid_dir, grid_name, autoencoder_path, regressor_path = args[0], args[1], args[2], args[3]
    norm_mlp_path = None if (use_global_norm or use_no_norm) else args[4]

    _, _, test_dataset = load_data_regressor(
        grid_dir, grid_name, n_samples=int(1e3),
        norm=(None if use_no_norm else ('global' if use_global_norm else 'per-spectra'))
    )

    print(f"Loading autoencoder from {autoencoder_path}...")
    autoencoder, autoencoder_state, _ = load_autoencoder(autoencoder_path)

    print(f"Loading regressor from {regressor_path}...")
    regressor, regressor_state = load_regressor(regressor_path)

    if use_no_norm:
        print("Using no normalization (direct log-space, no MLP).")
        norm_mlp, norm_mlp_params = None, None
    elif use_global_norm:
        print("Using global normalization (no MLP).")
        norm_mlp, norm_mlp_params = None, None
    else:
        print(f"Loading normalization MLP from {norm_mlp_path}...")
        norm_mlp, norm_mlp_params = load_norm_mlp_model(norm_mlp_path)

    output_dir = 'figures/regressor_evaluation'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Evaluating model on {len(test_dataset)} test samples...")

    # Vectorized evaluation - process all samples at once
    norm_ages = test_dataset.conditions[:, 0]
    norm_metallicities = test_dataset.conditions[:, 1]

    # Generate all predicted spectra in one batch
    norm_pred_spectra = generate_spectrum(
        regressor, regressor_state, autoencoder, autoencoder_state, norm_ages, norm_metallicities
    )

    # Compute final spectra depending on normalization strategy
    if use_no_norm:
        true_spectra = test_dataset.spectra
        pred_spectra = norm_pred_spectra
    else:
        # Predict or fetch normalization parameters
        if use_global_norm:
            if np.isscalar(test_dataset.true_spec_mean) or (np.asarray(test_dataset.true_spec_mean).ndim == 0):
                pred_means = np.full((len(test_dataset), 1), float(np.asarray(test_dataset.true_spec_mean)))
                pred_stds = np.full((len(test_dataset), 1), float(np.asarray(test_dataset.true_spec_std)))
            else:
                pred_means = test_dataset.true_spec_mean
                pred_stds = test_dataset.true_spec_std
        else:
            conditions = jnp.stack([norm_ages, norm_metallicities], axis=1)
            norm_mlp_variables = {'params': norm_mlp_params}
            pred_means, pred_stds = norm_mlp.apply(norm_mlp_variables, conditions)

        # True spectra: use dataset-provided normalization
        if np.isscalar(test_dataset.true_spec_mean) or (np.asarray(test_dataset.true_spec_mean).ndim == 0):
            test_true_mean = np.full((len(test_dataset), 1), float(np.asarray(test_dataset.true_spec_mean)))
            test_true_std = np.full((len(test_dataset), 1), float(np.asarray(test_dataset.true_spec_std)))
        else:
            test_true_mean = test_dataset.true_spec_mean
            test_true_std = test_dataset.true_spec_std
        true_spectra = (test_dataset.spectra * test_true_std) + test_true_mean
        pred_spectra = (norm_pred_spectra * pred_stds) + pred_means

    # Un-normalize physical parameters for plotting
    ages = np.array([test_dataset.unnormalize_age(age) for age in norm_ages])
    metallicities = np.array([test_dataset.unnormalize_metallicity(met) for met in norm_metallicities])

    metrics = compute_metrics(true_spectra, pred_spectra)
    print("\nReconstruction Metrics Summary:")
    print(f"Mean Absolute Fractional Error: {np.mean(metrics['mean_abs_frac_error']):.4f} ± {np.std(metrics['mean_abs_frac_error']):.4f} ({np.mean(metrics['mean_abs_frac_error'])*100:.2f}%)")
    print(f"Median Absolute Fractional Error: {np.median(metrics['mean_abs_frac_error'])*100:.2f}%")

    plot_metrics_distribution(metrics, output_dir)

    print("Plotting fractional error vs wavelength...")
    plot_fractional_error_vs_wavelength(true_spectra, pred_spectra, test_dataset.wavelength, f'{output_dir}/fractional_error_vs_wavelength.png')

    print("Saving example spectrum comparison plots...")
    for i in range(min(5, len(test_dataset))):
        plot_spectrum_comparison(
            true_spectra[i], pred_spectra[i], test_dataset.wavelength, ages[i], metallicities[i],
            f'{output_dir}/spectrum_comparison_{i+1}.png'
        )

if __name__ == "__main__":
    main() 
