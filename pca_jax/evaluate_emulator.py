import sys
sys.path.append('..')

import numpy as np
import jax
from flax.training import train_state
import optax
import h5py
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple

from pca_jax.train_emulator import load_model as load_emulator_model
from grids import SpectralDatasetSynthesizer


def plot_reconstruction(
    wavelength, true_spectrum_linear, pred_spectrum_linear, 
    true_spectrum_log, pred_spectrum_log, condition, mean_frac_error, save_path
):
    """Plots the original and reconstructed spectra in both linear and log space."""
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    title = f'Emulator Reconstruction | Condition: {np.round(condition, 3)} | Mean Frac Error: {mean_frac_error:.2%}'
    fig.suptitle(title, fontsize=16)

    # Linear scale plot
    axs[0].plot(wavelength, true_spectrum_linear, label='True Spectrum (Linear Scale)', color='black', lw=1.5)
    axs[0].plot(wavelength, pred_spectrum_linear, label='Predicted Spectrum (Linear Scale)', color='red', linestyle='--', lw=1.5)
    axs[0].set_title('Linear Scale Comparison')
    axs[0].set_ylabel('Flux')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Log scale plot
    axs[1].plot(wavelength, true_spectrum_log, label='True Spectrum (Log10 Scale)', color='black', lw=1.5)
    axs[1].plot(wavelength, pred_spectrum_log, label='Predicted Spectrum (Log10 Scale)', color='red', linestyle='--', lw=1.5)
    axs[1].set_title('Log10 Scale Comparison')
    axs[1].set_ylabel('Log10(Flux)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Fractional error plot
    fractional_error = (pred_spectrum_linear - true_spectrum_linear) / (true_spectrum_linear + 1e-9)
    axs[2].plot(wavelength, fractional_error, label='Fractional Error', color='blue', lw=1)
    axs[2].axhline(0, color='grey', linestyle='--', lw=1)
    axs[2].set_title('Fractional Error: (Pred - True) / True')
    axs[2].set_xlabel('Wavelength (Angstrom)')
    axs[2].set_ylabel('Fractional Error')
    axs[2].set_ylim(-0.1, 0.1) # Zoom in on +/- 10% error
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path)
    plt.close()

def plot_error_distribution(errors, save_path):
    """Plots the distribution of the mean absolute fractional error."""
    plt.figure(figsize=(10, 6))
    plt.hist(errors * 100, bins=50, alpha=0.7, density=True)
    plt.xlabel('Mean Absolute Fractional Error (%)')
    plt.ylabel('Density')
    plt.title('Distribution of Reconstruction Errors')
    
    mean_err = np.mean(errors) * 100
    median_err = np.median(errors) * 100
    plt.axvline(mean_err, color='r', linestyle='--', label=f'Mean: {mean_err:.3f}%')
    plt.axvline(median_err, color='g', linestyle='--', label=f'Median: {median_err:.3f}%')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_fractional_error_vs_wavelength(true_spectra, pred_spectra, wavelengths, save_path, wl_min=2000.0, wl_max=10000.0, n_components=None, n_train_samples=None):
    """Computes and plots the median fractional error and percentile ranges vs. wavelength."""
    fractional_error = (pred_spectra - true_spectra) / (true_spectra + 1e-9)
    
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
    # Annotate PCA params if provided
    if (n_components is not None) or (n_train_samples is not None):
        parts = []
        if n_components is not None:
            parts.append(f'Components: {int(n_components)}')
        if n_train_samples is not None:
            parts.append(f'Train samples: {int(n_train_samples)}')
        if parts:
            ax = plt.gca()
            ax.text(0.98, 0.02, ' | '.join(parts), transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.xlim(wl_min, wl_max)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.savefig(save_path, dpi=300)
    plt.close()

# === Main Evaluation Function ===
@jax.jit
def predict_weights(state, conditions):
    """Predicts PCA weights from physical conditions."""
    return state.apply_fn({'params': state.params}, conditions)

if __name__ == '__main__':
    if len(sys.argv) not in (5,6):
        print("Usage: python evaluate_emulator.py <grid_dir> <grid_name> <pca_data_path> <emulator_model_path> [pca_group]")
        sys.exit(1)

    grid_dir, grid_name_arg, pca_data_path, emulator_model_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    pca_group = sys.argv[5] if len(sys.argv) == 6 else None
    
    # Define paths
    grid_name = os.path.splitext(grid_name_arg)[0] if grid_name_arg.endswith('.hdf5') else grid_name_arg
    output_dir = f'evaluation_plots/emulator_{grid_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading emulator from: {emulator_model_path}")
    model, params, bundle_pca_path, bundle_pca_group, bundle_whitened = load_emulator_model(emulator_model_path)

    # Prefer explicit CLI path/group; else fall back to bundle
    if pca_data_path is None:
        pca_data_path = bundle_pca_path
    if pca_group is None:
        pca_group = bundle_pca_group

    print(f"Loading PCA data from: {pca_data_path} | group: {pca_group}")
    if not os.path.exists(pca_data_path):
        print(f"Error: PCA data file not found at {pca_data_path}")
        sys.exit(1)
        
    def _select_pca_group(f, group_name=None):
        if group_name and group_name in f:
            return f[group_name]
        if 'latest_group' in f.attrs and f.attrs['latest_group'] in f:
            return f[f.attrs['latest_group']]
        groups = [k for k in f.keys() if isinstance(f.get(k, getclass=True), h5py.Group) and ('n_' in k)]
        if groups:
            import re
            def parse_n(s):
                m = re.search(r'n_(\d+)$', s)
                return int(m.group(1)) if m else -1
            gname = max(groups, key=parse_n)
            return f[gname]
        return f

    with h5py.File(pca_data_path, 'r') as f:
        g = _select_pca_group(f, pca_group)
        pca_input_mean = g['pca_input_mean'][:]
        pca_components = g['pca_components'][:]
        wavelengths = g['wavelengths'][:]
        eigenvalues = g['eigenvalues'][:]
        # Load normalization scalars from attributes
        true_spec_mean = g.attrs['true_spec_mean']
        true_spec_std = g.attrs['true_spec_std']
        used_group = g.name.split('/')[-1]

    # Parse training sample count from group name
    n_train_samples_meta = None
    if used_group:
        import re
        m = re.search(r'n_(\d+)$', used_group)
        if m:
            try:
                n_train_samples_meta = int(m.group(1))
            except Exception:
                n_train_samples_meta = None
        
    print(f"Loading test data from grid: {grid_name}")
    # Load a larger set for statistical evaluation
    test_dataset = SpectralDatasetSynthesizer(
        grid_dir=grid_dir,
        grid_name=grid_name_arg,
        num_samples=1000, 
        norm=False,
        # true_spec_mean=true_spec_mean,
        # true_spec_std=true_spec_std
    )
    
    # Create a dummy TrainState for inference
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))

    print("Running inference on the test set...")
    pred_weights_all = predict_weights(state, test_dataset.conditions)
    # If model trained on whitened weights, unwhiten with sqrt(eigenvalues)
    if bundle_whitened:
        eps = 1e-8
        sqrt_eigs = np.sqrt(eigenvalues + eps)
        pred_weights_all = np.asarray(pred_weights_all) * sqrt_eigs

    print("Reconstructing all spectra...")
    pred_normalized_spectra = (pred_weights_all @ pca_components) + pca_input_mean
    pred_log_spectra = pred_normalized_spectra * true_spec_std + true_spec_mean
    pred_linear_spectra = 10**pred_log_spectra
    
    true_log_spectra = test_dataset.spectra
    true_linear_spectra = 10**true_log_spectra

    print("Computing metrics...")
    # Apply wavelength selection for statistics and error plot
    wl_min, wl_max = 2000.0, 10000.0
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    wavelengths_sel = wavelengths[mask]
    pred_linear_stats = pred_linear_spectra[:, mask]
    true_linear_stats = true_linear_spectra[:, mask]
    abs_frac_error = np.abs((pred_linear_stats - true_linear_stats) / (true_linear_stats + 1e-9))
    mean_abs_frac_error = np.mean(abs_frac_error, axis=1).squeeze()

    # --- Print Report ---
    print("\n--- Emulator Performance Report ---")
    print(f"  Median Mean Fractional Error: {np.median(mean_abs_frac_error):.4%}")
    print(f"  Mean of Mean Fractional Error: {np.mean(mean_abs_frac_error):.4%}")
    print(f"  95th Percentile Error: {np.percentile(mean_abs_frac_error, 95):.4%}")
    print("-------------------------------------\n")

    # --- Generate Plots ---
    print("Generating diagnostic plots...")
    
    # Plot error distribution
    plot_error_distribution(mean_abs_frac_error, os.path.join(output_dir, 'error_distribution.png'))

    # Plot fractional error vs wavelength
    plot_fractional_error_vs_wavelength(
        true_linear_stats, pred_linear_stats, wavelengths_sel,
        os.path.join(output_dir, 'error_vs_wavelength.png'),
        wl_min=wl_min, wl_max=wl_max,
        n_components=pca_components.shape[0], n_train_samples=n_train_samples_meta
    )

    # Plot a few example reconstructions
    print("Plotting example reconstructions...")
    num_examples_to_plot = 5
    for i in range(num_examples_to_plot):
        plot_path = os.path.join(output_dir, f'reconstruction_example_{i}.png')
        plot_reconstruction(
            wavelength=wavelengths,
            true_spectrum_linear=true_linear_spectra[i],
            pred_spectrum_linear=pred_linear_spectra[i],
            true_spectrum_log=true_log_spectra[i],
            pred_spectrum_log=pred_log_spectra[i],
            condition=test_dataset.conditions[i],
            mean_frac_error=mean_abs_frac_error[i],
            save_path=plot_path
        )
        
    print(f"\nEvaluation complete. Plots saved to {output_dir}")



    
