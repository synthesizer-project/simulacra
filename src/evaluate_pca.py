import sys
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from grids import SpectralDatasetSynthesizer


def plot_fractional_error_vs_wavelength(true_spectra, recon_spectra, wavelengths, save_path, wl_min=2000.0, wl_max=10000.0, n_components=None, n_train_samples=None):
    """
    Plot median and percentile bands of fractional error vs wavelength.

    Fractional error is defined as (recon - true) / (true + 1e-9) in linear flux space.
    """
    frac_err = (recon_spectra - true_spectra) / (true_spectra + 1e-9)

    median = np.median(frac_err, axis=0)
    p16, p84 = np.percentile(frac_err, [16, 84], axis=0)
    p2_5, p97_5 = np.percentile(frac_err, [2.5, 97.5], axis=0)
    p0_15, p99_85 = np.percentile(frac_err, [0.15, 99.85], axis=0)

    plt.figure(figsize=(14, 7))
    # 3σ in grey, 2σ and 1σ in red
    plt.fill_between(wavelengths, p0_15 * 100, p99_85 * 100, color='grey', alpha=0.15, label='3σ (99.7%)')
    plt.fill_between(wavelengths, p2_5 * 100, p97_5 * 100, color='red', alpha=0.15, label='2σ (95%)')
    plt.fill_between(wavelengths, p16 * 100, p84 * 100, color='red', alpha=0.30, label='1σ (68%)')
    plt.plot(wavelengths, median * 100, color='black', lw=2, label='Median')
    # Zero and ±1% reference lines
    plt.axhline(0, color='black', linestyle='--', lw=1)
    plt.axhline(1, color='blue', linestyle='--', lw=1, label='+1%')
    plt.axhline(-1, color='blue', linestyle='--', lw=1, label='-1%')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Fractional Error (%)')
    plt.title('PCA Reconstruction Error vs. Wavelength')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Annotate PCA model parameters
    if (n_components is not None) or (n_train_samples is not None):
        label_parts = []
        if n_components is not None:
            label_parts.append(f'Components: {int(n_components)}')
        if n_train_samples is not None:
            label_parts.append(f'Train samples: {int(n_train_samples)}')
        if label_parts:
            anno = ' | '.join(label_parts)
            ax = plt.gca()
            ax.text(0.98, 0.02, anno, transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.xlim(wl_min, wl_max)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_error_distribution(errors, save_path):
    """Plot the distribution of mean absolute fractional error per spectrum."""
    plt.figure(figsize=(10, 6))
    plt.hist(errors * 100, bins=60, alpha=0.75, density=True)
    plt.xlabel('Mean Absolute Fractional Error (%)')
    plt.ylabel('Density')
    plt.title('Distribution of PCA Reconstruction Errors')
    mean_err = np.mean(errors) * 100
    median_err = np.median(errors) * 100
    p95 = np.percentile(errors, 95) * 100
    plt.axvline(mean_err, color='r', linestyle='--', label=f'Mean: {mean_err:.3f}%')
    plt.axvline(median_err, color='g', linestyle='--', label=f'Median: {median_err:.3f}%')
    plt.axvline(p95, color='b', linestyle='--', label=f'95th: {p95:.3f}%')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_reconstruction_examples(true_log_spectra, recon_log_spectra, wavelengths, 
                                conditions, save_dir, n_examples=5, wl_min=2000.0, wl_max=10000.0):
    """Plot individual reconstruction examples.
    
    Args:
        true_log_spectra: Ground truth log-flux spectra
        recon_log_spectra: Reconstructed log-flux spectra
        wavelengths: Wavelength array
        conditions: Physical conditions (age, metallicity) - should be denormalized
        save_dir: Directory to save plots
        n_examples: Number of examples to plot
        wl_min: Minimum wavelength for plotting
        wl_max: Maximum wavelength for plotting
    """
    print(f"Plotting {n_examples} PCA reconstruction examples...")
    
    # Apply wavelength mask for plotting
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    wavelengths_plot = wavelengths[mask]
    true_plot = true_log_spectra[:, mask]
    recon_plot = recon_log_spectra[:, mask]
    
    # Select random examples
    n_spectra = len(true_log_spectra)
    indices = np.random.choice(n_spectra, min(n_examples, n_spectra), replace=False)
    
    for i, idx in enumerate(indices):
        plt.figure(figsize=(14, 10))
        
        # Full spectrum comparison (log space)
        plt.subplot(3, 1, 1)
        plt.plot(wavelengths_plot, true_plot[idx], label='True', alpha=0.8, linewidth=1.5, color='blue')
        plt.plot(wavelengths_plot, recon_plot[idx], label='PCA Reconstruction', alpha=0.8, linewidth=1.5, 
                linestyle='--', color='red')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Log Flux')
        plt.legend()
        
        # Extract physical conditions for title
        if len(conditions[idx]) >= 2:
            age, metallicity = conditions[idx, 0], conditions[idx, 1]
            plt.title(f'PCA Spectrum Reconstruction - Age={age:.2f} Gyr, Z={metallicity:.4f}')
        else:
            plt.title(f'PCA Spectrum Reconstruction - Example {i+1}')
        plt.grid(True, alpha=0.3)
        plt.xlim(wl_min, wl_max)
        
        # Residual in log space
        plt.subplot(3, 1, 2)
        residual_log = recon_plot[idx] - true_plot[idx]
        plt.plot(wavelengths_plot, residual_log, color='green', alpha=0.7, linewidth=1.5)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Residual (Log)')
        plt.title('Residual: Reconstructed - True (Log Space)')
        plt.grid(True, alpha=0.3)
        plt.xlim(wl_min, wl_max)
        
        # Fractional error in linear space
        plt.subplot(3, 1, 3)
        true_linear = 10**true_plot[idx]
        recon_linear = 10**recon_plot[idx]
        frac_error = (recon_linear - true_linear) / (true_linear + 1e-9)
        plt.plot(wavelengths_plot, frac_error * 100, color='red', alpha=0.7, linewidth=1.5)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(1, color='blue', linestyle=':', alpha=0.7, label='±1%')
        plt.axhline(-1, color='blue', linestyle=':', alpha=0.7)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Fractional Error (%)')
        plt.title('Fractional Error in Linear Flux')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(wl_min, wl_max)
        
        # Compute and display error statistics for this spectrum
        mean_abs_frac_error = np.mean(np.abs(frac_error)) * 100
        rms_frac_error = np.sqrt(np.mean(frac_error**2)) * 100
        plt.text(0.02, 0.95, f'Mean |Error|: {mean_abs_frac_error:.3f}%\nRMS Error: {rms_frac_error:.3f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'pca_reconstruction_example_{i}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {n_examples} reconstruction examples to {save_dir}")


def _select_pca_group(f, group_name=None):
    if group_name and group_name in f:
        return f[group_name], group_name
    # Auto-select: prefer attr 'latest_group', else max n_*
    if 'latest_group' in f.attrs and f.attrs['latest_group'] in f:
        gname = f.attrs['latest_group']
        return f[gname], gname
    # Accept both legacy 'n_*' and namespaced '<method>_n_*'
    groups = [k for k in f.keys() if isinstance(f.get(k, getclass=True), h5py.Group) and ('n_' in k)]
    if groups:
        import re
        def parse_n(s):
            m = re.search(r'n_(\d+)$', s)
            return int(m.group(1)) if m else -1
        gname = max(groups, key=parse_n)
        return f[gname], gname
    # Fallback to root (back-compat)
    return f, None


def main():
    if len(sys.argv) not in (4, 5, 6):
        print('Usage: python evaluate_pca.py <grid_dir> <grid_name> <pca_data_path> [num_samples] [pca_group]')
        sys.exit(1)

    grid_dir = sys.argv[1]
    grid_name_arg = sys.argv[2]
    pca_data_path = sys.argv[3]
    # num_samples may be 4th or 5th arg depending on presence of pca_group
    num_samples = 5000
    pca_group = None
    if len(sys.argv) >= 5:
        try:
            num_samples = int(sys.argv[4])
            pca_group = sys.argv[5] if len(sys.argv) == 6 else None
        except ValueError:
            # If the 4th arg is not an int, treat it as pca_group
            pca_group = sys.argv[4]

    grid_name = os.path.splitext(grid_name_arg)[0] if grid_name_arg.endswith('.hdf5') else grid_name_arg

    # Output directory
    out_dir = os.path.join('figures/pca_evaluation', f'pca_{grid_name}')
    os.makedirs(out_dir, exist_ok=True)

    # Load PCA data
    print(f'Loading PCA data from: {pca_data_path}')
    if not os.path.exists(pca_data_path):
        print(f'Error: PCA data file not found at {pca_data_path}')
        sys.exit(1)

    with h5py.File(pca_data_path, 'r') as f:
        g, used_group = _select_pca_group(f, pca_group)
        pca_input_mean = g['pca_input_mean'][:]
        pca_components = g['pca_components'][:]
        wavelengths = g['wavelengths'][:]
        sigma_lambda = g['sigma_lambda'][:] if 'sigma_lambda' in g else None
        true_spec_mean = g.attrs['true_spec_mean'] if 'true_spec_mean' in g.attrs else None
        true_spec_std = g.attrs['true_spec_std'] if 'true_spec_std' in g.attrs else None
        if used_group is None:
            print('Warning: using legacy root datasets (no group).')
    # Parse training sample count from group name if available
    n_train_samples = None
    if used_group is not None:
        import re
        m = re.search(r'n_(\d+)$', used_group)
        if m:
            try:
                n_train_samples = int(m.group(1))
            except Exception:
                n_train_samples = None

    print('Building validation dataset...')
    # Choose dataset normalization based on presence of per-wavelength z-score params
    if sigma_lambda is not None:
        # Work directly in log space for z-score PCA
        val_ds = SpectralDatasetSynthesizer(
            grid_dir=grid_dir,
            grid_name=grid_name_arg,
            num_samples=num_samples,
            norm=False,
        )
    else:
        # Use the same global normalization scalars used during PCA training
        val_ds = SpectralDatasetSynthesizer(
            grid_dir=grid_dir,
            grid_name=grid_name_arg,
            num_samples=num_samples,
            norm='global',
            true_spec_mean=true_spec_mean,
            true_spec_std=true_spec_std,
        )

    # Restrict to wavelength range (default: 2000–10000 Å)
    wl_min, wl_max = 2000.0, 10000.0
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    wavelengths_sel = wavelengths[mask]

    # Project normalized spectra onto PCA basis and reconstruct
    # val_ds.spectra are normalized already: (log10 F - mean) / std
    if sigma_lambda is not None:
        # val_ds.spectra are raw log10 fluxes; apply same global normalization used in training
        if true_spec_mean is None or true_spec_std is None:
            raise ValueError("Missing global normalization scalars for z-score PCA group.")
        Xg = (val_ds.spectra - true_spec_mean) / true_spec_std
        z = (Xg - pca_input_mean) / sigma_lambda
        weights = z @ pca_components.T
        zhat = weights @ pca_components
        recon_norm = zhat * sigma_lambda + pca_input_mean
        recon_log = recon_norm * true_spec_std + true_spec_mean
        true_log = val_ds.spectra
    else:
        spectra_demeaned = val_ds.spectra - pca_input_mean
        weights = spectra_demeaned @ pca_components.T
        recon_norm = weights @ pca_components + pca_input_mean
        # Convert both recon and true to log space
        recon_log = recon_norm * true_spec_std + true_spec_mean
        true_log = val_ds.spectra * true_spec_std + true_spec_mean

    # Convert both recon and true to linear flux for fractional errors
    recon_linear = 10 ** recon_log
    true_linear = 10 ** true_log

    # Apply wavelength selection to spectra
    recon_linear = recon_linear[:, mask]
    true_linear = true_linear[:, mask]

    # Metrics
    abs_frac_err = np.abs((recon_linear - true_linear) / (true_linear + 1e-9))
    per_spec_mean_abs = np.mean(abs_frac_err, axis=1).squeeze()

    print('\n--- PCA Reconstruction Report ---')
    print(f'  Validation samples: {len(val_ds)}')
    print(f'  Wavelength bins: {len(wavelengths)}')
    print(f'  PCA components: {pca_components.shape[0]}')
    print(f'  Median mean fractional error: {np.median(per_spec_mean_abs):.4%}')
    print(f'  Mean of mean fractional error: {np.mean(per_spec_mean_abs):.4%}')
    print(f'  95th percentile error: {np.percentile(per_spec_mean_abs, 95):.4%}')
    print('--------------------------------\n')

    # Plots
    print('Generating plots...')
    plot_error_distribution(per_spec_mean_abs, os.path.join(out_dir, 'pca_error_distribution.png'))
    plot_fractional_error_vs_wavelength(
        true_linear, recon_linear, wavelengths_sel,
        os.path.join(out_dir, 'pca_error_vs_wavelength.png'),
        wl_min=wl_min, wl_max=wl_max,
        n_components=pca_components.shape[0], n_train_samples=n_train_samples
    )
    
    # Plot individual reconstruction examples
    print('Generating reconstruction examples...')
    # Need to get denormalized conditions for proper display
    # The val_ds.conditions are normalized, so we need to denormalize them
    if hasattr(val_ds, 'age_mean') and hasattr(val_ds, 'age_std') and hasattr(val_ds, 'met_mean') and hasattr(val_ds, 'met_std'):
        denormalized_conditions = np.copy(val_ds.conditions)
        denormalized_conditions[:, 0] = val_ds.conditions[:, 0] * val_ds.age_std + val_ds.age_mean  # Age
        denormalized_conditions[:, 1] = val_ds.conditions[:, 1] * val_ds.met_std + val_ds.met_mean  # Metallicity
    else:
        # Fallback to normalized conditions if denormalization parameters not available
        denormalized_conditions = val_ds.conditions
        print("Warning: Using normalized conditions for plot titles (denormalization parameters not available)")
    
    plot_reconstruction_examples(
        true_log_spectra=true_log,
        recon_log_spectra=recon_log,
        wavelengths=wavelengths,
        conditions=denormalized_conditions,
        save_dir=out_dir,
        n_examples=5,
        wl_min=wl_min,
        wl_max=wl_max
    )
    
    print(f'All plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
