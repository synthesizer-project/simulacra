"""
Train a PCA basis for spectra and save it to HDF5.

Overview
--------
This script samples synthetic spectra using `SpectralDatasetSynthesizer`,
applies a chosen normalization scheme, and fits a PCA/SVD basis to the
normalized log10 spectra. It saves the PCA components, per-wavelength
centering vector, eigenvalues, the wavelength grid, and the normalization
scalars required to invert normalization during reconstruction.

Key points
----------
- Methods:
  - "pca": covariance-based PCA on normalized spectra
  - "svd": SVD-based PCA on normalized spectra (default)
  - "pca_z" / "svd_z": use an additional per-wavelength z-score stage
    (dataset_norm set to "zscore").
- Normalization:
  - "global" mode stores scalar `true_spec_mean` and `true_spec_std` over
    the log-spectra; PCA is performed on spectra further centered by a
    per-wavelength mean (`pca_input_mean`).
  - "zscore" mode stores per-wavelength `lambda_mean`/`lambda_std` as datasets
    (`sigma_lambda` is saved as the std), and PCA is performed in that space.
- Wavelengths:
  - You can constrain the wavelength range by passing optional `wl_min` and
    `wl_max` (Å). The saved HDF5 includes the exact `wavelengths` array used.
- Outputs:
  - HDF5 at `pca_models/pca_model_<grid_name>.h5` containing a group named
    `<method>_n_<N>` where `N` is the number of training samples.
  - The group contains datasets: `pca_components`, `pca_input_mean`,
    `eigenvalues`, `wavelengths`, and (if z-score) `sigma_lambda`.
  - Attributes include: `true_spec_mean`, `true_spec_std`, and `method`.
  - A plot of the first components is saved to
    `pca_models/pca_components_<grid_name>.png`.

Usage
-----
Basic (SVD, global normalization):
  python src/train_pca.py <grid_dir> <grid_name.hdf5> <n_components> [num_samples] [method]

Examples:
  # SVD with 20 PCs, 2000 samples, global norm, 2000–10000 Å
  python src/train_pca.py /path/to/grids bc03-2003-padova00_chabrier03-0.1,100.hdf5 20 2000 svd 2000 10000

  # Covariance PCA with 30 PCs, 5000 samples, z-score norm
  python src/train_pca.py /path/to/grids bc03-2003-padova00_chabrier03-0.1,100.hdf5 30 5000 pca_z

Integration
-----------
- The unified pipeline (`train_unified_regressor.py` / `evaluate_unified_regressor.py`)
  can load this HDF5 directly. Ensure wavelength limits and normalization
  settings are consistent between PCA training and regressor training/eval.
- When using whitening in the regressor pipeline, do so consistently across
  training and evaluation.
"""

import sys
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

from grids import SpectralDatasetSynthesizer

def plot_pca_components(wavelengths, pca_components, output_path):
    """Plots the first few PCA components."""
    n_plot = min(8, pca_components.shape[0])  # Plot up to 8 components
    
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2 * n_plot), sharex=True)
    fig.suptitle('PCA Basis Components', fontsize=16)

    for i in range(n_plot):
        axes[i].plot(wavelengths, pca_components[i, :], lw=2)
        axes[i].set_ylabel(f'Component {i+1}')
        axes[i].grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel('Wavelength (Angstrom)')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path)
    plt.close()

def main():
    if len(sys.argv) not in (4, 5, 6, 8):
        print("Usage: python train_pca.py <grid_dir> <grid_name> <n_components> [num_samples] [method] [wl_min] [wl_max]")
        sys.exit(1)

    grid_dir, grid_name_arg, n_components = sys.argv[1], sys.argv[2], int(sys.argv[3])
    num_samples = int(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[4].isdigit() else int(1e3)
    method = None
    wl_min = None
    wl_max = None
    if len(sys.argv) >= 7:
        try:
            wl_min = float(sys.argv[-2])
            wl_max = float(sys.argv[-1])
        except Exception:
            wl_min = None
            wl_max = None
    if len(sys.argv) == 6:
        method = sys.argv[5]
    elif len(sys.argv) == 5 and not sys.argv[4].isdigit():
        method = sys.argv[4]
    method = (method or 'svd').strip().lower()
    allowed = ('svd', 'pca', 'svd_z', 'pca_z')
    if method not in allowed:
        raise ValueError(f"method must be one of {allowed}, got {method}")
    # Handle if user provides full filename for output path consistency
    grid_name = os.path.splitext(grid_name_arg)[0] if grid_name_arg.endswith('.hdf5') else grid_name_arg
    
    output_dir = 'pca_models'
    os.makedirs(output_dir, exist_ok=True)

    # choose model at runtime
    if method in ('svd', 'svd_z'):
        model = 'SVD'
    else:
        model = 'PCA'

    # Use a consistent name for the output file based on the grid name
    output_path_model = f'{output_dir}/pca_model_{grid_name}.h5'
    output_path_plot = f'{output_dir}/pca_components_{grid_name}.png'

    print("Loading training data for PCA...")
    # Note: We pass the original grid_name_arg to synthesizer
    dataset_norm = 'zscore' if method.endswith('_z') else 'global'
    training_dataset = SpectralDatasetSynthesizer(
        grid_dir=grid_dir,
        grid_name=grid_name_arg,
        num_samples=num_samples,
        norm=dataset_norm,
        wl_min=wl_min,
        wl_max=wl_max,
        compute_lambda_stats=method.endswith('_z')
    )
    training_spectra = training_dataset.spectra
    
    # Prepare data matrix for PCA/SVD
    use_z = method.endswith('_z')
    if use_z:
        # Spectra are already z-scored by dataset using per-wavelength stats on log spectra
        spectra_proc = training_spectra
        sigma_lambda = np.array(training_dataset.lambda_std)
        pca_input_mean = np.array(training_dataset.lambda_mean)
    else:
        # Global-normalized spectra: center per wavelength
        pca_input_mean = np.mean(training_spectra, axis=0)
        spectra_proc = training_spectra - pca_input_mean
        sigma_lambda = None

    if model == 'PCA':
        print("Performing PCA on normalized spectra...")
        # The spectra are already normalized, center/standardize as requested
        spectra_demeaned = spectra_proc

        # Compute covariance matrix and its eigendecomposition
        print("Computing covariance matrix...")
        cov_matrix = np.cov(spectra_demeaned, rowvar=False)
        
        print("Computing eigenvalues and eigenvectors...")
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort components by explained variance (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Transpose eigenvectors to get components for projection (n_components, n_features)
        pca_components_full = sorted_eigenvectors.T

        # Truncate to the specified number of components
        print(f"Truncating to {n_components} principal components.")
        pca_components = pca_components_full[:n_components, :]
        eigenvalues_truncated = sorted_eigenvalues[:n_components]
    elif model == 'SVD':
        print("Performing SVD on normalized spectra...")
        # Use centered/standardized matrix
        spectra_demeaned = spectra_proc

        # SVD-based PCA implementation
        print("Computing SVD...")
        # Perform SVD on the demeaned data matrix
        # training_spectra shape: (n_samples, n_features)
        # U: (n_samples, min(n_samples, n_features))
        # s: (min(n_samples, n_features),)
        # Vt: (min(n_samples, n_features), n_features)
        U, s, Vt = np.linalg.svd(spectra_demeaned, full_matrices=False)

        # The principal components are the rows of Vt (or columns of V)
        # These are already sorted by explained variance (descending)
        pca_components_full = Vt  # Shape: (n_components_max, n_features)

        # Compute explained variance (eigenvalues) from singular values
        # For SVD: eigenvalues = (singular_values^2) / (n_samples - 1)
        n_samples = spectra_demeaned.shape[0]
        eigenvalues_full = (s**2) / (n_samples - 1)

        # Truncate to the specified number of components
        print(f"Truncating to {n_components} principal components.")
        pca_components = pca_components_full[:n_components, :]
        eigenvalues_truncated = eigenvalues_full[:n_components]
        # For reporting
        sorted_eigenvalues = eigenvalues_full

    # --- Plot the components ---
    print(f"Plotting PCA components and saving to {output_path_plot}...")
    plot_pca_components(training_dataset.wavelength, pca_components, output_path_plot)

    # --- Save PCA model to HDF5 ---
    print(f"Saving PCA model and normalization data to {output_path_model}...")
    with h5py.File(output_path_model, 'a') as f:
        group_name = f"{method}_n_{training_dataset.spectra.shape[0]}"
        if group_name in f:
            del f[group_name]
        g = f.create_group(group_name)
        g.create_dataset('pca_input_mean', data=pca_input_mean)
        g.create_dataset('pca_components', data=pca_components)
        g.create_dataset('eigenvalues', data=eigenvalues_truncated)
        g.create_dataset('wavelengths', data=training_dataset.wavelength)
        # Z-score parameters if used
        if sigma_lambda is not None:
            g.create_dataset('sigma_lambda', data=sigma_lambda)
        # Store normalization scalars as attributes on the group (if available)
        if training_dataset.true_spec_mean is not None and training_dataset.true_spec_std is not None:
            g.attrs['true_spec_mean'] = np.array(training_dataset.true_spec_mean)
            g.attrs['true_spec_std'] = np.array(training_dataset.true_spec_std)
        g.attrs['method'] = method
        # Optionally set/overwrite a pointer to the latest group
        f.attrs['latest_group'] = group_name

    print("\nPCA training complete.")
    # Report variance based on the full set of components to give context
    explained_variance_ratio_full = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cumulative_variance_full = np.cumsum(explained_variance_ratio_full)

    if n_components <= len(cumulative_variance_full):
      print(f"\nExplained Variance with {n_components} components: {cumulative_variance_full[n_components-1]:.6f}")

    print("\nCumulative Explained Variance (overall):")
    n_components_to_show = [1, 5, 10, 20, 30, 40, 50, 100, 150]
    for n in n_components_to_show:
        if n <= len(cumulative_variance_full):
            print(f"  {n:>3} components: {cumulative_variance_full[n-1]:.6f}")

if __name__ == '__main__':
    main() 
