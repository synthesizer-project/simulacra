import sys
sys.path.append('..')

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import subprocess

from grids import SpectralDatasetSynthesizer


def run_train_pca(grid_dir, grid_name, n_components, num_samples, pca_out_path, method):
    # Call the existing train_pca.py with given sample count
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), 'train_pca.py'),
        grid_dir, grid_name, str(n_components), str(num_samples), method
    ]
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)


def select_group(f, group_name=None):
    if group_name and group_name in f:
        return f[group_name]
    if 'latest_group' in f.attrs and f.attrs['latest_group'] in f:
        return f[f.attrs['latest_group']]
    groups = [k for k in f.keys() if isinstance(f.get(k, getclass=True), h5py.Group) and k.startswith('n_')]
    if groups:
        def parse_n(s):
            try:
                return int(s.split('_', 1)[1])
            except Exception:
                return -1
        gname = max(groups, key=parse_n)
        return f[gname]
    return f


def evaluate_group(grid_dir, grid_name, g, num_val=10000):
    pca_input_mean = g['pca_input_mean'][:]
    pca_components = g['pca_components'][:]
    wavelengths = g['wavelengths'][:]
    true_spec_mean = g.attrs['true_spec_mean']
    true_spec_std = g.attrs['true_spec_std']

    val_ds = SpectralDatasetSynthesizer(
        grid_dir=grid_dir,
        grid_name=grid_name,
        num_samples=num_val,
        norm='global',
        true_spec_mean=true_spec_mean,
        true_spec_std=true_spec_std,
    )

    spectra_demeaned = val_ds.spectra - pca_input_mean
    weights = spectra_demeaned @ pca_components.T
    recon_norm = weights @ pca_components + pca_input_mean

    recon_log = recon_norm * true_spec_std + true_spec_mean
    true_log = val_ds.spectra * true_spec_std + true_spec_mean

    recon_linear = 10 ** recon_log
    true_linear = 10 ** true_log

    abs_frac_err = np.abs((recon_linear - true_linear) / (true_linear + 1e-9))
    per_spec_mean_abs = np.mean(abs_frac_err, axis=1).squeeze()
    return {
        'median': float(np.median(per_spec_mean_abs)),
        'mean': float(np.mean(per_spec_mean_abs)),
        'p95': float(np.percentile(per_spec_mean_abs, 95)),
        'wavelengths': wavelengths.shape[0],
        'components': pca_components.shape[0],
    }


def plot_metrics(ns, means, medians, p95s, out_path):
    plt.figure(figsize=(8,6))
    plt.plot(ns, np.array(means)*100, 'o-', label='Mean')
    plt.plot(ns, np.array(medians)*100, 's-', label='Median')
    plt.plot(ns, np.array(p95s)*100, 'd-', label='95th percentile')
    plt.xlabel('Number of PCA training samples')
    plt.ylabel('Mean Abs. Fractional Error (%)')
    plt.title('PCA Reconstruction Error vs Training Samples (20 PCs)')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    if len(sys.argv) < 4:
        print('Usage: python benchmark_pca_training.py <grid_dir> <grid_name> <pca_output_path> [n_components] [sample_list or start:stop:step] [comp_range or start:stop:step] [comp_train_samples] [val_samples]')
        sys.exit(1)

    grid_dir = sys.argv[1]
    grid_name = sys.argv[2]
    pca_out_path = sys.argv[3]
    n_components = int(sys.argv[4]) if len(sys.argv) >= 5 else 20

    # Determine sample sizes
    if len(sys.argv) >= 6:
        spec = sys.argv[5]
        if ':' in spec:
            start, stop, step = map(int, spec.split(':'))
            sample_sizes = list(range(start, stop + 1, step))
        else:
            sample_sizes = [int(x) for x in spec.split(',')]
    else:
        sample_sizes = [1000, 2000, 5000, 10000]

    # Methods to compare
    methods = ['svd', 'pca']
    # Collect metrics per method
    metrics_by_method = {m: {} for m in methods}

    for method in methods:
        print(f'--- Training method: {method} ---')
        for ns in sample_sizes:
            run_train_pca(grid_dir, grid_name, n_components, ns, pca_out_path, method)
            with h5py.File(pca_out_path, 'r') as f:
                gname = f'{method}_n_{ns}'
                if gname not in f:
                    print(f'Error: Expected group {gname} not found in {pca_out_path}')
                    sys.exit(1)
                g = f[gname]
                metrics_by_method[method][ns] = evaluate_group(grid_dir, grid_name, g, num_val=10000)
                m = metrics_by_method[method][ns]
                print(f"{method} ns={ns}: median={m['median']*100:.3f}% mean={m['mean']*100:.3f}% p95={m['p95']*100:.3f}%")

    # Plot
    out_dir = os.path.join(os.path.dirname(pca_out_path), '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    # Comparison plot: PCA (solid) vs SVD (dashed)
    out_path = os.path.join(out_dir, f'pca_error_vs_samples_compare_{n_components}pc.png')
    plt.figure(figsize=(8,6))
    for metric_name, color in [('mean','C0'), ('median','C1'), ('p95','C2')]:
        for method, style in [('pca','-'), ('svd','--')]:
            ns_sorted = sorted(metrics_by_method[method].keys())
            ys = [metrics_by_method[method][n][metric_name] * 100 for n in ns_sorted]
            label = f"{metric_name.capitalize()} ({method.upper()})"
            plt.plot(ns_sorted, ys, linestyle=style, marker='o', color=color, label=label)
    plt.xlabel('Number of PCA training samples')
    plt.ylabel('Mean Abs. Fractional Error (%)')
    plt.title(f'PCA Reconstruction Error vs Training Samples ({n_components} PCs)')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'Benchmark comparison plot saved to {out_path}')

    # Optional: sweep number of PCA components for a fixed number of training samples
    comp_range = None
    comp_train_samples = 1000
    val_samples = 10000
    if len(sys.argv) >= 7:
        comp_spec = sys.argv[6]
        if ':' in comp_spec:
            cstart, cstop, cstep = map(int, comp_spec.split(':'))
            comp_range = list(range(cstart, cstop + 1, cstep))
        else:
            comp_range = [int(x) for x in comp_spec.split(',')]
    if len(sys.argv) >= 8:
        comp_train_samples = int(sys.argv[7])
    if len(sys.argv) >= 9:
        val_samples = int(sys.argv[8])

    if comp_range:
        print(f'Running PCA component sweep: N_train={comp_train_samples}, components={comp_range}, N_val={val_samples}')
        # Build training dataset with global normalization
        train_ds = SpectralDatasetSynthesizer(
            grid_dir=grid_dir,
            grid_name=grid_name,
            num_samples=comp_train_samples,
            norm='global',
        )
        # Cache normalization scalars for validation
        true_spec_mean = train_ds.true_spec_mean
        true_spec_std = train_ds.true_spec_std
        # PCA centering and eigendecomposition
        train_mean = np.mean(train_ds.spectra, axis=0)
        X = train_ds.spectra - train_mean
        cov = np.cov(X, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        components_full = evecs.T  # (D, W)

        # Validation dataset using same normalization
        val_ds = SpectralDatasetSynthesizer(
            grid_dir=grid_dir,
            grid_name=grid_name,
            num_samples=val_samples,
            norm='global',
            true_spec_mean=true_spec_mean,
            true_spec_std=true_spec_std,
        )
        comp_list, med_list, mean_list, p95_list = [], [], [], []
        for k in comp_range:
            comps = components_full[:k, :]
            # Project and reconstruct in normalized space
            w = (val_ds.spectra - train_mean) @ comps.T
            recon_norm = w @ comps + train_mean
            # Back to linear flux
            recon_log = recon_norm * true_spec_std + true_spec_mean
            true_log = val_ds.spectra * true_spec_std + true_spec_mean
            recon_linear = 10 ** recon_log
            true_linear = 10 ** true_log
            abs_frac_err = np.abs((recon_linear - true_linear) / (true_linear + 1e-9))
            per_spec_mean_abs = np.mean(abs_frac_err, axis=1).squeeze()
            comp_list.append(k)
            med_list.append(float(np.median(per_spec_mean_abs)))
            mean_list.append(float(np.mean(per_spec_mean_abs)))
            p95_list.append(float(np.percentile(per_spec_mean_abs, 95)))
            print(f'k={k}: median={med_list[-1]*100:.3f}% mean={mean_list[-1]*100:.3f}% p95={p95_list[-1]*100:.3f}%')

        # Plot accuracy vs number of components
        plt.figure(figsize=(8,6))
        plt.plot(comp_list, np.array(mean_list)*100, 'o-', label='Mean')
        plt.plot(comp_list, np.array(med_list)*100, 's-', label='Median')
        plt.plot(comp_list, np.array(p95_list)*100, 'd-', label='95th percentile')
        plt.xlabel('Number of PCA components')
        plt.ylabel('Mean Abs. Fractional Error (%)')
        plt.title(f'PCA Error vs Components (N_train={comp_train_samples})')
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()
        out_path2 = os.path.join(out_dir, f'pca_error_vs_components_ns_{comp_train_samples}.png')
        plt.savefig(out_path2, dpi=200)
        plt.close()
        print(f'Component sweep plot saved to {out_path2}')


if __name__ == '__main__':
    main()
