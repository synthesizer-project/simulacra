"""
Diagnostic tool for unified regressor: true vs predicted latents and per-component R^2.

Usage:
  python src/diagnose_unified_regressor.py <method> <latent_model_path> <regressor_path> <grid_dir> <grid_name>
      [--samples N] [--seed S] [--batch-size B] [--save-dir DIR]
      [--pca-group G] [--no-whitening] [--wl-min A] [--wl-max B] [--plot-components K]

Generates:
  - per_component_r2.png: Bar plot of R^2 across latent dimensions
  - latent_scatter_component_{i}.png: Scatter plots for up to K components
"""

import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_unified_regressor import (
    create_latent_representation,
    load_data_unified,
    prepare_training_data,
)
from train_regressor import load_regressor


def predict_latents(model, params, conditions: jnp.ndarray, batch_size: int = 128) -> jnp.ndarray:
    preds = []
    for i in range(0, len(conditions), batch_size):
        batch_cond = conditions[i:i + batch_size]
        lat = model.apply({'params': params}, batch_cond, training=False)
        preds.append(lat)
    return jnp.concatenate(preds) if preds else jnp.array([])


def per_component_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # y_* shape: (N, D)
    y_true_c = y_true - y_true.mean(axis=0, keepdims=True)
    ss_tot = (y_true_c ** 2).sum(axis=0)
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1.0 - (ss_res / ss_tot)
    # Handle zero-variance components
    r2 = np.where(ss_tot <= 1e-12, np.nan, r2)
    return r2


def plot_r2_bars(r2: np.ndarray, save_path: str):
    plt.figure(figsize=(10, 4))
    x = np.arange(len(r2))
    plt.bar(x, r2, color='#1f77b4', edgecolor='black', linewidth=0.5)
    plt.xlabel('Latent Component')
    plt.ylabel('R^2 (true vs predicted)')
    plt.ylim(-0.5, 1.05)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_latent_scatter(y_true: np.ndarray, y_pred: np.ndarray, comp_idx: int, save_path: str):
    plt.figure(figsize=(5, 5))
    t = y_true[:, comp_idx]
    p = y_pred[:, comp_idx]
    lim = np.percentile(np.abs(np.concatenate([t, p])), 99.5)
    lim = max(lim, 1e-3)
    plt.scatter(t, p, s=6, alpha=0.5, edgecolors='none')
    plt.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
    plt.xlabel(f'True latent c{comp_idx}')
    plt.ylabel(f'Pred latent c{comp_idx}')
    plt.title(f'Component {comp_idx}')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description='Diagnose unified regressor (latents)')
    p.add_argument('method', choices=['autoencoder', 'pca'])
    p.add_argument('latent_model_path')
    p.add_argument('regressor_path')
    p.add_argument('grid_dir')
    p.add_argument('grid_name')
    p.add_argument('--samples', type=int, default=4000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--save-dir', type=str, default='figures/regressor_evaluation')
    p.add_argument('--plot-components', type=int, default=6, help='Number of components to scatter plot')
    # PCA
    p.add_argument('--pca-group', type=str, default=None)
    p.add_argument('--no-whitening', action='store_true')
    # Wavelengths
    p.add_argument('--wl-min', type=float, default=None)
    p.add_argument('--wl-max', type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"JAX devices: {jax.devices()}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Create latent representation
    latent_kwargs = {}
    if args.method == 'pca':
        latent_kwargs['pca_group'] = args.pca_group
        latent_kwargs['whitened'] = not args.no_whitening
    latent_repr = create_latent_representation(args.method, args.latent_model_path, **latent_kwargs)

    # Determine wavelengths from model if not provided
    latent_norm = latent_repr.get_normalization_params()
    model_wl = latent_norm.get('wavelength') if latent_norm else None
    wl_min = args.wl_min if args.wl_min is not None else (float(model_wl.min()) if model_wl is not None else None)
    wl_max = args.wl_max if args.wl_max is not None else (float(model_wl.max()) if model_wl is not None else None)

    # Load datasets (use test split for diagnostics)
    train_ds, val_ds, test_ds = load_data_unified(
        grid_dir=args.grid_dir,
        grid_name=args.grid_name,
        n_samples=args.samples,
        latent_repr=latent_repr,
        seed=args.seed,
        wl_min=wl_min,
        wl_max=wl_max,
    )
    eval_ds = test_ds

    # Compute true latents on eval set
    print("Encoding true latents on evaluation set...")
    true_lat = prepare_training_data(latent_repr, eval_ds, batch_size=args.batch_size)

    # Load regressor and predict latents
    print(f"Loading regressor from {args.regressor_path}")
    regressor_model, regressor_state = load_regressor(args.regressor_path)
    pred_lat = predict_latents(regressor_model, regressor_state['params'], jnp.array(eval_ds.conditions), batch_size=args.batch_size)

    # Convert to numpy
    true_lat = np.array(true_lat)
    pred_lat = np.array(pred_lat)
    print(f"Latent shapes | true: {true_lat.shape} pred: {pred_lat.shape}")

    # Compute R^2 per component
    r2 = per_component_r2(true_lat, pred_lat)
    print(f"R^2 summary | mean={np.nanmean(r2):.4f}, median={np.nanmedian(r2):.4f}, min={np.nanmin(r2):.4f}")
    plot_r2_bars(r2, os.path.join(args.save_dir, 'per_component_r2.png'))

    # Scatter plots for selected components (first K)
    k = min(args.plot_components, true_lat.shape[1])
    for i in range(k):
        outp = os.path.join(args.save_dir, f'latent_scatter_component_{i}.png')
        plot_latent_scatter(true_lat, pred_lat, i, outp)

    print(f"Diagnostics saved to {args.save_dir}")


if __name__ == '__main__':
    main()

