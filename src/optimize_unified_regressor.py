"""
Hyperparameter optimization for the unified regressor using Optuna.

This script tunes the MLP that maps (age, metallicity) → latent space for
both autoencoder and PCA latent methods using the unified training pipeline.

Example (Autoencoder):
  python src/optimize_unified_regressor.py \
      autoencoder src/models/autoencoder_simple_dense.msgpack \
      /path/to/grids bc03-2003-padova00_chabrier03-0.1,100.hdf5 \
      --samples 20000 --epochs 200 --trials 40

Example (PCA):
  python src/optimize_unified_regressor.py \
      pca src/pca_models/pca_model_bc03-2003-padova00_chabrier03-0.1,100.h5 \
      /path/to/grids bc03-2003-padova00_chabrier03-0.1,100.hdf5 \
      --samples 20000 --epochs 200 --trials 40
"""

import argparse
import os
from typing import Tuple

import jax
import optuna

from train_unified_regressor import (
    create_latent_representation,
    load_data_unified,
    prepare_training_data,
    train_and_evaluate_unified,
)


def parse_args():
    p = argparse.ArgumentParser(description="Optuna optimization for unified regressor")

    # Positional
    p.add_argument("method", choices=["autoencoder", "pca"], help="Latent method")
    p.add_argument("model_path", type=str, help="Path to AE .msgpack or PCA .h5 model")
    p.add_argument("grid_dir", type=str, help="Directory containing spectral grids")
    p.add_argument("grid_name", type=str, help="Grid filename (e.g., bc03-...hdf5)")

    # Data/training
    p.add_argument("--samples", type=int, default=20000, help="Total samples (train+val+test)")
    p.add_argument("--epochs", type=int, default=200, help="Max training epochs per trial")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--trials", type=int, default=40, help="Number of Optuna trials")
    p.add_argument("--study-name", type=str, default="unified_regressor_opt", help="Optuna study name")
    p.add_argument("--storage", type=str, default=None, help="Optuna storage URL (optional)")
    p.add_argument("--save-best", action="store_true", help="Save best model params at end")
    p.add_argument("--save-dir", type=str, default="models", help="Directory to save best model")
    p.add_argument("--wl-min", type=float, default=None, help="Minimum wavelength (Å) for dataset")
    p.add_argument("--wl-max", type=float, default=None, help="Maximum wavelength (Å) for dataset")

    # PCA-specific
    p.add_argument("--pca-group", type=str, default=None, help="PCA group name (default: latest)")
    p.add_argument("--no-whitening", action="store_true", help="Disable PCA whitening")

    return p.parse_args()


def suggest_hyperparams(trial: optuna.Trial) -> Tuple[Tuple[int, ...], str, float, float, float, int, int]:
    # Depth and hidden sizes
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dims = []
    prev_max = 1024
    for i in range(n_layers):
        # Encourage tapering but allow flexibility
        lo = 64
        hi = prev_max
        size = trial.suggest_int(f"hidden_{i}", lo, hi, step=64)
        hidden_dims.append(size)
        prev_max = max(64, size)  # next layer not allowed to exceed previous too much

    activation = trial.suggest_categorical("activation", ["relu", "parametric_gated"]) 
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    patience = trial.suggest_int("patience", 10, 40)

    return tuple(hidden_dims), activation, dropout, lr, weight_decay, batch_size, patience


def main():
    args = parse_args()
    print(f"JAX devices: {jax.devices()}")
    print(f"Optimizing unified regressor for method={args.method}")

    latent_kwargs = {}
    if args.method == "pca":
        latent_kwargs["pca_group"] = args.pca_group
        latent_kwargs["whitened"] = not args.no_whitening

    # Create latent representation (AE or PCA)
    latent_repr = create_latent_representation(
        method=args.method, model_path=args.model_path, **latent_kwargs
    )
    print(f"Loaded latent representation. Latent dim = {latent_repr.get_latent_dim()}")

    # Load datasets once (consistent normalization derived from latent_repr)
    print("Loading datasets and preparing latent targets (once)...")
    train_ds, val_ds, test_ds = load_data_unified(
        grid_dir=args.grid_dir,
        grid_name=args.grid_name,
        n_samples=args.samples,
        latent_repr=latent_repr,
        seed=args.seed,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
    )
    train_latent = prepare_training_data(latent_repr, train_ds)
    val_latent = prepare_training_data(latent_repr, val_ds)
    print(f"Train latent shape: {train_latent.shape} | Val latent shape: {val_latent.shape}")

    # Objective uses fixed datasets/latents; only MLP and optimizer hyperparams vary
    def objective(trial: optuna.Trial):
        hidden_dims, activation, dropout, lr, weight_decay, batch_size, patience = suggest_hyperparams(trial)

        results = train_and_evaluate_unified(
            latent_repr=latent_repr,
            train_dataset=train_ds,
            val_dataset=val_ds,
            train_latent=train_latent,
            val_latent=val_latent,
            num_epochs=args.epochs,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            hidden_dims=hidden_dims,
            activation_name=activation,
            dropout_rate=dropout,
            rng_seed=args.seed,
            patience=patience,
            save_path=None,  # Do not save per-trial
            verbose=False,
        )

        best_val = float(results["best_val_loss"])  # minimize
        trial.set_user_attr("hidden_dims", hidden_dims)
        return best_val

    # Create study
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="minimize",
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="minimize")

    print(f"Starting optimization: {args.trials} trials")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Report best
    print("\nBest trial:")
    best = study.best_trial
    print(f"  Value (best val loss): {best.value:.6f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    if best.user_attrs:
        for k, v in best.user_attrs.items():
            print(f"    {k}: {v}")

    # Optionally retrain and save best model
    if args.save_best:
        os.makedirs(args.save_dir, exist_ok=True)
        hidden_dims, activation, dropout, lr, weight_decay, batch_size, patience = (
            tuple(best.user_attrs.get("hidden_dims", ())),
            best.params.get("activation", "parametric_gated"),
            float(best.params.get("dropout", 0.1)),
            float(best.params.get("learning_rate", 1e-3)),
            float(best.params.get("weight_decay", 1e-5)),
            int(best.params.get("batch_size", 128)),
            int(best.params.get("patience", 20)),
        )

        # Fallback if hidden_dims not set in user_attrs (older Optuna)
        if not hidden_dims:
            # Reconstruct in order from params
            n_layers = int(best.params.get("n_layers", 2))
            hidden_dims = tuple(int(best.params[f"hidden_{i}"]) for i in range(n_layers))

        grid_basename = os.path.splitext(args.grid_name)[0]
        out_path = os.path.join(
            args.save_dir, f"unified_regressor_{args.method}_{grid_basename}_best.msgpack"
        )
        print(f"\nRetraining best configuration and saving to {out_path} ...")
        _ = train_and_evaluate_unified(
            latent_repr=latent_repr,
            train_dataset=train_ds,
            val_dataset=val_ds,
            train_latent=train_latent,
            val_latent=val_latent,
            num_epochs=args.epochs,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            hidden_dims=hidden_dims,
            activation_name=activation,
            dropout_rate=dropout,
            rng_seed=args.seed,
            patience=patience,
            save_path=out_path,
            verbose=True,
        )
        print("Saved best regressor model.")


if __name__ == "__main__":
    main()
