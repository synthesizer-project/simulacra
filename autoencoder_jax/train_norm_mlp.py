"""
Train a normalization MLP to predict per-spectrum mean and std.

Description:
- Given normalized physical conditions (age, metallicity), trains a small MLP
  to predict the spectrum-level normalization parameters (mean, std) used to
  de/normalize log10 spectra for the autoencoder pipeline.
- Saves a bundled msgpack containing model hyperparameters and parameters.

CLI:
  python train_norm_mlp.py <grid_dir> <grid_name>
                           [--samples N] [--epochs E]
                           [--batch-size B] [--save PATH]

Arguments:
- grid_dir: Directory containing the spectral grid HDF5 files.
- grid_name: File name of the grid to load.
- --samples: Number of spectra to sample (default: 100000).
- --epochs: Number of training epochs (default: 100).
- --batch-size: Mini-batch size (default: 64).
- --save: Output path for the bundled model (default: models/norm_mlp.msgpack).

Outputs:
- Bundled normalization MLP at the provided --save path.

Notes:
- This model is consumed by evaluate_autoencoder.py to recover final
  log-flux spectra from the AE’s normalized outputs.
"""

import sys
sys.path.append('..')

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax import serialization
from tqdm import tqdm
from typing import Sequence, Dict
import optuna

from grids import SpectralDatasetSynthesizer

def save_model(model, params, model_path):
    """Saves the MLP model state and hyperparameters to a single file."""
    hyperparams = {
        'features': model.features
    }
    bundled_data = {
        'hyperparams': hyperparams,
        'params': params,
    }
    with open(model_path, 'wb') as f:
        f.write(serialization.to_bytes(bundled_data))

def load_norm_mlp_model(model_path: str) -> (nn.Module, Dict):
    """Loads a NormalizationMLP model and its parameters from a single file."""
    with open(model_path, 'rb') as f:
        bundled_data = serialization.from_bytes(None, f.read())
    
    hyperparams = bundled_data['hyperparams']
    params = bundled_data['params']
    
    # Robustly handle features that might be stored as lists or dicts,
    # and ensure they are integers. This prevents data type errors.
    features = hyperparams.get('features')
    if isinstance(features, dict):
        # Sort by key ('0', '1', ...) to maintain order
        sorted_features = sorted(features.items(), key=lambda item: int(item[0]))
        hyperparams['features'] = tuple(int(v) for k, v in sorted_features)
    elif isinstance(features, (list, tuple)):
        hyperparams['features'] = tuple(int(f) for f in features)
    
    model = NormalizationMLP(**hyperparams)
    return model, params

# === Model Architecture ===
class NormalizationMLP(nn.Module):
    """A simple MLP to predict spectrum mean and std from physical conditions."""
    features: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, x, training: bool = True):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        
        # Output two values: one for the mean, one for the std
        x = nn.Dense(2)(x)
        
        pred_mean = x[..., 0:1]
        
        # Ensure std is positive using a softplus activation
        pred_std = nn.softplus(x[..., 1:2])
        
        return pred_mean, pred_std

# === Training State and Functions ===
def create_train_state(rng, model, learning_rate, weight_decay=1e-4):
    """Creates initial training state."""
    params = model.init(rng, jnp.ones((1, 2)))['params']
    
    # Use inject_hyperparams to make the learning rate a dynamic parameter.
    optimizer = optax.inject_hyperparams(optax.adamw)(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Chain with gradient clipping.
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optimizer
    )
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    """Performs a single training step."""
    conditions, true_means, true_stds = batch
    
    def loss_fn(params):
        pred_means, pred_stds = state.apply_fn({'params': params}, conditions)
        
        mean_loss = jnp.mean((pred_means - true_means) ** 2)
        std_loss = jnp.mean((pred_stds - true_stds) ** 2)
        
        loss = mean_loss + std_loss
        return loss, (mean_loss, std_loss)
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (mean_loss, std_loss)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, {'total_loss': loss, 'mean_loss': mean_loss, 'std_loss': std_loss}

@jax.jit
def eval_step(state, batch):
    """Performs a single evaluation step."""
    conditions, true_means, true_stds = batch
    pred_means, pred_stds = state.apply_fn({'params': state.params}, conditions)
    
    mean_loss = jnp.mean((pred_means - true_means) ** 2)
    std_loss = jnp.mean((pred_stds - true_stds) ** 2)
    
    return mean_loss + std_loss

# === Main Training Loop ===
def train_and_evaluate(
    train_data, test_data, num_epochs, batch_size, learning_rate, rng,
    model=None, patience=10, lr_patience=3, lr_factor=0.5, min_lr=1e-7,
    min_delta=1e-5, weight_decay=1e-4, trial=None, verbose=True,
    save_path=None
):
    """Trains the MLP and evaluates it."""
    train_conditions, train_means, train_stds = train_data
    test_conditions, test_means, test_stds = test_data

    if model is None:
        model = NormalizationMLP()
    
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, learning_rate, weight_decay=weight_decay)

    best_test_loss = float('inf')
    best_params = None
    no_improve_epochs = 0
    lr_no_improve_epochs = 0
    current_lr = learning_rate
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        # --- Training Epoch ---
        train_ds_size = len(train_conditions)
        steps_per_epoch = train_ds_size // batch_size
        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
        
        train_loss_metrics = {'total_loss': [], 'mean_loss': [], 'std_loss': []}

        epoch_iterator = tqdm(perms, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=False) if verbose else perms
        for perm in epoch_iterator:
            batch_conditions = train_conditions[perm]
            batch_means = train_means[perm]
            batch_stds = train_stds[perm]
            batch = (batch_conditions, batch_means, batch_stds)
            
            state, metrics = train_step(state, batch)
            for k, v in metrics.items():
                train_loss_metrics[k].append(v)
        
        avg_train_loss = {k: np.mean(v) for k, v in train_loss_metrics.items()}
        train_losses.append(avg_train_loss['total_loss'])

        # --- Evaluation ---
        test_ds_size = len(test_conditions)
        steps_per_epoch_test = test_ds_size // batch_size
        epoch_test_losses = []
        for i in range(steps_per_epoch_test):
            start, end = i * batch_size, (i+1) * batch_size
            batch = (test_conditions[start:end], test_means[start:end], test_stds[start:end])
            epoch_test_losses.append(eval_step(state, batch))
        
        avg_test_loss = np.mean(epoch_test_losses)
        test_losses.append(avg_test_loss)
        
        if verbose:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss['total_loss']:.4f} "
                  f"(Mean: {avg_train_loss['mean_loss']:.4f}, Std: {avg_train_loss['std_loss']:.4f}) | "
                  f"Test Loss: {avg_test_loss:.4f} | LR: {current_lr:.2e}")
        
        # --- Early Stopping & LR Scheduler Logic ---
        if avg_test_loss < best_test_loss - min_delta:
            best_test_loss = float(avg_test_loss)
            best_params = state.params
            no_improve_epochs = 0
            lr_no_improve_epochs = 0
            if verbose:
                print(f"   -> New best test loss: {best_test_loss:.4f}")
        else:
            no_improve_epochs += 1
            lr_no_improve_epochs += 1
            if verbose:
                print(f"   -> No improvement for {no_improve_epochs} epochs.")
        
            if lr_no_improve_epochs >= lr_patience:
                new_lr = max(current_lr * lr_factor, min_lr)
                if new_lr < current_lr:
                    if verbose:
                        print(f"   -> Reducing learning rate from {current_lr:.2e} to {new_lr:.2e}")
                    current_lr = new_lr
                    # The `InjectHyperparamsState` is the second element (index 1) in the chain.
                    state.opt_state[1].hyperparams['learning_rate'] = current_lr
                lr_no_improve_epochs = 0
        
        if trial:
            trial.report(avg_test_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if no_improve_epochs >= patience:
            if verbose:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs. Best test loss: {best_test_loss:.4f}")
            break

    # --- Save Model ---
    if save_path and best_params is not None:
        save_model(model, best_params, save_path)
        if verbose:
            print(f"\nBest normalization MLP saved to {save_path}")
    elif save_path:
        if verbose:
            print("\nTraining finished without improvement. Model not saved.")
            
    return state, train_losses, test_losses, best_test_loss

def main():
    print(f"JAX devices: {jax.devices()}")
    
    # Basic CLI: python train_norm_mlp.py <grid_dir> <grid_name> [--samples N] [--epochs E] [--batch-size B] [--save PATH]
    if len(sys.argv) < 3:
        print("Usage: python train_norm_mlp.py <grid_dir> <grid_name> [--samples N] [--epochs E] [--batch-size B] [--save PATH]")
        sys.exit(1)
        
    grid_dir, grid_name = sys.argv[1], sys.argv[2]
    # Defaults
    N_samples = int(1e5)
    num_epochs = 100
    batch_size = 64
    save_path = 'models/norm_mlp.msgpack'

    # Parse optional args
    args = sys.argv[3:]
    def get_arg(flag, cast):
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args):
                return cast(args[i+1])
        return None
    N_samples = get_arg('--samples', int) or N_samples
    num_epochs = get_arg('--epochs', int) or num_epochs
    batch_size = get_arg('--batch-size', int) or batch_size
    save_path = get_arg('--save', str) or save_path

    # --- Load Data ---
    dataset = SpectralDatasetSynthesizer(grid_dir=grid_dir, grid_name=grid_name, num_samples=N_samples)
    rng = jax.random.PRNGKey(0)
    perm = jax.random.permutation(rng, len(dataset))
    split = int(0.8 * len(dataset))
    train_dataset = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[:split])
    test_dataset = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[split:])
    
    # --- Prefetch data to device (GPU/TPU) ---
    print("Prefetching data to the accelerator...")
    train_data = jax.device_put((
        jnp.array(train_dataset.conditions),
        jnp.array(train_dataset.true_spec_mean),
        jnp.array(train_dataset.true_spec_std)
    ))
    test_data = jax.device_put((
        jnp.array(test_dataset.conditions),
        jnp.array(test_dataset.true_spec_mean),
        jnp.array(test_dataset.true_spec_std)
    ))
    print("...done.")

    # --- Train ---
    train_and_evaluate(
        train_data=train_data,
        test_data=test_data,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        rng=rng,
        patience=10,
        lr_patience=4,
        lr_factor=0.5,
        min_delta=1e-5,
        weight_decay=1e-4,
        save_path=save_path
    )

if __name__ == "__main__":
    main() 
