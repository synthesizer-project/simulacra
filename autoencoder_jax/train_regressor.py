import sys
sys.path.append('..')

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
from flax import serialization
import matplotlib.pyplot as plt
from typing import Sequence, Callable
from flax.training import train_state as flax_train_state
import optuna

from grids import SpectralDatasetSynthesizer
from train_autoencoder import load_model as load_autoencoder
from activations import ParametricGatedActivation


class RegressorMLP(nn.Module):
    """MLP regressor that maps parameters to latent space."""
    hidden_dims: Sequence[int]
    latent_dim: int
    dropout_rate: float = 0.1
    activation_name: str = 'relu'
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        """Forward pass of the MLP.
        
        Args:
            x: Input parameters (age, metallicity) of shape (batch_size, 2)
            training: Whether the model is in training mode
            
        Returns:
            Predicted latent vector of shape (batch_size, latent_dim)
        """
        # Initial dense layer
        x = nn.Dense(self.hidden_dims[0])(x)
        if self.activation_name == 'relu':
            x = nn.relu(x)
        elif self.activation_name == 'parametric_gated':
            x = ParametricGatedActivation(features=self.hidden_dims[0])(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Hidden layers
        for hidden_dim in self.hidden_dims[1:]:
            x = nn.Dense(hidden_dim)(x)
            if self.activation_name == 'relu':
                x = nn.relu(x)
            elif self.activation_name == 'parametric_gated':
                x = ParametricGatedActivation(features=hidden_dim)(x)
            else:
                raise ValueError(f"Unsupported activation: {self.activation_name}")
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output layer
        x = nn.Dense(self.latent_dim)(x)
        return x


class TrainState(flax_train_state.TrainState):
    # This is a simple TrainState for the regressor without batch_stats
    pass


def create_train_state(rng, model, input_shape, learning_rate, weight_decay):
    """Creates initial regressor training state."""
    variables = model.init(rng, jnp.ones(input_shape))
    params = variables['params']
    
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=learning_rate, warmup_steps=1000,
        decay_steps=10000, end_value=learning_rate * 1e-3
    )
    
    tx = optax.adamw(learning_rate=scheduler, weight_decay=weight_decay)
    
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch, rng):
    """Performs a single training step for the regressor."""
    dropout_rng = jax.random.fold_in(rng, state.step)
        
    def loss_fn(params):
        pred_latent = state.apply_fn(
            {'params': params}, batch['conditions'], training=True, rngs={'dropout': dropout_rng}
        )
        loss = jnp.mean((pred_latent - batch['latent']) ** 2)
        return loss, pred_latent
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred_latent), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    
    metrics = {
        'loss': loss,
        'mse': jnp.mean((pred_latent - batch['latent']) ** 2),
        'mae': jnp.mean(jnp.abs(pred_latent - batch['latent']))
    }
    
    return state, metrics


@jax.jit
def eval_step(state, batch):
    """Performs a single evaluation step for the regressor."""
    pred_latent = state.apply_fn({'params': state.params}, batch['conditions'], training=False)
    metrics = {
        'mse': jnp.mean((pred_latent - batch['latent']) ** 2),
        'mae': jnp.mean(jnp.abs(pred_latent - batch['latent']))
    }
    return metrics


def save_regressor(model, params, model_path):
    """Saves the regressor params and hyperparameters to a single file."""
    hyperparams = {
        'hidden_dims': model.hidden_dims,
        'latent_dim': model.latent_dim,
        'dropout_rate': model.dropout_rate,
        'activation_name': getattr(model, 'activation_name', 'relu'),
    }
    bundled_data = {
        'hyperparams': hyperparams,
        'params': params
    }
    with open(model_path, 'wb') as f:
        f.write(serialization.to_bytes(bundled_data))


def load_regressor(model_path):
    """Loads a regressor model and its params from a single file."""
    with open(model_path, 'rb') as f:
        bundled_data = serialization.from_bytes(None, f.read())
    
    hyperparams = bundled_data['hyperparams']
    # Ensure hidden_dims are integers
    hidden_dims = hyperparams['hidden_dims']
    if isinstance(hidden_dims, dict):
        sorted_dims = sorted(hidden_dims.items(), key=lambda item: int(item[0]))
        hyperparams['hidden_dims'] = tuple(int(v) for k, v in sorted_dims)
    elif isinstance(hidden_dims, (list, tuple)):
        hyperparams['hidden_dims'] = tuple(int(d) for d in hidden_dims)
    
    params = bundled_data['params']
    
    # Create model instance with the loaded hyperparameters
    model = RegressorMLP(**hyperparams)
    
    # For inference, we only need the parameters
    state = {'params': params}
    return model, state


def train_and_evaluate(
    model, train_dataset, val_dataset, train_latent, val_latent, num_epochs,
    batch_size, learning_rate, weight_decay, rng, patience=20, min_delta=1e-4,
    trial=None, verbose=True, save_path=None
):
    """Train and evaluate the regressor model."""
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, input_shape=(1, train_dataset.conditions.shape[1]), learning_rate=learning_rate, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_metrics_history, val_metrics_history = [], []
    best_params = None
    
    for epoch in range(num_epochs):
        # Training
        epoch_train_metrics = []
        perm = jax.random.permutation(rng, len(train_dataset))
        for i in range(0, len(train_dataset), batch_size):
            batch_indices = perm[i:i + batch_size]
            batch = {'conditions': train_dataset.conditions[batch_indices], 'latent': train_latent[batch_indices]}
            rng, step_rng = jax.random.split(rng)
            state, metrics = train_step(state, batch, step_rng)
            epoch_train_metrics.append(metrics)
        
        # Validation
        epoch_val_metrics = []
        for i in range(0, len(val_dataset), batch_size):
            batch = {'conditions': val_dataset.conditions[i:i + batch_size], 'latent': val_latent[i:i + batch_size]}
            metrics = eval_step(state, batch)
            epoch_val_metrics.append(metrics)
        
        # Aggregate and log metrics
        train_epoch_metrics = {k: np.mean([m[k] for m in epoch_train_metrics]) for k in epoch_train_metrics[0]}
        val_epoch_metrics = {k: np.mean([m[k] for m in epoch_val_metrics]) for k in epoch_val_metrics[0]}
        train_metrics_history.append(train_epoch_metrics)
        val_metrics_history.append(val_epoch_metrics)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_epoch_metrics['loss']:.4f} | Val MSE: {val_epoch_metrics['mse']:.4f}")
        
        # Early stopping & Optuna pruning
        val_loss = val_epoch_metrics['mse']
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_params = state.params
        else:
            patience_counter += 1
            
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}...")
            break
    
    if save_path and best_params is not None:
        save_regressor(model, best_params, save_path)
        if verbose:
            print(f"Best regressor model saved to {save_path}")
            
    return state, train_metrics_history, val_metrics_history, best_val_loss


def plot_training_history(train_metrics, val_metrics, save_path):
    """Plot training history metrics.
    
    Args:
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_metrics) + 1)
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, np.log10([m['loss'] for m in train_metrics]), label='Train Loss')
    # plt.plot(epochs, np.log10([m['loss'] for m in val_metrics]), label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log10(Loss)')
    plt.legend()
    
    # Plot MSE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, np.log10([m['mse'] for m in train_metrics]), label='Train MSE')
    plt.plot(epochs, np.log10([m['mse'] for m in val_metrics]), label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_latent_representations(autoencoder, autoencoder_state, dataset, batch_size=64):
    """Create latent representations for a dataset using a trained autoencoder."""
    latent_vectors = []
    
    @jax.jit
    def encode_batch(spectra):
        variables = {'params': autoencoder_state.params, 'batch_stats': autoencoder_state.batch_stats}
        return autoencoder.apply(variables, spectra, method='encode', training=False)
    
    for i in range(0, len(dataset.spectra), batch_size):
        batch_spectra = dataset.spectra[i:i + batch_size]
        latent_batch = encode_batch(batch_spectra)
        latent_vectors.append(latent_batch)
    return jnp.concatenate(latent_vectors)


def load_data_regressor(grid_dir, grid_name, n_samples, norm: str = 'per-spectra'):
    """Load and preprocess data for the regressor.

    Args:
        grid_dir: Directory of spectral grids
        grid_name: Grid filename
        n_samples: Number of samples to draw
        norm: Normalization mode for spectra ('per-spectra', 'global', or 'zscore')
    """
    dataset = SpectralDatasetSynthesizer(grid_dir=grid_dir, grid_name=grid_name, num_samples=n_samples, norm=norm)
    perm = jax.random.permutation(jax.random.PRNGKey(0), len(dataset))
    train_size, val_size = int(0.7 * len(dataset)), int(0.15 * len(dataset))
    train_ds = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[:train_size])
    val_ds = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[train_size:train_size + val_size])
    test_ds = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[train_size + val_size:])
    return train_ds, val_ds, test_ds


def main():
    print(f"JAX devices: {jax.devices()}")

    # Parse CLI: python train_regressor.py <grid_dir> <grid_name> [--global-norm] [--no-norm] [--samples N] [--epochs E] [--batch-size B]
    args = sys.argv[1:]
    use_global_norm = False
    if "--global-norm" in args:
        args.remove("--global-norm")
        use_global_norm = True
    use_no_norm = False
    if "--no-norm" in args:
        args.remove("--no-norm")
        use_no_norm = True
    if len(args) < 2:
        print("Usage: python train_regressor.py <grid_dir> <grid_name> [--global-norm] [--no-norm] [--samples N] [--epochs E] [--batch-size B]")
        sys.exit(1)

    # Load the trained autoencoder
    autoencoder_path = 'models/autoencoder_simple_dense.msgpack'
    # load_model returns (model, state, norm_params). We don't need norm_params here.
    autoencoder, autoencoder_state, _ = load_autoencoder(autoencoder_path)
    latent_dim = autoencoder.latent_dim
    
    # Load the dataset
    N_samples = int(1e4)
    grid_dir, grid_name = args[0], args[1]

    # Optional args
    def get_arg(flag, cast):
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args):
                return cast(args[i+1])
        return None
    def get_flag(flag):
        return flag in args
    N_samples = get_arg('--samples', int) or N_samples
    num_epochs = get_arg('--epochs', int) or 200
    batch_size = get_arg('--batch-size', int) or 128
    if use_no_norm:
        norm_mode = None
        norm_label = 'none'
    else:
        norm_mode = 'global' if use_global_norm else 'per-spectra'
        norm_label = norm_mode
    print(f"Normalization mode: {norm_label}")
    train_dataset, val_dataset, test_dataset = load_data_regressor(grid_dir, grid_name, N_samples, norm=norm_mode)
    rng = jax.random.PRNGKey(0)

    # Create latent representations
    train_latent = create_latent_representations(autoencoder, autoencoder_state, train_dataset)
    val_latent = create_latent_representations(autoencoder, autoencoder_state, val_dataset)
    
    # Initialize the regressor model
    model = RegressorMLP(
        hidden_dims=(512, 256),
        latent_dim=latent_dim,
        dropout_rate=0.1,
        activation_name='parametric_gated'
    )
    
    # Train and evaluate the model
    print("Training regressor...")
    state, train_history, val_history, best_val_loss = train_and_evaluate(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_latent=train_latent,
        val_latent=val_latent,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        weight_decay=1e-5,
        rng=rng,
        save_path='models/best_regressor.msgpack'
    )
    
    plot_training_history(train_history, val_history, 'figures/regressor_training_history.png')
    print("Regressor training complete.")


if __name__ == "__main__":
    main() 
