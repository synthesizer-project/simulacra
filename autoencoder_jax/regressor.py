import sys
sys.path.append('..')

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
from flax import serialization
import matplotlib.pyplot as plt
from typing import Sequence, Callable, Dict, Tuple

from grids import SpectralDatasetSynthesizer
from autoencoder import SpectrumAutoencoder, TrainState


class RegressorMLP(nn.Module):
    """MLP regressor that maps parameters to latent space."""
    hidden_dims: Sequence[int]
    latent_dim: int
    activation: Callable = nn.relu
    dropout_rate: float = 0.1
    
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
        x = self.activation(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        # Hidden layers
        for hidden_dim in self.hidden_dims[1:]:
            x = nn.Dense(hidden_dim)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        # Output layer
        x = nn.Dense(self.latent_dim)(x)
        return x


def load_autoencoder(model_path, spectrum_dim, latent_dim):
    """Load a trained autoencoder model."""
    # Create model instance
    model = SpectrumAutoencoder(
        spectrum_dim=spectrum_dim,
        latent_dim=latent_dim
    )
    
    # Load state
    with open(model_path, 'rb') as f:
        state_dict = serialization.from_bytes(model, f.read())
    
    return model, state_dict


def create_train_state(rng, model, input_shape, learning_rate):
    """Creates initial training state."""
    # Initialize parameters
    variables = model.init(rng, jnp.ones(input_shape))
    
    # Create optimizer with learning rate scheduler
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=1000,
        decay_steps=10000,
        end_value=learning_rate * 1e-3
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=scheduler,
            weight_decay=1e-4,
            b1=0.9,
            b2=0.999,
            eps=1e-8
        )
    )
    
    # Create training state
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables.get('batch_stats', {}),
        tx=tx
    ), scheduler


@jax.jit
def train_step(state, batch, rng):
    """Performs a single training step."""
    def loss_fn(params, rng):
        # Split RNG key for dropout
        dropout_rng = jax.random.fold_in(rng, state.step)
        
        pred_latent = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['conditions'],
            training=True,
            rngs={'dropout': dropout_rng}
        )
        loss = jnp.mean((pred_latent - batch['latent']) ** 2)
        return loss, pred_latent
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred_latent), grads = grad_fn(state.params, rng)
    
    # Update optimizer state
    updates, new_opt_state = state.tx.update(
        grads, state.opt_state, state.params
    )
    
    # Apply updates to parameters
    new_params = optax.apply_updates(state.params, updates)
    
    # Create new state
    state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state
    )
    
    metrics = {
        'loss': loss,
        'mse': jnp.mean((pred_latent - batch['latent']) ** 2),
        'mae': jnp.mean(jnp.abs(pred_latent - batch['latent']))
    }
    
    return state, metrics


@jax.jit
def eval_step(state, batch):
    """Performs a single evaluation step."""
    pred_latent = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch['conditions'],
        training=False,
        rngs=None  # No RNG needed for evaluation
    )
    
    metrics = {
        'mse': jnp.mean((pred_latent - batch['latent']) ** 2),
        'mae': jnp.mean(jnp.abs(pred_latent - batch['latent']))
    }
    
    return metrics


def train_and_evaluate(
    model: RegressorMLP,
    train_dataset: SpectralDatasetSynthesizer,
    val_dataset: SpectralDatasetSynthesizer,
    train_latent: jnp.ndarray,
    val_latent: jnp.ndarray,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 20,
    min_delta: float = 1e-4,
    rng: jnp.ndarray = None
) -> Tuple[TrainState, Dict[str, list], Dict[str, list]]:
    """Train and evaluate the regressor model.
    
    Args:
        model: The regressor model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        train_latent: Latent vectors for training data
        val_latent: Latent vectors for validation data
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        patience: Number of epochs to wait for improvement before early stopping
        min_delta: Minimum change in validation loss to be considered an improvement
        rng: Random number generator key
        
    Returns:
        Tuple containing:
        - Final model state
        - Training metrics history
        - Validation metrics history
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    # Initialize training state
    rng, init_rng = jax.random.split(rng)
    state, scheduler = create_train_state(
        init_rng,
        model,
        input_shape=(1, 2),  # (batch_size, n_parameters)
        learning_rate=learning_rate
    )
    
    # Initialize metrics tracking
    best_val_loss = float('inf')
    train_metrics = {'loss': [], 'mse': [], 'mae': []}
    val_metrics = {'mse': [], 'mae': []}
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        train_losses = []
        train_mses = []
        train_maes = []
        
        # Shuffle training data
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, len(train_dataset))
        
        for i in range(0, len(train_dataset), batch_size):
            batch_indices = perm[i:i + batch_size]
            batch = {
                'conditions': train_dataset.conditions[batch_indices],
                'latent': train_latent[batch_indices]
            }
            
            rng, step_rng = jax.random.split(rng)
            state, metrics = train_step(state, batch, step_rng)
            train_losses.append(metrics['loss'])
            train_mses.append(metrics['mse'])
            train_maes.append(metrics['mae'])
        
        # Validation
        val_mses = []
        val_maes = []
        
        for i in range(0, len(val_dataset), batch_size):
            batch = {
                'conditions': val_dataset.conditions[i:i + batch_size],
                'latent': val_latent[i:i + batch_size]
            }
            
            metrics = eval_step(state, batch)
            val_mses.append(metrics['mse'])
            val_maes.append(metrics['mae'])
        
        # Compute epoch metrics
        train_metrics['loss'].append(np.mean(train_losses))
        train_metrics['mse'].append(np.mean(train_mses))
        train_metrics['mae'].append(np.mean(train_maes))
        val_metrics['mse'].append(np.mean(val_mses))
        val_metrics['mae'].append(np.mean(val_maes))
        
        # Get current learning rate from scheduler
        current_lr = scheduler(state.step)
        
        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_metrics['loss'][-1]:.4f}, LR: {current_lr:.2e}")
        # print(f"Train MSE: {train_metrics['mse'][-1]:.4f}, MAE: {train_metrics['mae'][-1]:.4f}")
        # print(f"Val MSE: {val_metrics['mse'][-1]:.4f}, MAE: {val_metrics['mae'][-1]:.4f}")
        
        # Early stopping check
        if val_metrics['mse'][-1] < best_val_loss - min_delta:
            best_val_loss = val_metrics['mse'][-1]
            patience_counter = 0
            
            # Save best model
            state_dict = {
                'params': state.params,
                'batch_stats': state.batch_stats,
                'step': state.step
            }
            with open('models/best_regressor.msgpack', 'wb') as f:
                f.write(serialization.to_bytes(state_dict))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    return state, train_metrics, val_metrics


def plot_training_history(train_metrics: Dict[str, list], val_metrics: Dict[str, list], save_path: str):
    """Plot training history metrics.
    
    Args:
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_metrics['mse'], label='Train MSE')
    plt.plot(val_metrics['mse'], label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Mean Squared Error')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_metrics['mae'], label='Train MAE')
    plt.plot(val_metrics['mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Mean Absolute Error')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Load dataset
    grid_dir = '../../synthesizer_grids/grids/'
    dataset = SpectralDatasetSynthesizer(grid_dir=grid_dir, grid_name='bc03-2016-Miles_chabrier-0.1,100.hdf5', num_samples=int(1e4))
    
    # Split dataset
    rng = jax.random.PRNGKey(0)
    rng, split_rng = jax.random.split(rng)
    perm = jax.random.permutation(split_rng, len(dataset))
    split = int(0.8 * len(dataset))
    
    # Load autoencoder
    autoencoder, autoencoder_state = load_autoencoder(
        'models/best_autoencoder.msgpack',
        spectrum_dim=dataset.n_wavelength,
        latent_dim=128
    )
    
    # Encode all spectra to get latent vectors
    def encode_spectra(spectra):
        variables = {'params': autoencoder_state['params'], 'batch_stats': autoencoder_state['batch_stats']}
        return autoencoder.apply(
            variables,
            spectra,
            method='encode',
            training=False
        )
    

    train_dataset = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[:split])
    val_dataset = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[split:])
    train_latent = encode_spectra(train_dataset.spectra)
    val_latent = encode_spectra(val_dataset.spectra)
    
    # Create regressor model
    model = RegressorMLP(
        hidden_dims=[256, 512, 1024], # 512, 1024, 512, 256],
        latent_dim=128,
        dropout_rate=0.1
    )
    
    # Train and evaluate model
    state, train_metrics, val_metrics = train_and_evaluate(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_latent=train_latent,
        val_latent=val_latent,
        num_epochs=250,
        batch_size=16,
        learning_rate=1e-3,
        patience=30,
        min_delta=1e-4,
        rng=rng
    )
    
    # Plot training history
    plot_training_history(
        train_metrics,
        val_metrics,
        'figures/regressor_training_history.png'
    )


if __name__ == "__main__":
    main() 