import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Sequence
from flax.training import train_state
from flax import serialization
import optuna

from grids import SpectralDatasetSynthesizer
from activations import ParametricGatedActivation


def save_model(model, state, model_path):
    """Saves the model state and hyperparameters to a single file."""
    hyperparams = {
        'spectrum_dim': model.spectrum_dim,
        'latent_dim': model.latent_dim,
        'encoder_features': model.encoder_features,
        'decoder_features': model.decoder_features,
        'dropout_rate': model.dropout_rate,
        'activation_name': getattr(model, 'activation_name', 'relu'),
    }
    state_dict = {
        'params': state.params,
        'batch_stats': state.batch_stats,
        'step': state.step
    }
    bundled_data = {
        'hyperparams': hyperparams,
        'state_dict': state_dict
    }
    with open(model_path, 'wb') as f:
        f.write(serialization.to_bytes(bundled_data))

def load_model(model_path):
    """Loads a model and its state from a single file."""
    with open(model_path, 'rb') as f:
        # The target can be None when deserializing a dictionary
        bundled_data = serialization.from_bytes(None, f.read())
    
    hyperparams = bundled_data['hyperparams']
    state_dict = bundled_data['state_dict']
    
    # Robustly handle features that might be stored as lists or dicts,
    # and ensure they are integers. This prevents data type errors.
    for key in ['encoder_features', 'decoder_features']:
        features = hyperparams.get(key)
        if isinstance(features, dict):
            # Sort by key ('0', '1', ...) to maintain order
            sorted_features = sorted(features.items(), key=lambda item: int(item[0]))
            hyperparams[key] = tuple(int(v) for k, v in sorted_features)
        elif isinstance(features, (list, tuple)):
            hyperparams[key] = tuple(int(f) for f in features)

    # Create model instance with the loaded hyperparameters
    model = SpectrumAutoencoder(**hyperparams)
    
    # Create a TrainState object to hold the loaded state
    state = TrainState(
        step=state_dict['step'],
        apply_fn=model.apply,
        params=state_dict['params'],
        batch_stats=state_dict['batch_stats'],
        tx=None,  # Optimizer state is not needed for inference
        opt_state={}
    )
    
    return model, state


# === Model Architecture ===
class SpectrumEncoder(nn.Module):
    features: Sequence[int] = (1024, 512, 256)
    latent_dim: int = 128
    dropout_rate: float = 0.2
    activation_name: str = 'relu'
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        for feat in self.features:
            x = nn.Dense(feat, kernel_init=nn.initializers.he_normal())(x)
            x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
            if self.activation_name == 'relu':
                x = nn.relu(x)
            elif self.activation_name == 'parametric_gated':
                x = ParametricGatedActivation(features=feat)(x)
            else:
                raise ValueError(f"Unsupported activation: {self.activation_name}")
            if training:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.latent_dim, kernel_init=nn.initializers.he_normal())(x)
        return x

class SpectrumDecoder(nn.Module):
    features: Sequence[int] = (256, 512, 1024)
    spectrum_dim: int = 10787
    dropout_rate: float = 0.2
    activation_name: str = 'relu'
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        for feat in self.features:
            x = nn.Dense(feat, kernel_init=nn.initializers.he_normal())(x)
            x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
            if self.activation_name == 'relu':
                x = nn.relu(x)
            elif self.activation_name == 'parametric_gated':
                x = ParametricGatedActivation(features=feat)(x)
            else:
                raise ValueError(f"Unsupported activation: {self.activation_name}")
            if training:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.spectrum_dim, kernel_init=nn.initializers.he_normal())(x)
        return x

class SpectrumAutoencoder(nn.Module):
    spectrum_dim: int
    latent_dim: int
    encoder_features: Sequence[int]
    decoder_features: Sequence[int]
    dropout_rate: float
    activation_name: str = 'relu'
    
    def setup(self):
        self.encoder = SpectrumEncoder(
            features=self.encoder_features,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            activation_name=self.activation_name
        )
        self.decoder = SpectrumDecoder(
            features=self.decoder_features,
            spectrum_dim=self.spectrum_dim,
            dropout_rate=self.dropout_rate,
            activation_name=self.activation_name
        )
    
    def encode(self, spectrum, training: bool = True):
        """Encodes a spectrum into a latent vector. To be called via .apply(..., method='encode')"""
        return self.encoder(spectrum, training=training)
    
    def decode(self, latent, training: bool = True):
        """Decodes a latent vector into a spectrum. To be called via .apply(..., method='decode')"""
        return self.decoder(latent, training=training)
    
    def __call__(self, spectrum, training: bool = True):
        """The full autoencoder forward pass."""
        latent = self.encode(spectrum, training=training)
        return self.decode(latent, training=training)

# === Training State ===
class TrainState(train_state.TrainState):
    batch_stats: dict

# === Training Functions ===
def create_train_state(rng, model, learning_rate, weight_decay=1e-4):
    """Creates initial training state."""
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, jnp.ones((1, model.spectrum_dim)))
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    # Use inject_hyperparams to make the learning rate a dynamic parameter
    # that can be changed efficiently during training.
    optimizer = optax.inject_hyperparams(optax.adamw)(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        b1=0.9,
        b2=0.999,
        eps=1e-8
    )
    
    # Chain the optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optimizer
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=tx
    )

@jax.jit
def train_step(state, batch, rng):
    """Performs a single training step."""
    def loss_fn(params, rng):
        spectrum = batch
        variables = {'params': params, 'batch_stats': state.batch_stats}
        rng, dropout_rng = jax.random.split(rng)
        
        # We need to call the top-level SpectrumAutoencoder's apply method
        pred_spectrum, new_model_state = state.apply_fn(
            variables,
            spectrum,
            mutable=['batch_stats'],
            training=True,
            rngs={'dropout': dropout_rng}
        )
        # Add L2 regularization
        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        reconstruction_loss = jnp.mean((pred_spectrum - spectrum) ** 2)
        loss = reconstruction_loss + 1e-4 * l2_loss
        return loss, (new_model_state, reconstruction_loss, l2_loss)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_model_state, recon_loss, l2_loss)), grads = grad_fn(state.params, rng)
    
    # Create a new state with updated gradients and batch stats
    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats']
    )
    return state, loss, recon_loss, l2_loss

@jax.jit
def eval_step(state, batch):
    """Performs a single evaluation step."""
    spectrum = batch
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    pred_spectrum = state.apply_fn(
        variables,
        spectrum,
        training=False
    )
    loss = jnp.mean((pred_spectrum - spectrum) ** 2)
    return loss

def train_epoch(state, train_ds, batch_size, rng):
    """Trains for a single epoch."""
    train_ds_size = len(train_ds)
    steps_per_epoch = train_ds_size // batch_size
    
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    
    epoch_loss, epoch_recon_loss, epoch_l2_loss = [], [], []
    
    for perm in perms:
        batch = train_ds.spectra[perm]
        rng, step_rng = jax.random.split(rng)
        state, loss, recon_loss, l2_loss = train_step(state, batch, step_rng)
        epoch_loss.append(loss)
        epoch_recon_loss.append(recon_loss)
        epoch_l2_loss.append(l2_loss)
    
    return state, {
        'total_loss': np.mean(epoch_loss),
        'recon_loss': np.mean(epoch_recon_loss),
        'l2_loss': np.mean(epoch_l2_loss)
    }, rng

def eval_model(state, test_ds, batch_size):
    """Evaluates the model on the test set."""
    test_ds_size = len(test_ds)
    steps_per_epoch = test_ds_size // batch_size
    
    all_losses = []
    for i in range(steps_per_epoch):
        batch = test_ds.spectra[i * batch_size:(i + 1) * batch_size]
        loss = eval_step(state, batch)
        all_losses.append(loss)
    
    return np.mean(all_losses)

def train_and_evaluate(
    model, train_ds, test_ds, num_epochs, batch_size, learning_rate, rng,
    patience=10, min_delta=1e-4, lr_patience=3, lr_factor=0.5, min_lr=1e-7,
    weight_decay=1e-4, trial=None, verbose=True, save_path=None
):
    """Trains the model and evaluates it."""
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, learning_rate, weight_decay)
    
    train_losses, test_losses = [], []
    best_test_loss, best_epoch, no_improve_epochs = float('inf'), 0, 0
    best_state = None
    
    current_lr, lr_no_improve_epochs = learning_rate, 0
    
    for epoch in range(num_epochs):
        rng, input_rng = jax.random.split(rng)
        state, train_losses_epoch, rng = train_epoch(state, train_ds, batch_size, input_rng)
        test_loss = eval_model(state, test_ds, batch_size)
        
        train_losses.append(train_losses_epoch['total_loss'])
        test_losses.append(test_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses_epoch['total_loss']:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr:.2e}")
        
        if test_loss < best_test_loss - min_delta:
            best_test_loss = test_loss
            best_epoch = epoch
            no_improve_epochs = 0
            lr_no_improve_epochs = 0
            best_state = state
        else:
            no_improve_epochs += 1
            lr_no_improve_epochs += 1
            if verbose:
                print(f"No improvement for {no_improve_epochs} epochs")
            
            if lr_no_improve_epochs >= lr_patience:
                new_lr = max(current_lr * lr_factor, min_lr)
                if new_lr != current_lr:
                    if verbose:
                        print(f"Reducing learning rate from {current_lr:.2e} to {new_lr:.2e}")
                    current_lr = new_lr
                    lr_no_improve_epochs = 0
                    
                    # More efficient learning rate update.
                    # This directly mutates the learning rate in the optimizer state
                    # without requiring a full state re-initialization.
                    # The optimizer state is a tuple: (ClipState, InjectHyperparamsState)
                    # We modify the hyperparams dict in the InjectHyperparamsState at index 1.
                    state.opt_state[1].hyperparams['learning_rate'] = current_lr
        
        if trial:
            trial.report(float(test_loss), epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if no_improve_epochs >= patience:
            if verbose:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs. Best test loss: {best_test_loss:.4f} at epoch {best_epoch + 1}")
            break
            
    if save_path and best_state:
        save_model(model, best_state, save_path)
        if verbose:
            print(f"\nBest model from epoch {best_epoch + 1} saved to {save_path}")
    
    return state, train_losses, test_losses, best_test_loss

def load_data(grid_dir, grid_name, n_samples):
    dataset = SpectralDatasetSynthesizer(grid_dir=grid_dir, grid_name=grid_name, num_samples=n_samples)
    rng = jax.random.PRNGKey(0)
    perm = jax.random.permutation(rng, len(dataset))
    split = int(0.8 * len(dataset))
    train_dataset = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[:split])
    test_dataset = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[split:])
    return train_dataset, test_dataset, rng

def main():
    print(f"JAX devices: {jax.devices()}")
    
    N_samples = int(1e4)
    grid_dir, grid_name = sys.argv[1], sys.argv[2]
    train_dataset, test_dataset, rng = load_data(grid_dir, grid_name, N_samples)
    
    model = SpectrumAutoencoder(
        spectrum_dim=train_dataset.n_wavelength,
        latent_dim=128,
        encoder_features=(1024, 512, 256),
        decoder_features=(256, 512, 1024),
        dropout_rate=0.2,
        activation_name='parametric_gated'
    )
    
    train_and_evaluate(
        model=model,
        train_ds=train_dataset,
        test_ds=test_dataset,
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-3,
        rng=rng,
        weight_decay=1e-4,
        save_path='models/best_autoencoder.msgpack'
    )

if __name__ == "__main__":
    main() 
