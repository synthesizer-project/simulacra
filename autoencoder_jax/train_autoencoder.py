"""
Train a spectrum autoencoder (Flax/JAX) on synthetic spectra.

Description:
- Samples spectra from a synthesizer Grid via SpectralDatasetSynthesizer,
  normalizes them, and trains a dense autoencoder to reconstruct log10 spectra.
- Saves a single bundled msgpack with model hyperparameters, parameters,
  batch statistics, and dataset normalization scalars for later inference.

CLI:
  python train_autoencoder.py <grid_dir> <grid_name>
                              [--samples N] [--epochs E]
                              [--batch-size B] [--save PATH]

Arguments:
- grid_dir: Directory containing the spectral grid HDF5 files.
- grid_name: File name of the grid to load (e.g. bc03-....hdf5).
- --samples: Number of spectra to sample (default: 10000).
- --epochs: Number of training epochs (default: 100).
- --batch-size: Mini-batch size (default: 64).
- --save: Output path for the bundled model (default:
          models/autoencoder_simple_dense.msgpack).

Outputs:
- Bundled model at the provided --save path.

Notes:
- Uses a dense architecture by default (latent=128, dropout=0.2, ReLU).
- JAX selects available devices automatically; set XLA flags as needed.
"""

import sys
sys.path.append('..')

import numpy as np
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


def save_model(model, state, dataset, model_path):
    """Saves the model state, hyperparameters, and normalization constants to a single file."""
    hyperparams = {
        'spectrum_dim': model.spectrum_dim,
        'latent_dim': model.latent_dim,
        'encoder_features': model.encoder_features,
        'decoder_features': model.decoder_features,
        'dropout_rate': model.dropout_rate,
        'activation_name': getattr(model, 'activation_name', 'relu'),
        'architecture': getattr(model, 'architecture', 'dense'), # Save architecture type
    }
    norm_params = {
        'spec_mean': dataset.true_spec_mean,
        'spec_std': dataset.true_spec_std,
    }
    state_dict = {
        'params': state.params,
        'batch_stats': state.batch_stats,
        'step': state.step
    }
    bundled_data = {
        'hyperparams': hyperparams,
        'state_dict': state_dict,
        'norm_params': norm_params,
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
    norm_params = bundled_data.get('norm_params', None)  # For backward compatibility
    
    # Add architecture to hyperparams for loading, default to 'dense' for older models
    hyperparams['architecture'] = hyperparams.get('architecture', 'dense')

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
    
    return model, state, norm_params


# === Model Architecture ===

# --- Dense Architecture ---
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

# --- Convolutional Architecture ---
class ConvSpectrumEncoder(nn.Module):
    """A 1D convolutional encoder for spectra."""
    features: Sequence[int] = (16, 32, 64)
    kernel_sizes: Sequence[int] = (7, 5, 3)
    strides: Sequence[int] = (2, 2, 2)
    latent_dim: int = 128
    dropout_rate: float = 0.2
    activation_name: str = 'relu'

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Add a channel dimension for convolution
        x = jnp.expand_dims(x, axis=-1)

        for i, (feat, k_size, stride) in enumerate(zip(self.features, self.kernel_sizes, self.strides)):
            x = nn.Conv(
                features=feat,
                kernel_size=(k_size,),
                strides=(stride,),
                padding='SAME',
                kernel_init=nn.initializers.he_normal()
            )(x)
            x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
            if self.activation_name == 'relu':
                x = nn.relu(x)
            elif self.activation_name == 'parametric_gated':
                x = ParametricGatedActivation(features=feat)(x)
            else:
                raise ValueError(f"Unsupported activation: {self.activation_name}")
            if training:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(self.latent_dim, kernel_init=nn.initializers.he_normal())(x)
        return x

class ConvSpectrumDecoder(nn.Module):
    """A 1D transposed convolutional decoder for spectra."""
    features: Sequence[int] = (64, 32, 16)
    kernel_sizes: Sequence[int] = (3, 5, 7)
    strides: Sequence[int] = (2, 2, 2)
    spectrum_dim: int = 10787
    encoder_features: Sequence[int] = (16, 32, 64)
    encoder_strides: Sequence[int] = (2, 2, 2)
    dropout_rate: float = 0.2
    activation_name: str = 'relu'

    @nn.compact
    def __call__(self, x, training: bool = True):
        conv_output_dim = self.spectrum_dim
        for stride in self.encoder_strides:
            conv_output_dim = (conv_output_dim + stride - 1) // stride
        
        dense_features = conv_output_dim * self.encoder_features[-1]

        x = nn.Dense(dense_features, kernel_init=nn.initializers.he_normal())(x)
        x = x.reshape((x.shape[0], conv_output_dim, self.encoder_features[-1]))

        for i, (feat, k_size, stride) in enumerate(zip(self.features, self.kernel_sizes, self.strides)):
            x = nn.ConvTranspose(
                features=feat,
                kernel_size=(k_size,),
                strides=(stride,),
                padding='SAME',
                kernel_init=nn.initializers.he_normal()
            )(x)
            x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
            if self.activation_name == 'relu':
                x = nn.relu(x)
            elif self.activation_name == 'parametric_gated':
                x = ParametricGatedActivation(features=feat)(x)
            else:
                raise ValueError(f"Unsupported activation: {self.activation_name}")
            if training:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        x = nn.ConvTranspose(
            features=1, kernel_size=(self.kernel_sizes[-1],), padding='SAME'
        )(x)
        
        x = x[:, :self.spectrum_dim, :]
        x = jnp.squeeze(x, axis=-1)
        return x


class SpectrumAutoencoder(nn.Module):
    spectrum_dim: int
    latent_dim: int
    encoder_features: Sequence[int]
    decoder_features: Sequence[int]
    dropout_rate: float
    activation_name: str = 'relu'
    architecture: str = 'dense'
    
    def setup(self):
        if self.architecture == 'dense':
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
        elif self.architecture == 'conv':
            conv_kernel_sizes = (7, 5, 3)
            conv_strides = (2, 2, 2)
            self.encoder = ConvSpectrumEncoder(
                features=self.encoder_features,
                kernel_sizes=conv_kernel_sizes,
                strides=conv_strides,
                latent_dim=self.latent_dim,
                dropout_rate=self.dropout_rate,
                activation_name=self.activation_name
            )
            self.decoder = ConvSpectrumDecoder(
                features=self.decoder_features,
                kernel_sizes=tuple(reversed(conv_kernel_sizes)),
                strides=tuple(reversed(conv_strides)),
                spectrum_dim=self.spectrum_dim,
                encoder_features=self.encoder_features,
                encoder_strides=conv_strides,
                dropout_rate=self.dropout_rate,
                activation_name=self.activation_name
            )
        else:
            raise ValueError(f"Unsupported architecture: '{self.architecture}'")

    
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
    _, norm_spectra = batch # Unpack, we only need the spectra for the AE
    
    def loss_fn(params, rng):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        rng, dropout_rng = jax.random.split(rng)
        
        pred_spectra, new_model_state = state.apply_fn(
            variables,
            norm_spectra,
            mutable=['batch_stats'],
            training=True,
            rngs={'dropout': dropout_rng}
        )
        
        loss = jnp.mean((pred_spectra - norm_spectra) ** 2)
        return loss, new_model_state
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, new_model_state), grads = grad_fn(state.params, rng)
    
    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats']
    )
    return state, loss

@jax.jit
def eval_step(state, batch):
    """Performs a single evaluation step."""
    _, norm_spectra = batch # Unpack, we only need the spectra for the AE
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    
    pred_spectra = state.apply_fn(
        variables,
        norm_spectra,
        training=False
    )
    
    loss = jnp.mean((pred_spectra - norm_spectra) ** 2)
    return loss

def train_epoch(state, train_ds, batch_size, rng):
    """Trains for a single epoch."""
    train_ds_size = len(train_ds)
    steps_per_epoch = train_ds_size // batch_size
    
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    
    epoch_loss = []
    
    for perm in perms:
        batch = (train_ds.conditions[perm], train_ds.spectra[perm])
        rng, step_rng = jax.random.split(rng)
        state, loss = train_step(state, batch, step_rng)
        epoch_loss.append(loss)
    
    return state, {'loss': np.mean(epoch_loss)}, rng

def eval_model(state, test_ds, batch_size):
    """Evaluates the model on the test set."""
    test_ds_size = len(test_ds)
    steps_per_epoch = test_ds_size // batch_size
    
    all_losses = []
    for i in range(steps_per_epoch):
        start, end = i * batch_size, (i + 1) * batch_size
        batch = (test_ds.conditions[start:end], test_ds.spectra[start:end])
        loss = eval_step(state, batch)
        all_losses.append(loss)
    
    return np.mean(all_losses)

def train_and_evaluate(
    model, train_ds, test_ds, num_epochs, batch_size, learning_rate, rng,
    patience=10, min_delta=1e-4, lr_patience=5, lr_factor=0.5, min_lr=1e-7,
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
        
        train_losses.append(train_losses_epoch['loss'])
        test_losses.append(test_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses_epoch['loss']:.4f} | Test Loss: {test_loss:.4f} | LR: {current_lr:.2e}")
        
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
        save_model(model, best_state, train_ds, save_path)
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


def plot_training_validation_loss(
    train_losses,
    val_losses,
    save_dir: str = 'figures/autoencoder_evaluation',
    filename: str = 'loss_vs_epoch.png',
    use_log10: bool = True,
    title: str = 'Autoencoder Training and Validation Loss',
):
    """Plots training and validation loss vs. epoch and saves the figure.

    Args:
        train_losses: Sequence of training loss values per epoch.
        val_losses: Sequence of validation loss values per epoch.
        save_dir: Directory where the figure will be saved.
        filename: Filename for the saved figure (PNG).
        use_log10: If True, plot log10 of the losses.
        title: Plot title.

    Returns:
        The full path to the saved figure.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    epochs = np.arange(1, len(train_losses) + 1)
    train_arr = np.asarray(train_losses)
    val_arr = np.asarray(val_losses)
    if use_log10:
        eps = 1e-12
        train_arr = np.log10(np.clip(train_arr, eps, None))
        val_arr = np.log10(np.clip(val_arr, eps, None))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_arr, label='Training Loss', color='#1f77b4')
    plt.plot(epochs, val_arr, label='Validation Loss', color='#ff7f0e')
    plt.xlabel('Epoch')
    plt.ylabel('log10 MSE Loss' if use_log10 else 'MSE Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def main():
    print(f"JAX devices: {jax.devices()}")
    
    # Basic CLI: python train_autoencoder.py <grid_dir> <grid_name> [--samples N] [--epochs E] [--batch-size B] [--save PATH]
    if len(sys.argv) < 3:
        print("Usage: python train_autoencoder.py <grid_dir> <grid_name> [--samples N] [--epochs E] [--batch-size B] [--save PATH]")
        sys.exit(1)

    grid_dir, grid_name = sys.argv[1], sys.argv[2]
    # Defaults
    N_samples = int(1e4)
    num_epochs = 100
    batch_size = 64
    save_path = 'models/autoencoder_simple_dense.msgpack'

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

    train_dataset, test_dataset, rng = load_data(grid_dir, grid_name, N_samples)
    
    print("--- Using Standard Dense Architecture ---")
    model = SpectrumAutoencoder(
        spectrum_dim=train_dataset.n_wavelength,
        latent_dim=128,
        encoder_features=(2048, 1024, 512),
        decoder_features=(512, 1024, 2048),
        dropout_rate=0.2,
        activation_name='relu',
        architecture='dense'
    )
    learning_rate = 1e-4

    state, train_losses, test_losses, best_test_loss = train_and_evaluate(
        model=model,
        train_ds=train_dataset,
        test_ds=test_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rng=rng,
        weight_decay=1e-4,
        save_path=save_path
    )

    # Save loss curves to figures directory with a timestamped filename
    try:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        save_dir = 'figures/autoencoder_evaluation'
        fname = f"loss_vs_epoch_{model.architecture}_z{model.latent_dim}_{len(train_losses)}ep_{ts}.png"
        out_path = plot_training_validation_loss(
            train_losses=train_losses,
            val_losses=test_losses,
            save_dir=save_dir,
            filename=fname,
            use_log10=True,
            title='Autoencoder Training and Validation Loss',
        )
        print(f"Saved loss plot to {out_path}")
    except Exception as e:
        print(f"Could not save loss plot: {e}")

if __name__ == "__main__":
    main()
