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
from typing import Sequence, Dict, Tuple
import h5py
import os

from grids import SpectralDatasetSynthesizer
from activations import ParametricGatedActivation

# === Model Saving and Loading ===
def save_model(model, params, pca_data_path, save_path, pca_group=None, whitened=False):
    """Saves the emulator model, its hyperparameters, and the path to the PCA data."""
    hyperparams = {
        'features': model.features,
        'n_components': model.n_components,
        'activation_name': model.activation_name,
    }
    bundled_data = {
        'hyperparams': hyperparams,
        'params': params,
        'pca_data_path': pca_data_path,
        'pca_group': pca_group,
        'whitened': bool(whitened),
    }
    with open(save_path, 'wb') as f:
        f.write(serialization.to_bytes(bundled_data))

def load_model(model_path: str) -> Tuple[nn.Module, Dict, str, str, bool]:
    """
    Loads an emulator model, its parameters, and the path to the PCA data file.
    """
    with open(model_path, 'rb') as f:
        bundle = serialization.from_bytes(None, f.read())

    hyperparams = bundle['hyperparams']
    params = bundle['params']
    pca_data_path = bundle.get('pca_data_path')
    pca_group = bundle.get('pca_group')
    whitened = bool(bundle.get('whitened', False))

    # Robustly handle features that might be stored as lists or dicts.
    features = hyperparams.get('features')
    if isinstance(features, dict):
        # Sort by key ('0', '1', ...) to maintain order
        sorted_features = sorted(features.items(), key=lambda item: int(item[0]))
        hyperparams['features'] = tuple(int(v) for k, v in sorted_features)
    elif isinstance(features, (list, tuple)):
        hyperparams['features'] = tuple(int(f) for f in features)

    # Recreate model from hyperparameters
    model = PCAEmulator(
        features=hyperparams['features'], 
        n_components=hyperparams['n_components'],
        activation_name=hyperparams.get('activation_name', 'relu') # Default to relu for older models
    )
    
    return model, params, pca_data_path, pca_group, whitened

# === Model Architecture ===
class PCAEmulator(nn.Module):
    """An MLP to emulate the PCA components of spectra from physical conditions."""
    features: Sequence[int]
    n_components: int
    activation_name: str = 'relu'

    @nn.compact
    def __call__(self, x, training: bool = True):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            if self.activation_name == 'relu':
                x = nn.relu(x)
            elif self.activation_name == 'parametric_gated':
                x = ParametricGatedActivation(features=feat)(x)
            else:
                raise ValueError(f"Unsupported activation: {self.activation_name}")
        x = nn.Dense(self.n_components)(x)
        return x

# === Training State and Functions ===
def create_train_state(rng, model, learning_rate, weight_decay=1e-4):
    """Creates initial training state."""
    params = model.init(rng, jnp.ones((1, 2)))['params'] # 2 input features (age, Z)
    # Use inject_hyperparams to allow for dynamic LR changes
    optimizer = optax.inject_hyperparams(optax.adamw)(
        learning_rate=learning_rate, 
        weight_decay=weight_decay
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optimizer)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    conditions, pca_weights = batch
    def loss_fn(params):
        pred_weights = state.apply_fn({'params': params}, conditions)
        loss = jnp.mean((pred_weights - pca_weights) ** 2)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch):
    conditions, pca_weights = batch
    pred_weights = state.apply_fn({'params': state.params}, conditions)
    return jnp.mean((pred_weights - pca_weights) ** 2)

# === Main Training Loop ===
def train_and_evaluate(
    train_data, test_data, n_components, num_epochs, batch_size, learning_rate, rng,
    model_features=(256, 256, 256), activation_name='relu', patience=15, lr_patience=5, lr_factor=0.5, 
    min_lr=1e-7, min_delta=1e-6, weight_decay=1e-4, verbose=True, 
    save_path=None, pca_data_path=None, extra_save_kwargs=None
):
    """Trains the PCA emulator and evaluates it."""
    train_conditions, train_labels = train_data
    test_conditions, test_labels = test_data

    model = PCAEmulator(
        features=model_features, 
        n_components=n_components,
        activation_name=activation_name
    )
    
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, learning_rate, weight_decay=weight_decay)

    best_test_loss = float('inf')
    best_params = None
    no_improve_epochs = 0
    lr_no_improve_epochs = 0
    current_lr = learning_rate

    for epoch in range(num_epochs):
        train_ds_size = len(train_conditions)
        steps_per_epoch = train_ds_size // batch_size
        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
        
        train_loss = 0
        epoch_iterator = tqdm(perms, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False) if verbose else perms
        for perm in epoch_iterator:
            batch = (train_conditions[perm], train_labels[perm])
            state, loss = train_step(state, batch)
            train_loss += loss
        
        avg_train_loss = train_loss / steps_per_epoch

        test_ds_size = len(test_conditions)
        steps_per_epoch_test = test_ds_size // batch_size
        
        # Add a check to prevent division by zero
        if steps_per_epoch_test == 0:
            print("Warning: test set size is smaller than batch size, reporting loss on a single batch.")
            # Create a single batch from the entire test set
            batch = (test_conditions, test_labels)
            avg_test_loss = eval_step(state, batch)
        else:
            test_loss = 0
            for i in range(steps_per_epoch_test):
                start, end = i * batch_size, (i+1) * batch_size
                batch = (test_conditions[start:end], test_labels[start:end])
                test_loss += eval_step(state, batch)
            
            avg_test_loss = test_loss / steps_per_epoch_test
        
        if verbose:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f} | LR: {current_lr:.2e}")
        
        if avg_test_loss < best_test_loss - min_delta:
            best_test_loss = float(avg_test_loss)
            best_params = state.params
            no_improve_epochs = 0
            lr_no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            lr_no_improve_epochs += 1
            if no_improve_epochs >= patience:
                if verbose: print(f"\nEarly stopping after {epoch + 1} epochs.")
                break
            if lr_no_improve_epochs >= lr_patience:
                new_lr = max(current_lr * lr_factor, min_lr)
                if new_lr < current_lr:
                    if verbose: print(f"Reducing LR to {new_lr:.2e}")
                    current_lr = new_lr
                    
                    # Correct way to update hyperparams with optax.inject_hyperparams
                    # The optimizer is the second element in the chain
                    state = state.replace(
                        opt_state=jax.tree_util.tree_map(
                            lambda s: s.replace(hyperparams={'learning_rate': new_lr}) if hasattr(s, 'hyperparams') else s,
                            state.opt_state
                        )
                    )
                lr_no_improve_epochs = 0
    
    if save_path and best_params is not None:
        # Save the best parameters, not the final state's parameters
        final_state_to_save = state.replace(params=best_params)
        kw = extra_save_kwargs or {}
        save_model(model, final_state_to_save.params, pca_data_path, save_path, **kw)
        if verbose: print(f"\nBest model saved to {save_path}")

    return best_test_loss

def main():
    if len(sys.argv) not in (4,5):
        print("Usage: python train_emulator.py <grid_dir> <grid_name> <pca_data_path> [pca_group]")
        sys.exit(1)
    grid_dir, grid_name_arg, pca_data_path = sys.argv[1], sys.argv[2], sys.argv[3]
    pca_group = sys.argv[4] if len(sys.argv) == 5 else None
    # Handle if user provides full filename for consistency
    grid_name = os.path.splitext(grid_name_arg)[0] if grid_name_arg.endswith('.hdf5') else grid_name_arg
    
    # Define paths based on the clean grid name
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)
    emulator_save_path = f'{output_dir}/emulator_model_{grid_name}.msgpack'

    print(f"JAX devices: {jax.devices()}")

    # --- Load PCA components and Normalization Data ---
    print(f"Loading PCA data from {pca_data_path}...")
    def _select_pca_group(f, group_name=None):
        if group_name and group_name in f:
            return f[group_name]
        if 'latest_group' in f.attrs and f.attrs['latest_group'] in f:
            return f[f.attrs['latest_group']]
        groups = [k for k in f.keys() if isinstance(f.get(k, getclass=True), h5py.Group) and ('n_' in k)]
        if groups:
            import re
            def parse_n(s):
                m = re.search(r'n_(\d+)$', s)
                return int(m.group(1)) if m else -1
            gname = max(groups, key=parse_n)
            return f[gname]
        return f

    with h5py.File(pca_data_path, 'r') as f:
        g = _select_pca_group(f, pca_group)
        pca_input_mean = g['pca_input_mean'][:]
        pca_components = g['pca_components'][:]
        eigenvalues = g['eigenvalues'][:]
        # Load normalization scalars from attributes
        true_spec_mean = g.attrs['true_spec_mean']
        true_spec_std = g.attrs['true_spec_std']
    
    # The number of components is determined by the trained PCA model file
    n_components = pca_components.shape[0]
    print(f"Using {n_components} PCA components from the loaded file.")

    # --- Load & Transform Full Dataset ---
    print("Loading and transforming full dataset...")
    full_dataset = SpectralDatasetSynthesizer(
        grid_dir=grid_dir,
        grid_name=grid_name_arg, # Use original arg to load the correct file
        num_samples=int(5e4),
        norm='global',
        true_spec_mean=true_spec_mean,
        true_spec_std=true_spec_std
    )
    
    # Project onto PCA components to get the weights. 
    # The spectra from the generator are already normalized.
    spectra_demeaned = full_dataset.spectra - pca_input_mean
    pca_weights = spectra_demeaned @ pca_components.T
    # Whitening
    eps = 1e-8
    inv_sqrt_eigs = 1.0 / np.sqrt(eigenvalues + eps)
    pca_weights_wh = pca_weights * inv_sqrt_eigs

    conditions = full_dataset.conditions
    
    print("Loading and transforming validation dataset...")
    validation_dataset = SpectralDatasetSynthesizer(
        grid_dir=grid_dir,
        grid_name=grid_name_arg,
        num_samples=int(5e3),
        norm='global',
        # seed=43,  # Different seed for validation set
        true_spec_mean=true_spec_mean,
        true_spec_std=true_spec_std
    )
    validation_spectra_demeaned = validation_dataset.spectra - pca_input_mean
    validation_weights = validation_spectra_demeaned @ pca_components.T
    validation_weights_wh = validation_weights * inv_sqrt_eigs

    # --- Create Training and Validation Sets ---
    train_conditions = conditions
    train_labels = pca_weights_wh
    
    test_conditions = validation_dataset.conditions
    test_labels = validation_weights_wh

    print(f"Training set size: {len(train_conditions)}")
    print(f"Validation set size: {len(test_conditions)}")

    # --- Train Model ---
    rng = jax.random.PRNGKey(42)
    
    train_and_evaluate(
        train_data=(train_conditions, train_labels),
        test_data=(test_conditions, test_labels),
        n_components=n_components,
        num_epochs=1000,
        batch_size=1024,
        learning_rate=1e-3,
        rng=rng,
        model_features=(256, 256, 256),
        activation_name='parametric_gated',
        patience=25,
        lr_patience=10,
        save_path=emulator_save_path,
        pca_data_path=pca_data_path,
        extra_save_kwargs={'pca_group': pca_group, 'whitened': True},
    )

if __name__ == '__main__':
    main() 
