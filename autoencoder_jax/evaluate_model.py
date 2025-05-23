import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import serialization

from autoencoder import SpectrumAutoencoder, TrainState
from grids import SpectralDatasetJAX


def load_model(model_path, model_params):
    """Load a trained model from saved parameters."""

    state_dict = {
        'params': None,
        'batch_stats': None,
        'step': 0
    }
    
    # Load saved parameters using flax serialization
    with open(model_path, 'rb') as f:
        state_dict = serialization.from_bytes(state_dict, f.read())
    
    # Create model instance
    model = SpectrumAutoencoder(
        spectrum_dim=model_params['spectrum_dim'],
        latent_dim=model_params['latent_dim'],
        param_dim=model_params['param_dim']
    )
    
    # Create state with loaded parameters
    state = TrainState.create(
        apply_fn=model.apply,
        params=state_dict['params'],
        batch_stats=state_dict['batch_stats'],
        tx=None  # No optimizer needed for evaluation
    )
    
    return model, state

def plot_reconstruction(model, state, test_ds, num_samples=4):
    """Plots reconstructions of test samples."""
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    
    # Get predictions
    pred_spectrum = model.apply(
        variables,
        test_ds.spectra[:num_samples],
        test_ds.conditions[:num_samples],
        training=False
    )
    
    # Plot results
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        true = test_ds.spectra[i]
        pred = pred_spectrum[i]
        
        # Denormalize conditions
        age = test_ds.conditions[i, 0] * test_ds.ages.std() + test_ds.ages.mean()
        met = test_ds.conditions[i, 1] * test_ds.metallicities.std() + test_ds.metallicities.mean()
        
        ax.plot(test_ds.wavelength, true, label='True', alpha=0.7, color='blue')
        ax.plot(test_ds.wavelength, pred, label='Reconstructed', alpha=0.7, color='red')
        
        # Add error
        error = np.abs(true - pred)
        ax.fill_between(test_ds.wavelength, 
                       true - error, 
                       true + error, 
                       alpha=0.2, 
                       color='gray', 
                       label='Error')
        
        ax.set_title(f'Age: {age:.2f} Gyr, Metallicity: {met:.2f}')
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel('Normalized Flux')
        ax.legend()
        
        # Add error metrics
        mse = np.mean((true - pred) ** 2)
        mae = np.mean(np.abs(true - pred))
        ax.text(0.02, 0.98, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figures/autoencoder_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load test dataset
    grid_dir = '../../synthesizer_grids/grids/'
    dataset = SpectralDatasetJAX(f'{grid_dir}/bc03-2016-Miles_chabrier-0.1,100.hdf5')
    
    # Split dataset (using same split as training)
    rng = jax.random.PRNGKey(0)
    rng, split_rng = jax.random.split(rng)
    perm = jax.random.permutation(split_rng, len(dataset))
    split = int(0.8 * len(dataset))
    test_dataset = SpectralDatasetJAX(parent_dataset=dataset, split=perm[split:])
    
    # Create model parameters based on dataset dimensions
    model_params = {
        'spectrum_dim': test_dataset.n_wavelength,
        'latent_dim': 128,  # This is a hyperparameter we choose
        'param_dim': test_dataset.conditions.shape[1]  # Number of parameters (age, metallicity)
    }
    
    # Load model
    model_path = 'models/best_autoencoder.msgpack'
    model, state = load_model(model_path, model_params)
    
    plot_reconstruction(model, state, test_dataset, num_samples=4)

if __name__ == "__main__":
    main()