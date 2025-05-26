import sys
sys.path.append('..')

import jax
import jax.numpy as jnp
from flax import serialization
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from autoencoder import SpectrumAutoencoder
from regressor import RegressorMLP
from grids import SpectralDatasetSynthesizer


def load_models(autoencoder_path, regressor_path, spectrum_dim, latent_dim):
    """Load trained autoencoder and regressor models."""
    # Load autoencoder
    autoencoder = SpectrumAutoencoder(
        spectrum_dim=spectrum_dim,
        latent_dim=latent_dim
    )
    with open(autoencoder_path, 'rb') as f:
        autoencoder_state = serialization.from_bytes(autoencoder, f.read())
    
    # Load regressor
    regressor = RegressorMLP(
        hidden_dims=[256, 512, 1024],
        latent_dim=latent_dim,
        dropout_rate=0.1
    )
    with open(regressor_path, 'rb') as f:
        regressor_state = serialization.from_bytes(regressor, f.read())
    
    return autoencoder, autoencoder_state, regressor, regressor_state


def generate_spectrum(regressor, regressor_state, autoencoder, autoencoder_state, age, metallicity, dataset):
    """Generate a spectrum for given age and metallicity."""
    # Normalize inputs
    norm_age = (age - dataset.ages.mean()) / dataset.ages.std()
    norm_met = (metallicity - dataset.metallicities.mean()) / dataset.metallicities.std()
    
    # Create input array
    conditions = jnp.array([[norm_age, norm_met]])
    
    # Generate latent vector using regressor
    variables = {'params': regressor_state['params'], 'batch_stats': regressor_state['batch_stats']}
    latent = regressor.apply(
        variables,
        conditions,
        training=False
    )
    
    # Decode latent vector to spectrum using autoencoder
    variables = {'params': autoencoder_state['params'], 'batch_stats': autoencoder_state['batch_stats']}
    spectrum = autoencoder.apply(
        variables,
        latent,
        method='decode',
        training=False
    )
    
    return spectrum[0]  # Remove batch dimension


def compute_gradients(regressor, regressor_state, autoencoder, autoencoder_state, age, metallicity, dataset):
    """Compute gradients of spectrum with respect to age and metallicity.
    
    Returns:
        Tuple of (age_gradients, metallicity_gradients) where each is an array
        of shape (n_wavelength,) containing the gradient at each wavelength.
    """
    def spectrum_fn(params):
        return generate_spectrum(
            regressor,
            regressor_state,
            autoencoder,
            autoencoder_state,
            params[0],  # age
            params[1],  # metallicity
            dataset
        )
    
    # Compute Jacobian (gradients for all wavelengths at once)
    jac_fn = jax.jacobian(spectrum_fn)
    jacobian = jac_fn(jnp.array([age, metallicity]))
    
    # jacobian shape is (n_wavelength, 2) where 2 is for [age, metallicity]
    return jacobian.T  # Transpose to get (2, n_wavelength) for consistency with previous output


def plot_gradients(spectrum, gradients, wavelengths, age, metallicity, save_path):
    """Plot spectrum and its gradients with respect to parameters."""
    plt.figure(figsize=(15, 10))
    
    # Create colormaps for positive and negative values
    pos_cmap = plt.cm.Reds
    neg_cmap = plt.cm.Blues
    
    # Plot spectrum
    plt.subplot(3, 1, 1)
    plt.plot(wavelengths, spectrum, label='Spectrum')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title(f'Spectrum (Age: {age:.2f} Gyr, Z: {metallicity:.2f})')
    plt.legend()
    
    # Plot gradient with respect to age
    plt.subplot(3, 1, 2)
    age_grad = gradients[0]
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Color array for age gradient
    age_colors = np.zeros((len(wavelengths), 4))  # RGBA array
    pos_mask = age_grad >= 0
    neg_mask = age_grad < 0
    
    # Normalize values for coloring
    if np.any(pos_mask):
        pos_norm = age_grad[pos_mask] / np.max(age_grad[pos_mask])
        age_colors[pos_mask] = pos_cmap(pos_norm)
    if np.any(neg_mask):
        neg_norm = -age_grad[neg_mask] / np.min(age_grad[neg_mask])
        age_colors[neg_mask] = neg_cmap(neg_norm)
    
    # Plot with colored segments
    for j in range(len(wavelengths)-1):
        plt.plot(
            wavelengths[j:j+2],
            age_grad[j:j+2],
            color=age_colors[j],
            linewidth=2,
            alpha=0.7
        )
    
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Gradient')
    plt.title('Gradient with respect to Age')
    
    # Add legend for age gradient
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', label='Zero Gradient'),
        Line2D([0], [0], color=pos_cmap(0.8), label='Positive Gradient'),
        Line2D([0], [0], color=neg_cmap(0.8), label='Negative Gradient')
    ]
    plt.legend(handles=legend_elements)
    
    # Plot gradient with respect to metallicity
    plt.subplot(3, 1, 3)
    met_grad = gradients[1]
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Color array for metallicity gradient
    met_colors = np.zeros((len(wavelengths), 4))  # RGBA array
    pos_mask = met_grad >= 0
    neg_mask = met_grad < 0
    
    # Normalize values for coloring
    if np.any(pos_mask):
        pos_norm = met_grad[pos_mask] / np.max(met_grad[pos_mask])
        met_colors[pos_mask] = pos_cmap(pos_norm)
    if np.any(neg_mask):
        neg_norm = -met_grad[neg_mask] / np.min(met_grad[neg_mask])
        met_colors[neg_mask] = neg_cmap(neg_norm)
    
    # Plot with colored segments
    for j in range(len(wavelengths)-1):
        plt.plot(
            wavelengths[j:j+2],
            met_grad[j:j+2],
            color=met_colors[j],
            linewidth=2,
            alpha=0.7
        )
    
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Gradient')
    plt.title('Gradient with respect to Metallicity')
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def test_parameter_sensitivity(
    regressor,
    regressor_state,
    autoencoder,
    autoencoder_state,
    dataset,
    test_points: list[Tuple[float, float]]
):
    """Test sensitivity of spectrum to parameter changes."""
    results = []
    
    for age, metallicity in test_points:
        print(f'Testing age: {age:.1f} Gyr, metallicity: {metallicity:.2f}')
        
        # Generate spectrum
        spectrum = generate_spectrum(
            regressor,
            regressor_state,
            autoencoder,
            autoencoder_state,
            age,
            metallicity,
            dataset
        )
        
        # Compute gradients
        gradients = compute_gradients(
            regressor,
            regressor_state,
            autoencoder,
            autoencoder_state,
            age,
            metallicity,
            dataset
        )
        
        # Store results
        results.append({
            'age': age,
            'metallicity': metallicity,
            'spectrum': spectrum,
            'gradients': gradients
        })
        
        # Plot results
        plot_gradients(
            spectrum,
            gradients,
            dataset.wavelength,
            age,
            metallicity,
            f'figures/gradients/gradients_age{age:.1f}_z{metallicity:.2f}.png'
        )
    
    return results


def main():
    # Load dataset
    grid_dir = '../../synthesizer_grids/grids/'
    dataset = SpectralDatasetSynthesizer(grid_dir=grid_dir, grid_name='bc03-2016-Miles_chabrier-0.1,100.hdf5')
    
    # Load models
    autoencoder, autoencoder_state, regressor, regressor_state = load_models(
        'models/best_autoencoder.msgpack',
        'models/best_regressor.msgpack',
        spectrum_dim=dataset.n_wavelength,
        latent_dim=128
    )
    
    # Test points covering different regions of parameter space
    test_points = [
        (0.1, 0.02),   # Young, low metallicity
        (1.0, 0.02),   # Intermediate age, low metallicity
        (10.0, 0.02),  # Old, low metallicity
        (0.1, 1.0),    # Young, solar metallicity
        (1.0, 1.0),    # Intermediate age, solar metallicity
        (10.0, 1.0),   # Old, solar metallicity
        (0.1, 2.0),    # Young, high metallicity
        (1.0, 2.0),    # Intermediate age, high metallicity
        (10.0, 2.0),   # Old, high metallicity
    ]
    
    # Run tests
    results = test_parameter_sensitivity(
        regressor,
        regressor_state,
        autoencoder,
        autoencoder_state,
        dataset,
        test_points
    )
    
    # Print summary statistics
    print("\nGradient Statistics:")
    for result in results:
        age = result['age']
        metallicity = result['metallicity']
        gradients = result['gradients']
        
        print(f"\nAge: {age:.1f} Gyr, Z: {metallicity:.2f}")
        print(f"Age gradient - Mean: {np.mean(gradients[0]):.2e}, Std: {np.std(gradients[0]):.2e}")
        print(f"Metallicity gradient - Mean: {np.mean(gradients[1]):.2e}, Std: {np.std(gradients[1]):.2e}")


if __name__ == "__main__":
    main() 