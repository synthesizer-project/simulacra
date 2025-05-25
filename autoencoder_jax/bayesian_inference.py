import sys
sys.path.append('..')

import jax
import jax.numpy as jnp
import blackjax
import numpy as np
import matplotlib.pyplot as plt
import corner
from typing import Tuple, Optional
from flax import serialization

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


def log_prior(params):
    """Log prior probability for age and metallicity."""
    age, metallicity, noise_scale = params
    
    # Uniform priors
    age_log_prob = jnp.log(jnp.where(
        (age >= 0.1) & (age <= 13.0),
        1.0 / (13.0 - 0.1),
        0.0
    ))
    
    met_log_prob = jnp.log(jnp.where(
        (metallicity >= 0.02) & (metallicity <= 2.0),
        1.0 / (2.0 - 0.02),
        0.0
    ))
    
    # Half-normal prior for noise scale
    noise_log_prob = -0.5 * (noise_scale / 0.1)**2 - jnp.log(0.1) - 0.5 * jnp.log(2 * jnp.pi)
    
    return age_log_prob + met_log_prob + noise_log_prob


def log_likelihood(params, observed_spectrum, dataset, regressor, regressor_state, autoencoder, autoencoder_state):
    """Log likelihood of observed spectrum given parameters."""
    age, metallicity, noise_scale = params
    
    # Generate predicted spectrum
    predicted_spectrum = generate_spectrum(
        regressor,
        regressor_state,
        autoencoder,
        autoencoder_state,
        age,
        metallicity,
        dataset
    )
    
    # Compute log likelihood (assuming Gaussian noise)
    log_prob = -0.5 * jnp.sum(
        ((observed_spectrum - predicted_spectrum) / noise_scale)**2
        + 2 * jnp.log(noise_scale)
        + jnp.log(2 * jnp.pi)
    )
    
    return log_prob


def log_posterior(params, observed_spectrum, dataset, regressor, regressor_state, autoencoder, autoencoder_state):
    """Log posterior probability."""
    log_prior_val = log_prior(params)
    log_likelihood_val = log_likelihood(
        params,
        observed_spectrum,
        dataset,
        regressor,
        regressor_state,
        autoencoder,
        autoencoder_state
    )
    return log_prior_val + log_likelihood_val


def run_inference(
    observed_spectrum: jnp.ndarray,
    dataset,
    regressor,
    regressor_state,
    autoencoder,
    autoencoder_state,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    rng_key: Optional[jnp.ndarray] = None
) -> Tuple[dict, dict]:
    """Run HMC inference on observed spectrum using BlackJAX."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    # Initialize parameters
    initial_position = jnp.array([2.0, 0.5, 0.1])  # [age, metallicity, noise_scale]
    
    # Create log posterior function
    log_post_fn = lambda params: log_posterior(
        params,
        observed_spectrum,
        dataset,
        regressor,
        regressor_state,
        autoencoder,
        autoencoder_state
    )
    
    # Initialize NUTS sampler
    nuts_kernel = blackjax.nuts(log_post_fn, step_size=0.01, inverse_mass_matrix=jnp.eye(3))
    
    # Initialize sampler state
    initial_state = nuts_kernel.init(initial_position)#, rng_key)
    
    # Run sampling
    print("Running sampling...")
    samples = []
    states = []

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states
    
    states = inference_loop(rng_key, nuts_kernel.step, initial_state, num_samples)
    
    samples = jnp.array(states.position)
    samples_dict = {
        'age': samples[:, 0],
        'metallicity': samples[:, 1],
        'noise_scale': samples[:, 2]
    }
    
    return samples_dict, {'states': states}


def plot_inference_results(
    samples: dict,
    true_age: Optional[float] = None,
    true_metallicity: Optional[float] = None,
    save_path: Optional[str] = None
):
    """Plot inference results using corner package."""
    # Convert samples to array format for corner
    samples_array = np.column_stack([
        samples['age'],
        samples['metallicity'],
        samples['noise_scale']
    ])
    
    # Define labels and ranges
    labels = ['Age (Gyr)', 'Metallicity (Z)', 'Noise Scale']
    ranges = [
        (0.1, 13.0),  # Age range
        (0.02, 2.0),  # Metallicity range
        (0.0, 0.5)    # Noise scale range
    ]
    
    # Create true values array if provided
    truths = None
    if true_age is not None and true_metallicity is not None:
        truths = [true_age, true_metallicity, 0.1]  # Using 0.1 as default noise scale
    
    # Create corner plot
    fig = corner.corner(
        samples_array,
        labels=labels,
        range=ranges,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 12},
        figsize=(10, 10)
    )
    
    # Add true values to the plot if provided
    if true_age is not None and true_metallicity is not None:
        # Add text annotation for true values
        fig.text(
            0.95, 0.95,
            f'True Age: {true_age:.2f} Gyr\nTrue Metallicity: {true_metallicity:.2f} Z',
            transform=fig.transFigure,
            ha='right',
            va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_spectrum_comparison(
    observed_spectrum: jnp.ndarray,
    predicted_spectrum: jnp.ndarray,
    wavelengths: jnp.ndarray,
    save_path: Optional[str] = None
):
    """Plot comparison between observed and predicted spectra."""
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, observed_spectrum, label='Observed', alpha=0.7)
    plt.plot(wavelengths, predicted_spectrum, label='Predicted', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title('Spectrum Comparison')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_test_spectrum(
        regressor,
        regressor_state,
        autoencoder,
        autoencoder_state,
        dataset,
        test_age,
        test_metallicity,
        rng
    ):
    """Generate a test spectrum for given age and metallicity."""
    # Generate a test spectrum (you would replace this with real observed data)
    test_spectrum = generate_spectrum(
        regressor,
        regressor_state,
        autoencoder,
        autoencoder_state,
        test_age,
        test_metallicity,
        dataset
    )

    noise = jax.random.normal(rng, test_spectrum.shape) * 0.1
    test_spectrum = test_spectrum + noise
    return test_spectrum

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

    rng = jax.random.PRNGKey(0)

    test_age = 2.0  # Gyr
    test_metallicity = 0.5  # Z
    observed_spectrum = generate_test_spectrum(regressor, regressor_state, autoencoder, autoencoder_state, dataset, test_age, test_metallicity, rng)
    
    rng, rng_key = jax.random.split(rng)
    # Run inference
    print("Running HMC inference...")
    samples, _ = run_inference(
        observed_spectrum,
        dataset,
        regressor,
        regressor_state,
        autoencoder,
        autoencoder_state,
        rng_key=rng,
    )
    
    # Print summary statistics
    print("\nInference Results:")
    print(f"True Age: {test_age:.2f} Gyr")
    print(f"True Metallicity: {test_metallicity:.2f} Z")
    print("\nPosterior Statistics:")
    print(f"Age - Mean: {np.mean(samples['age']):.2f} Gyr, Std: {np.std(samples['age']):.2f}")
    print(f"Metallicity - Mean: {np.mean(samples['metallicity']):.2f} Z, Std: {np.std(samples['metallicity']):.2f}")
    print(f"Noise Scale - Mean: {np.mean(samples['noise_scale']):.2f}, Std: {np.std(samples['noise_scale']):.2f}")
    
    # Plot results
    plot_inference_results(
        samples,
        true_age=test_age,
        true_metallicity=test_metallicity,
        save_path='figures/inference/inference_results.png'
    )
    
    # Plot spectrum comparison
    predicted_spectrum = generate_spectrum(
        regressor,
        regressor_state,
        autoencoder,
        autoencoder_state,
        np.mean(samples['age']),
        np.mean(samples['metallicity']),
        dataset
    )
    plot_spectrum_comparison(
        observed_spectrum,
        predicted_spectrum,
        dataset.wavelength,
        save_path='figures/inference/spectrum_comparison.png'
    )


if __name__ == "__main__":
    main() 