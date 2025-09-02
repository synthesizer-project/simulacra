"""
Autoencoder-based latent representation implementation.

This module implements the LatentRepresentation interface using a trained
autoencoder neural network for spectrum compression and reconstruction.
"""

import jax.numpy as jnp
from typing import Dict, Any
from latent_representations import LatentRepresentation
from train_autoencoder import load_model as load_autoencoder_model


class AutoencoderLatentRepresentation(LatentRepresentation):
    """Latent representation using a trained autoencoder."""
    
    def __init__(self, autoencoder_model, autoencoder_state, norm_params):
        """Initialize with loaded autoencoder components.
        
        Args:
            autoencoder_model: The trained autoencoder model
            autoencoder_state: The trained model state (params, batch_stats)
            norm_params: Normalization parameters for spectra
        """
        self.autoencoder = autoencoder_model
        self.state = autoencoder_state
        self.norm_params = norm_params
        self.latent_dim = autoencoder_model.latent_dim
    
    def get_latent_dim(self) -> int:
        """Return the dimensionality of the autoencoder latent space."""
        return self.latent_dim
    
    def encode_spectra(self, spectra: jnp.ndarray) -> jnp.ndarray:
        """Encode normalized spectra to latent vectors.
        
        Args:
            spectra: Normalized spectra array of shape (n_samples, n_wavelengths)
            
        Returns:
            Latent vectors of shape (n_samples, latent_dim)
        """
        variables = {
            'params': self.state.params, 
            'batch_stats': self.state.batch_stats
        }
        return self.autoencoder.apply(
            variables, spectra, method='encode', training=False
        )
    
    def decode_latents(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Decode latent vectors to normalized spectra.
        
        Args:
            latents: Latent vectors of shape (n_samples, latent_dim)
            
        Returns:
            Reconstructed normalized spectra of shape (n_samples, n_wavelengths)
        """
        variables = {
            'params': self.state.params, 
            'batch_stats': self.state.batch_stats
        }
        return self.autoencoder.apply(
            variables, latents, method='decode', training=False
        )
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Return autoencoder normalization parameters."""
        return self.norm_params if self.norm_params is not None else {}
    
    def save_model(self, path: str) -> None:
        """Save the autoencoder model (delegates to existing save function)."""
        from train_autoencoder import save_model
        # Create a dummy dataset with norm params for saving
        class DummyDataset:
            def __init__(self, norm_params):
                self.true_spec_mean = norm_params.get('spec_mean')
                self.true_spec_std = norm_params.get('spec_std')
        
        dummy_dataset = DummyDataset(self.norm_params)
        save_model(self.autoencoder, self.state, dummy_dataset, path)
    
    @classmethod
    def load_model(cls, path: str, **kwargs):
        """Load trained autoencoder from msgpack file.
        
        Args:
            path: Path to the autoencoder msgpack file
            **kwargs: Additional arguments (unused for autoencoder)
            
        Returns:
            AutoencoderLatentRepresentation instance
        """
        autoencoder, state, norm_params = load_autoencoder_model(path)
        return cls(autoencoder, state, norm_params)
    
    @property
    def method_name(self) -> str:
        """Return the method name."""
        return "autoencoder"
    
    def encode_spectra_batched(self, spectra: jnp.ndarray, batch_size: int = 64) -> jnp.ndarray:
        """Encode spectra in batches to manage memory usage.
        
        Args:
            spectra: Input spectra array
            batch_size: Size of batches for processing
            
        Returns:
            Concatenated latent vectors
        """
        latent_vectors = []
        
        for i in range(0, len(spectra), batch_size):
            batch_spectra = spectra[i:i + batch_size]
            latent_batch = self.encode_spectra(batch_spectra)
            latent_vectors.append(latent_batch)
        
        return jnp.concatenate(latent_vectors) if latent_vectors else jnp.array([])
    
    def decode_latents_batched(self, latents: jnp.ndarray, batch_size: int = 64) -> jnp.ndarray:
        """Decode latents in batches to manage memory usage.
        
        Args:
            latents: Input latent vectors
            batch_size: Size of batches for processing
            
        Returns:
            Concatenated reconstructed spectra
        """
        spectra_batches = []
        
        for i in range(0, len(latents), batch_size):
            batch_latents = latents[i:i + batch_size]
            spectra_batch = self.decode_latents(batch_latents)
            spectra_batches.append(spectra_batch)
        
        return jnp.concatenate(spectra_batches) if spectra_batches else jnp.array([])