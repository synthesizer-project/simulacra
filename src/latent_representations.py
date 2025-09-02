"""
Abstract base class and interface for latent representation methods.

This module defines the common interface for different latent representation
approaches (autoencoder, PCA, etc.) used in spectrum emulation.
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Dict, Any


class LatentRepresentation(ABC):
    """Abstract base class for latent representation methods."""
    
    @abstractmethod
    def get_latent_dim(self) -> int:
        """Return the dimensionality of the latent space."""
        pass
    
    @abstractmethod
    def encode_spectra(self, spectra: jnp.ndarray) -> jnp.ndarray:
        """Encode spectra to latent representations.
        
        Args:
            spectra: Input spectra array of shape (n_samples, n_wavelengths)
            
        Returns:
            Latent representations of shape (n_samples, latent_dim)
        """
        pass
    
    @abstractmethod
    def decode_latents(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Decode latent representations back to spectra.
        
        Args:
            latents: Latent vectors of shape (n_samples, latent_dim)
            
        Returns:
            Reconstructed spectra of shape (n_samples, n_wavelengths)
        """
        pass
    
    @abstractmethod
    def get_normalization_params(self) -> Dict[str, Any]:
        """Return normalization parameters needed for reconstruction.
        
        Returns:
            Dictionary containing normalization parameters
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the latent representation model.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load_model(cls, path: str, **kwargs):
        """Load a saved latent representation model.
        
        Args:
            path: Path to the saved model
            **kwargs: Additional arguments specific to the implementation
            
        Returns:
            Instance of the latent representation class
        """
        pass
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of the latent representation method."""
        pass