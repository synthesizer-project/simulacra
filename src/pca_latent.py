"""
PCA-based latent representation implementation.

This module implements the LatentRepresentation interface using PCA 
(Principal Component Analysis) for spectrum compression and reconstruction.
"""

import jax.numpy as jnp
import numpy as np
import h5py
from typing import Dict, Any, Optional
from latent_representations import LatentRepresentation


class PCALatentRepresentation(LatentRepresentation):
    """Latent representation using Principal Component Analysis."""
    
    def __init__(self, pca_components, pca_input_mean, eigenvalues, 
                 norm_params, whitened=True, wavelengths=None):
        """Initialize with PCA components and parameters.
        
        Args:
            pca_components: PCA basis vectors of shape (n_components, n_wavelengths)
            pca_input_mean: Mean spectrum used for centering
            eigenvalues: Eigenvalues from PCA (explained variances)
            norm_params: Normalization parameters for spectra
            whitened: Whether to apply whitening transformation
        """
        self.pca_components = jnp.array(pca_components)
        self.pca_input_mean = jnp.array(pca_input_mean)
        self.eigenvalues = jnp.array(eigenvalues)
        self.norm_params = norm_params
        self.whitened = whitened
        self.latent_dim = pca_components.shape[0]
        self.wavelengths = jnp.array(wavelengths) if wavelengths is not None else None
        
        # Precompute whitening transformation
        if whitened:
            eps = 1e-8
            self.inv_sqrt_eigs = 1.0 / jnp.sqrt(eigenvalues + eps)
            self.sqrt_eigs = jnp.sqrt(eigenvalues + eps)
        else:
            self.inv_sqrt_eigs = None
            self.sqrt_eigs = None
    
    def get_latent_dim(self) -> int:
        """Return the number of PCA components."""
        return self.latent_dim
    
    def encode_spectra(self, spectra: jnp.ndarray) -> jnp.ndarray:
        """Project spectra onto PCA components.
        
        Args:
            spectra: Input spectra array of shape (n_samples, n_wavelengths)
            
        Returns:
            PCA weights of shape (n_samples, n_components)
        """
        # Center the spectra
        spectra_centered = spectra - self.pca_input_mean
        
        # Project onto PCA components
        pca_weights = spectra_centered @ self.pca_components.T
        
        # Apply whitening if requested
        if self.whitened and self.inv_sqrt_eigs is not None:
            pca_weights = pca_weights * self.inv_sqrt_eigs
            
        return pca_weights
    
    def decode_latents(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct spectra from PCA weights.
        
        Args:
            latents: PCA weights of shape (n_samples, n_components)
            
        Returns:
            Reconstructed spectra of shape (n_samples, n_wavelengths)
        """
        pca_weights = latents
        
        # Undo whitening if it was applied
        if self.whitened and self.sqrt_eigs is not None:
            pca_weights = pca_weights * self.sqrt_eigs
        
        # Reconstruct from PCA components
        reconstructed = pca_weights @ self.pca_components + self.pca_input_mean
        return reconstructed
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Return PCA normalization parameters with unified keys."""
        if self.norm_params is None:
            return {}
        
        # Return parameters with consistent keys for unified interface
        unified_params = dict(self.norm_params)
        
        # Map PCA-specific keys to unified keys if needed
        if 'true_spec_mean' in unified_params:
            unified_params['spec_mean'] = unified_params['true_spec_mean']
        if 'true_spec_std' in unified_params:
            unified_params['spec_std'] = unified_params['true_spec_std']
        # Include wavelength grid if available
        if self.wavelengths is not None:
            unified_params['wavelength'] = np.array(self.wavelengths)
            
        return unified_params
    
    def save_model(self, path: str) -> None:
        """Save PCA model to HDF5 file.
        
        Args:
            path: Path to save the PCA model (.h5 file)
        """
        with h5py.File(path, 'w') as f:
            # Create a group for this model
            group_name = f"pca_unified_n_{len(self.pca_components)}"
            g = f.create_group(group_name)
            
            # Save PCA components and parameters
            g.create_dataset('pca_input_mean', data=np.array(self.pca_input_mean))
            g.create_dataset('pca_components', data=np.array(self.pca_components))
            g.create_dataset('eigenvalues', data=np.array(self.eigenvalues))
            
            # Save normalization parameters as attributes
            if self.norm_params:
                for key, value in self.norm_params.items():
                    g.attrs[key] = np.array(value) if hasattr(value, '__array__') else value
            
            # Save whitening flag
            g.attrs['whitened'] = self.whitened
            
            # Set as latest group
            f.attrs['latest_group'] = group_name
    
    @classmethod
    def load_model(cls, path: str, pca_group: Optional[str] = None, 
                   whitened: bool = True, **kwargs):
        """Load PCA model from HDF5 file.
        
        Args:
            path: Path to the PCA model file (.h5)
            pca_group: Specific group name to load (if None, uses latest)
            whitened: Whether to apply whitening transformation
            **kwargs: Additional arguments (unused)
            
        Returns:
            PCALatentRepresentation instance
        """
        def _select_pca_group(f, group_name=None):
            """Helper to select appropriate PCA group from file."""
            if group_name and group_name in f:
                return f[group_name]
            if 'latest_group' in f.attrs and f.attrs['latest_group'] in f:
                return f[f.attrs['latest_group']]
            
            # Fall back to finding groups with 'n_' pattern
            groups = [k for k in f.keys() if isinstance(f.get(k, getclass=True), h5py.Group) and ('n_' in k)]
            if groups:
                import re
                def parse_n(s):
                    m = re.search(r'n_(\d+)$', s)
                    return int(m.group(1)) if m else -1
                gname = max(groups, key=parse_n)
                return f[gname]
            return f
        
        with h5py.File(path, 'r') as f:
            g = _select_pca_group(f, pca_group)
            
            pca_components = g['pca_components'][:]
            pca_input_mean = g['pca_input_mean'][:]
            eigenvalues = g['eigenvalues'][:]
            wavelengths = g['wavelengths'][:] if 'wavelengths' in g else None
            
            # Load normalization parameters from attributes
            norm_params = {}
            for attr_name in g.attrs:
                if attr_name not in ['whitened', 'method']:
                    norm_params[attr_name] = g.attrs[attr_name]
            
            # Check if whitening preference was saved
            saved_whitened = g.attrs.get('whitened', whitened)
            use_whitened = whitened if 'whitened' not in kwargs else saved_whitened
            
        return cls(pca_components, pca_input_mean, eigenvalues, norm_params, use_whitened, wavelengths)
    
    @property
    def method_name(self) -> str:
        """Return the method name."""
        return "pca"
    
    def get_explained_variance_ratio(self) -> jnp.ndarray:
        """Calculate explained variance ratio for each component.
        
        Returns:
            Array of explained variance ratios
        """
        total_variance = jnp.sum(self.eigenvalues)
        return self.eigenvalues / total_variance
    
    def get_cumulative_variance_ratio(self) -> jnp.ndarray:
        """Calculate cumulative explained variance ratio.
        
        Returns:
            Array of cumulative explained variance ratios
        """
        variance_ratios = self.get_explained_variance_ratio()
        return jnp.cumsum(variance_ratios)
    
    def truncate_components(self, n_components: int) -> 'PCALatentRepresentation':
        """Create a new instance with fewer components.
        
        Args:
            n_components: Number of components to keep
            
        Returns:
            New PCALatentRepresentation with truncated components
        """
        if n_components > self.latent_dim:
            raise ValueError(f"Cannot truncate to {n_components} components (have {self.latent_dim})")
        
        return PCALatentRepresentation(
            pca_components=self.pca_components[:n_components],
            pca_input_mean=self.pca_input_mean,
            eigenvalues=self.eigenvalues[:n_components],
            norm_params=self.norm_params,
            whitened=self.whitened
        )
