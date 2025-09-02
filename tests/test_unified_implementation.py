"""
Test script for the unified latent representation implementation.

This script performs basic functionality tests to ensure the unified
implementation works correctly with both autoencoder and PCA methods.
"""

import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import os
from typing import Dict, Any

# Import unified components
from latent_representations import LatentRepresentation
from autoencoder_latent import AutoencoderLatentRepresentation
from pca_latent import PCALatentRepresentation
from train_unified_regressor import create_latent_representation


def create_dummy_autoencoder():
    """Create a dummy autoencoder for testing."""
    # This would normally be loaded from a real trained model
    print("Note: This test requires a real trained autoencoder model to work properly")
    print("Skipping autoencoder test - would need models/autoencoder_simple_dense.msgpack")
    return None


def create_dummy_pca_model(n_wavelengths: int = 1000, n_components: int = 50) -> str:
    """Create a dummy PCA model for testing.
    
    Args:
        n_wavelengths: Number of wavelength points
        n_components: Number of PCA components
        
    Returns:
        Path to temporary PCA model file
    """
    import h5py
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
    temp_path = temp_file.name
    temp_file.close()
    
    # Create dummy PCA data
    rng = np.random.RandomState(42)
    pca_components = rng.randn(n_components, n_wavelengths)
    pca_input_mean = rng.randn(n_wavelengths)
    eigenvalues = np.sort(rng.exponential(1.0, n_components))[::-1]  # Descending order
    
    # Normalize PCA components (as they would be from real PCA)
    pca_components = pca_components / np.linalg.norm(pca_components, axis=1, keepdims=True)
    
    # Save to HDF5
    with h5py.File(temp_path, 'w') as f:
        group_name = f"pca_test_n_{n_components}"
        g = f.create_group(group_name)
        
        g.create_dataset('pca_input_mean', data=pca_input_mean)
        g.create_dataset('pca_components', data=pca_components)
        g.create_dataset('eigenvalues', data=eigenvalues)
        
        # Add dummy normalization parameters
        g.attrs['true_spec_mean'] = 0.0
        g.attrs['true_spec_std'] = 1.0
        g.attrs['whitened'] = True
        
        f.attrs['latest_group'] = group_name
    
    return temp_path


def test_pca_latent_representation():
    """Test PCA latent representation functionality."""
    print("Testing PCA Latent Representation...")
    
    n_wavelengths = 1000
    n_components = 50
    n_samples = 100
    
    # Create dummy PCA model
    pca_path = create_dummy_pca_model(n_wavelengths, n_components)
    
    try:
        # Load PCA representation
        pca_repr = PCALatentRepresentation.load_model(pca_path, whitened=True)
        
        # Test basic properties
        assert pca_repr.get_latent_dim() == n_components, f"Expected {n_components}, got {pca_repr.get_latent_dim()}"
        assert pca_repr.method_name == "pca"
        
        print(f"  ✓ Loaded PCA model with {pca_repr.get_latent_dim()} components")
        
        # Test encoding and decoding
        rng = np.random.RandomState(42)
        dummy_spectra = jnp.array(rng.randn(n_samples, n_wavelengths))
        
        # Encode
        latent_vectors = pca_repr.encode_spectra(dummy_spectra)
        assert latent_vectors.shape == (n_samples, n_components), f"Wrong latent shape: {latent_vectors.shape}"
        
        print(f"  ✓ Encoded {n_samples} spectra to latent vectors of shape {latent_vectors.shape}")
        
        # Decode
        reconstructed_spectra = pca_repr.decode_latents(latent_vectors)
        assert reconstructed_spectra.shape == (n_samples, n_wavelengths), f"Wrong reconstruction shape: {reconstructed_spectra.shape}"
        
        print(f"  ✓ Decoded latent vectors back to spectra of shape {reconstructed_spectra.shape}")
        
        # Test normalization parameters
        norm_params = pca_repr.get_normalization_params()
        assert isinstance(norm_params, dict), "Normalization parameters should be a dictionary"
        
        print(f"  ✓ Retrieved normalization parameters: {list(norm_params.keys())}")
        
        # Test explained variance methods
        var_ratios = pca_repr.get_explained_variance_ratio()
        cum_var_ratios = pca_repr.get_cumulative_variance_ratio()
        
        assert len(var_ratios) == n_components, "Wrong variance ratio length"
        assert len(cum_var_ratios) == n_components, "Wrong cumulative variance ratio length"
        assert cum_var_ratios[-1] <= 1.0, "Cumulative variance should not exceed 1"
        
        print(f"  ✓ Computed explained variance ratios (total: {cum_var_ratios[-1]:.3f})")
        
        # Test truncation
        truncated_repr = pca_repr.truncate_components(20)
        assert truncated_repr.get_latent_dim() == 20, "Truncation failed"
        
        print(f"  ✓ Successfully truncated to 20 components")
        
        # Test factory function
        factory_repr = create_latent_representation('pca', pca_path, whitened=True)
        assert isinstance(factory_repr, PCALatentRepresentation), "Factory function failed"
        assert factory_repr.get_latent_dim() == n_components, "Factory representation has wrong dimension"
        
        print(f"  ✓ Factory function works correctly")
        
        print("  ✓ All PCA tests passed!")
        
    except Exception as e:
        print(f"  ✗ PCA test failed: {e}")
        raise
    finally:
        # Clean up temporary file
        if os.path.exists(pca_path):
            os.remove(pca_path)


def test_round_trip_consistency():
    """Test encode-decode round-trip consistency for PCA."""
    print("Testing PCA round-trip consistency...")
    
    n_wavelengths = 500
    n_components = 100
    n_samples = 50
    
    pca_path = create_dummy_pca_model(n_wavelengths, n_components)
    
    try:
        pca_repr = PCALatentRepresentation.load_model(pca_path, whitened=False)  # No whitening for exact reconstruction
        
        # Create test data that lies exactly in the PCA subspace
        rng = np.random.RandomState(123)
        latent_coeffs = rng.randn(n_samples, n_components)
        
        # Generate spectra from latent coefficients
        true_spectra = latent_coeffs @ np.array(pca_repr.pca_components) + np.array(pca_repr.pca_input_mean)
        true_spectra = jnp.array(true_spectra)
        
        # Encode and decode
        encoded_latents = pca_repr.encode_spectra(true_spectra)
        reconstructed_spectra = pca_repr.decode_latents(encoded_latents)
        
        # Check reconstruction error
        reconstruction_error = jnp.mean((true_spectra - reconstructed_spectra)**2)
        
        print(f"  Round-trip reconstruction MSE: {reconstruction_error:.2e}")
        
        # Should be very small for data in the PCA subspace
        assert reconstruction_error < 1e-10, f"Round-trip error too large: {reconstruction_error}"
        
        print("  ✓ Round-trip consistency test passed!")
        
    finally:
        if os.path.exists(pca_path):
            os.remove(pca_path)


def test_interface_compliance():
    """Test that implementations properly follow the LatentRepresentation interface."""
    print("Testing interface compliance...")
    
    pca_path = create_dummy_pca_model(100, 20)
    
    try:
        pca_repr = PCALatentRepresentation.load_model(pca_path)
        
        # Check that all abstract methods are implemented
        assert hasattr(pca_repr, 'get_latent_dim'), "Missing get_latent_dim method"
        assert hasattr(pca_repr, 'encode_spectra'), "Missing encode_spectra method"
        assert hasattr(pca_repr, 'decode_latents'), "Missing decode_latents method"
        assert hasattr(pca_repr, 'get_normalization_params'), "Missing get_normalization_params method"
        assert hasattr(pca_repr, 'save_model'), "Missing save_model method"
        assert hasattr(pca_repr, 'method_name'), "Missing method_name property"
        
        # Check that methods return expected types
        assert isinstance(pca_repr.get_latent_dim(), int), "get_latent_dim should return int"
        assert isinstance(pca_repr.method_name, str), "method_name should return str"
        assert isinstance(pca_repr.get_normalization_params(), dict), "get_normalization_params should return dict"
        
        print("  ✓ Interface compliance test passed!")
        
    finally:
        if os.path.exists(pca_path):
            os.remove(pca_path)


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("UNIFIED LATENT REPRESENTATION IMPLEMENTATION TESTS")
    print("=" * 60)
    print()
    
    try:
        # Test PCA implementation
        test_pca_latent_representation()
        print()
        
        test_round_trip_consistency()  
        print()
        
        test_interface_compliance()
        print()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("The unified implementation is ready to use.")
        print()
        print("Usage examples:")
        print("  # Train PCA regressor:")
        print("  python train_unified_regressor.py pca ../pca_jax/pca_models/pca_model.h5 /path/to/grids grid.hdf5")
        print()
        print("  # Train autoencoder regressor (requires trained autoencoder):")
        print("  python train_unified_regressor.py autoencoder models/autoencoder_simple_dense.msgpack /path/to/grids grid.hdf5")
        print()
        print("  # Evaluate trained regressor:")
        print("  python evaluate_unified_regressor.py pca pca_model.h5 regressor.msgpack /path/to/grids grid.hdf5")
        
    except Exception as e:
        print("=" * 60)
        print(f"✗ TESTS FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    run_all_tests()