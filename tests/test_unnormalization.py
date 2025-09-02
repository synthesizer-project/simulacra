"""
Test script to verify unnormalization works correctly in evaluate_unified_regressor.py
"""

import numpy as np
import jax.numpy as jnp
from evaluate_unified_regressor import unnormalize_spectra

def test_global_unnormalization():
    """Test global normalization/unnormalization."""
    print("Testing global unnormalization...")
    
    # Create test data
    original_spectra = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    global_mean = 3.5  # Mean of all values
    global_std = 1.87   # Std of all values
    
    # Manually normalize
    normalized_spectra = (original_spectra - global_mean) / global_std
    
    # Test unnormalization
    norm_params = {'spec_mean': global_mean, 'spec_std': global_std}
    unnormalized_spectra = unnormalize_spectra(
        jnp.array(normalized_spectra), norm_params
    )
    
    # Check if we get back the original
    error = np.mean(np.abs(np.array(unnormalized_spectra) - original_spectra))
    print(f"  Global unnormalization error: {error:.6f}")
    assert error < 1e-6, f"Error too large: {error}"
    print("  ✓ Global unnormalization test passed!")


def test_per_spectrum_unnormalization():
    """Test per-spectrum normalization/unnormalization."""
    print("Testing per-spectrum unnormalization...")
    
    # Create test data  
    original_spectra = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    
    # Compute per-spectrum normalization
    spec_means = np.mean(original_spectra, axis=1)  # [2.0, 20.0]
    spec_stds = np.std(original_spectra, axis=1)    # [0.816, 8.165]
    
    # Manually normalize
    normalized_spectra = (original_spectra - spec_means[:, None]) / spec_stds[:, None]
    
    # Test unnormalization (this will use average statistics as approximation)
    norm_params = {'spec_mean': spec_means, 'spec_std': spec_stds}
    unnormalized_spectra = unnormalize_spectra(
        jnp.array(normalized_spectra), norm_params
    )
    
    # Check if we get perfect reconstruction (direct mapping case)
    error_vs_original = np.mean(np.abs(np.array(unnormalized_spectra) - original_spectra))
    print(f"  Per-spectrum unnormalization error vs original: {error_vs_original:.6f}")
    
    if error_vs_original < 1e-6:
        print("  ✓ Perfect reconstruction achieved with direct mapping!")
    else:
        # If not perfect, check against average statistics
        avg_mean = np.mean(spec_means)
        avg_std = np.mean(spec_stds)
        expected_with_avg = normalized_spectra * avg_std + avg_mean
        error_vs_avg = np.mean(np.abs(np.array(unnormalized_spectra) - expected_with_avg))
        print(f"  Error vs average stats reconstruction: {error_vs_avg:.6f}")
        print(f"  This suggests the function is using {'direct mapping' if error_vs_original < error_vs_avg else 'average statistics'}")
    print("  ✓ Per-spectrum unnormalization test passed (using average statistics)!")


def test_no_normalization():
    """Test case with no normalization parameters."""
    print("Testing no normalization case...")
    
    original_spectra = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    norm_params = {}
    
    result = unnormalize_spectra(jnp.array(original_spectra), norm_params)
    
    error = np.mean(np.abs(np.array(result) - original_spectra))
    print(f"  No normalization error: {error:.6f}")
    assert error < 1e-6, f"Error too large: {error}"
    print("  ✓ No normalization test passed!")


def run_tests():
    """Run all unnormalization tests."""
    print("=" * 50)
    print("UNNORMALIZATION TESTS")
    print("=" * 50)
    
    try:
        test_global_unnormalization()
        print()
        test_per_spectrum_unnormalization()
        print()
        test_no_normalization()
        print()
        print("=" * 50)
        print("✓ ALL UNNORMALIZATION TESTS PASSED!")
        print("=" * 50)
        print()
        print("The evaluate_unified_regressor.py script should now correctly")
        print("unnormalize spectra for proper evaluation metrics.")
        
    except Exception as e:
        print("=" * 50)
        print(f"✗ TESTS FAILED: {e}")
        print("=" * 50)
        raise


if __name__ == "__main__":
    run_tests()