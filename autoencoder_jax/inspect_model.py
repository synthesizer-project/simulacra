import sys
from flax import serialization

def inspect_hyperparams(model_path):
    """Safely loads a model file and prints its stored hyperparameters."""
    try:
        with open(model_path, 'rb') as f:
            bundled_data = serialization.from_bytes(None, f.read())
        
        if 'hyperparams' not in bundled_data:
            print(f"Error: No 'hyperparams' key found in {model_path}")
            return
            
        hyperparams = bundled_data['hyperparams']
        
        print(f"--- Hyperparameters in {model_path} ---")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
        print("-------------------------------------------")

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_model.py <path_to_model_file>")
        sys.exit(1)
        
    model_file = sys.argv[1]
    inspect_hyperparams(model_file) 