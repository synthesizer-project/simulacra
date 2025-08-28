import sys
sys.path.append('..')

import jax
import jax.numpy as jnp
import optuna
from grids import SpectralDatasetSynthesizer
import train_norm_mlp as mlp_trainer

def objective(trial, train_data, test_data, rng):
    """Optuna objective function for the NormalizationMLP."""
    
    # Suggest hyperparameters for the MLP
    n_layers = trial.suggest_int('n_layers', 1, 6)
    
    features = []
    for i in range(n_layers):
        # Suggest layer size, e.g., between 32 and 256
        layer_size = trial.suggest_int(f'n_units_l{i}', 32, 512)
        features.append(layer_size)
    
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Create MLP model with sampled hyperparameters
    model = mlp_trainer.NormalizationMLP(features=tuple(features))
    
    # Train and evaluate the model
    _, _, _, best_test_loss = mlp_trainer.train_and_evaluate(
        model=model,
        train_data=train_data,
        test_data=test_data,
        num_epochs=100, # Use a fixed number of epochs for optimization
        batch_size=batch_size,
        learning_rate=learning_rate,
        rng=rng,
        weight_decay=weight_decay,
        verbose=False, # Disable verbose output for trials
        patience=10, # Use early stopping
        trial=trial
    )
    
    return best_test_loss

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python optimize_mlp.py <grid_dir> <grid_name>")
        sys.exit(1)

    grid_dir, grid_name = sys.argv[1], sys.argv[2]
    
    # Load data
    N_samples = int(1e5)
    dataset = SpectralDatasetSynthesizer(grid_dir=grid_dir, grid_name=grid_name, num_samples=N_samples)
    rng = jax.random.PRNGKey(0)
    perm = jax.random.permutation(rng, len(dataset))
    split = int(0.8 * len(dataset))
    train_dataset = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[:split])
    test_dataset = SpectralDatasetSynthesizer(parent_dataset=dataset, split=perm[split:])

    # Prefetch data to device for the entire optimization study
    print("Prefetching data to the accelerator for optimization...")
    train_data = jax.device_put((
        jnp.array(train_dataset.conditions),
        jnp.array(train_dataset.true_spec_mean),
        jnp.array(train_dataset.true_spec_std)
    ))
    test_data = jax.device_put((
        jnp.array(test_dataset.conditions),
        jnp.array(test_dataset.true_spec_mean),
        jnp.array(test_dataset.true_spec_std)
    ))
    print("...done.")


    # Create and run the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_data, test_data, rng), n_trials=50)
    
    # --- Training and saving the best model ---
    print("\nOptimization finished. Retraining the best model...")
    best_params = study.best_params
    
    # Reconstruct the best architecture
    best_n_layers = best_params['n_layers']
    best_features = []
    for i in range(best_n_layers):
        best_features.append(best_params[f'n_units_l{i}'])
        
    best_model = mlp_trainer.NormalizationMLP(features=tuple(best_features))

    # Train the final model with the best hyperparameters
    _, _, _, _ = mlp_trainer.train_and_evaluate(
        model=best_model,
        train_data=train_data,
        test_data=test_data,
        num_epochs=200,  # Train for longer on the final model
        batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate'],
        rng=rng,
        weight_decay=best_params['weight_decay'],
        verbose=True,
        save_path='models/best_norm_mlp.msgpack'
    )

    # Print best trial results
    print("\nBest trial found:")
    trial = study.best_trial
    
    print(f"  Value (Loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}") 