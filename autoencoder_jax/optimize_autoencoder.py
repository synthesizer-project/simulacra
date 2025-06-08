import sys
import jax
import optuna
import train_autoencoder as ae

def objective(trial, train_dataset, test_dataset, rng):
    """Optuna objective function."""
    
    # Suggest hyperparameters
    latent_dim = trial.suggest_int('latent_dim', 32, 256)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    
    encoder_features = []
    last_layer_size = 2048
    for i in range(n_layers):
        # Ensure the sampling range is valid (lower bound must be strictly less than upper bound)
        if last_layer_size <= latent_dim + 1:
            break # Stop adding layers if we can't make them smaller than the last one
        
        layer_size = trial.suggest_int(f'n_units_enc_l{i}', latent_dim + 1, last_layer_size -1)
        encoder_features.append(layer_size)
        last_layer_size = layer_size
    
    # If no layers were added (e.g., initial last_layer_size was too small),
    # we should handle this gracefully. For example, by pruning the trial.
    if not encoder_features:
        raise optuna.exceptions.TrialPruned("Could not form a valid encoder architecture.")
    
    decoder_features = encoder_features[::-1]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'parametric_gated'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Create model
    model = ae.SpectrumAutoencoder(
        spectrum_dim=train_dataset.n_wavelength,
        latent_dim=latent_dim,
        encoder_features=encoder_features,
        decoder_features=decoder_features,
        dropout_rate=dropout_rate,
        activation_name=activation_function
    )
    
    # Train model (without saving)
    _, _, _, best_test_loss = ae.train_and_evaluate(
        model=model,
        train_ds=train_dataset,
        test_ds=test_dataset,
        num_epochs=100,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rng=rng,
        weight_decay=weight_decay,
        trial=trial,
        verbose=False, # We don't want to spam the console during optimization
    )
    
    return best_test_loss

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python optimize_autoencoder.py <grid_dir> <grid_name>")
        sys.exit(1)

    grid_dir, grid_name = sys.argv[1], sys.argv[2]
    
    # Load data
    N_samples = int(1e4)
    train_dataset, test_dataset, rng = ae.load_data(grid_dir, grid_name, N_samples)

    # Create and run the study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_dataset, test_dataset, rng), n_trials=50)
    
    # --- Training and saving the best model ---
    print("\nOptimization finished. Retraining the best model...")
    best_params = study.best_params
    
    # Reconstruct the best architecture
    best_n_layers = best_params['n_layers']
    best_encoder_features = []
    for i in range(best_n_layers):
        best_encoder_features.append(best_params[f'n_units_enc_l{i}'])
        
    best_model = ae.SpectrumAutoencoder(
        spectrum_dim=train_dataset.n_wavelength,
        latent_dim=best_params['latent_dim'],
        encoder_features=best_encoder_features,
        decoder_features=best_encoder_features[::-1],
        dropout_rate=best_params['dropout_rate'],
    )

    # Train the final model with the best hyperparameters
    ae.train_and_evaluate(
        model=best_model,
        train_ds=train_dataset,
        test_ds=test_dataset,
        num_epochs=200,  # Train for longer on the final model
        batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate'],
        rng=rng,
        weight_decay=best_params['weight_decay'],
        verbose=True,
        save_path='models/best_autoencoder.msgpack'
    )

    # Print best trial results
    print("\nBest trial found:")
    trial = study.best_trial
    
    print(f"  Value (Loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}") 