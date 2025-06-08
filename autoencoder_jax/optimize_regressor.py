import sys
import jax
import optuna
import train_regressor as rg
from train_autoencoder import load_model as load_autoencoder

def objective(trial, train_dataset, val_dataset, train_latent, val_latent, rng):
    """Optuna objective function for the regressor."""
    
    n_layers = trial.suggest_int('n_layers', 1, 5)
    hidden_dims = [trial.suggest_int(f'n_units_l{i}', 128, 1024) for i in range(n_layers)]

    # Safeguard: If the suggestion logic were to allow n_layers=0,
    # this would prevent an empty model from being created.
    if not hidden_dims:
        raise optuna.exceptions.TrialPruned("Cannot create a regressor with no hidden layers.")

    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'parametric_gated'])
    
    model = rg.RegressorMLP(
        hidden_dims=hidden_dims,
        latent_dim=train_latent.shape[1],
        dropout_rate=dropout_rate,
        activation_name=activation_function
    )
    
    _, _, _, best_val_loss = rg.train_and_evaluate(
        model=model, train_dataset=train_dataset, val_dataset=val_dataset,
        train_latent=train_latent, val_latent=val_latent, num_epochs=100,
        batch_size=64, learning_rate=learning_rate, weight_decay=weight_decay,
        rng=rng, trial=trial, verbose=False
    )
    
    return best_val_loss

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python optimize_regressor.py <grid_dir> <grid_name> <autoencoder_model_path>")
        sys.exit(1)

    grid_dir, grid_name, autoencoder_model_path = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # Load data
    train_dataset, val_dataset, _ = rg.load_data_regressor(grid_dir, grid_name, n_samples=int(1e4))
    
    # Load autoencoder and create latent representations
    print(f"Loading autoencoder from {autoencoder_model_path}...")
    autoencoder, autoencoder_state = load_autoencoder(autoencoder_model_path)
    train_latent = rg.create_latent_representations(autoencoder, autoencoder_state, train_dataset)
    val_latent = rg.create_latent_representations(autoencoder, autoencoder_state, val_dataset)
    
    # Create a study
    rng = jax.random.PRNGKey(42)
    study = optuna.create_study(direction='minimize')
    
    # Start optimization
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, train_latent, val_latent, rng),
        n_trials=100
    )
    
    print("\nOptimization finished. Retraining the best model...")
    best_params = study.best_params
    best_hidden_dims = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]
    
    best_model = rg.RegressorMLP(
        hidden_dims=best_hidden_dims,
        latent_dim=autoencoder.latent_dim,
        dropout_rate=best_params['dropout_rate']
    )
    
    print("Training final regressor model...")
    rg.train_and_evaluate(
        model=best_model, train_dataset=train_dataset, val_dataset=val_dataset,
        train_latent=train_latent, val_latent=val_latent, num_epochs=200,
        batch_size=64, learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'], rng=rng, verbose=True,
        save_path='models/regressor_model.msgpack'
    )

    print("\nBest trial found:")
    trial = study.best_trial
    print(f"  Value (MSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}") 