import torch
import numpy as np
import matplotlib.pyplot as plt
from train_flow import SpectralDataset, create_flow_model
from torch.utils.data import DataLoader, random_split


def plot_spectra(true_spectra, predicted_spectra, conditions, dataset, wavelength, save_path='spectra_comparison.png'):
    n_samples = len(true_spectra)
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, (ax, true, pred, cond) in enumerate(zip(axes, true_spectra, predicted_spectra, conditions)):
        # Denormalize conditions
        age = cond[0] * dataset.ages.std() + dataset.ages.mean()
        met = cond[1] * dataset.metallicities.std() + dataset.metallicities.mean()
        
        # Plot
        ax.plot(wavelength, true, label='True', alpha=0.7, color='blue')
        ax.plot(wavelength, pred, label='Predicted', alpha=0.7, color='red')
        
        # Add error
        error = np.abs(true - pred)
        ax.fill_between(wavelength, 
                       true - error, 
                       true + error, 
                       alpha=0.2, 
                       color='gray', 
                       label='Error')
        
        ax.set_title(f'Age: {age:.2f} Gyr, Metallicity: {met:.2f}')
        ax.set_xlabel('Wavelength ($\\AA$)')
        ax.set_ylabel('Normalized Flux')
        ax.legend()
        
        # Add error metrics
        mse = np.mean((true - pred) ** 2)
        mae = np.mean(np.abs(true - pred))
        ax.text(0.02, 0.98, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(true_spectra, predicted_spectra, save_path='error_distribution.png'):
    errors = np.abs(true_spectra - predicted_spectra)
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors.flatten(), bins=50, alpha=0.7)
    plt.title('Distribution of Absolute Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def load_model(model_path, input_dim):
    model = create_flow_model('cpu', input_dim)
    model.load_state_dict(torch.load(model_path))
    return model

def generate_predictions(model, conditions, spectra_shape, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        conditions = conditions.to(device)
        # Generate random initial state with the same shape as spectra
        x0 = torch.randn(conditions.shape[0], spectra_shape, device=device)
        # Use t=1.0 for final prediction
        t = torch.ones(conditions.shape[0], device=device)
        # Generate prediction
        predictions = model(t, x0, conditions)
    
    return predictions.cpu()

def main():
    # Load dataset
    grid_dir = '../synthesizer_grids/grids/'
    dataset = SpectralDataset(f'{grid_dir}/bc03-2016-Miles_chabrier-0.1,100.hdf5')
    
    # Split into train and test sets (80-20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create dataloaders
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Load trained model
    model = load_model('models/spectral_flow_model.pt', dataset.n_wavelength)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_conditions, test_spectra = next(iter(test_loader))
    test_predictions = generate_predictions(model, test_conditions, dataset.n_wavelength)
    
    plot_spectra(
        test_spectra.numpy(),
        test_predictions.numpy(),
        test_conditions.numpy(),
        dataset,
        dataset.wavelength.numpy(),
        'figures/test_spectra_comparison.png'
    )
    
    # Evaluate on training set
    print("Evaluating on training set...")
    train_conditions, train_spectra = next(iter(train_loader))
    train_predictions = generate_predictions(model, train_conditions, dataset.n_wavelength)
    
    plot_spectra(
        train_spectra.numpy(),
        train_predictions.numpy(),
        train_conditions.numpy(),
        dataset,
        dataset.wavelength.numpy(),
        'figures/train_spectra_comparison.png'
    )
    
    # Plot error distributions
    print("Generating error distributions...")
    plot_error_distribution(
        test_spectra.numpy(),
        test_predictions.numpy(),
        'figures/test_error_distribution.png'
    )
    
    plot_error_distribution(
        train_spectra.numpy(),
        train_predictions.numpy(),
        'figures/train_error_distribution.png'
    )
    
    # Print overall metrics
    test_mse = np.mean((test_spectra.numpy() - test_predictions.numpy()) ** 2)
    test_mae = np.mean(np.abs(test_spectra.numpy() - test_predictions.numpy()))
    train_mse = np.mean((train_spectra.numpy() - train_predictions.numpy()) ** 2)
    train_mae = np.mean(np.abs(train_spectra.numpy() - train_predictions.numpy()))
    
    print("\nOverall Metrics:")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Train MAE: {train_mae:.4f}")

if __name__ == "__main__":
    main() 