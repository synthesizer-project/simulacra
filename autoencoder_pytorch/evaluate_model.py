import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from spectrum_autoencoder import SpectrumAutoencoderEmulator
from grids import SpectralDataset

def plot_reconstruction(model, test_loader, dataset, device, num_samples=4):
    model.eval()
    with torch.no_grad():
        param_batch, spectrum_batch = next(iter(test_loader))
        spectrum_batch = spectrum_batch.to(device)
        param_batch = param_batch.to(device)

        ages = dataset.ages.cpu().numpy()
        metallicities = dataset.metallicities.cpu().numpy()
        
        # Get predictions
        pred_spectrum = model(spectrum_batch, param_batch)
        
        # Plot results
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            ax = axes[i]
            true = spectrum_batch[i].cpu().numpy()
            pred = pred_spectrum[i].cpu().numpy()
            
            # Denormalize conditions
            age = param_batch[i, 0].cpu().numpy() * ages.std() + ages.mean()
            met = param_batch[i, 1].cpu().numpy() * metallicities.std() + metallicities.mean()
            
            ax.plot(true, label='True', alpha=0.7, color='blue')
            ax.plot(pred, label='Reconstructed', alpha=0.7, color='red')
            
            # Add error
            error = np.abs(true - pred)
            ax.fill_between(np.arange(len(error)), 
                           true - error, 
                           true + error, 
                           alpha=0.2, 
                           color='gray', 
                           label='Error')
            
            ax.set_title(f'Age: {age:.2f} Gyr, Metallicity: {met:.2f}')
            ax.set_xlabel('Wavelength Index')
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
        plt.savefig('figures/autoencoder_reconstruction.png', dpi=300, bbox_inches='tight')
        plt.close()

def load_model(model_path, dataset, latent_dim):
    model = SpectrumAutoencoderEmulator(
        spectrum_dim=dataset.n_wavelength,
        latent_dim=latent_dim,
        param_dim=dataset.conditions.shape[1]
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    return model        

if __name__ == "__main__":
    latent_dim = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    grid_dir = '../synthesizer_grids/grids/'
    dataset = SpectralDataset(f'{grid_dir}/bc03-2016-Miles_chabrier-0.1,100.hdf5')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = load_model('models/best_autoencoder.pt', dataset, latent_dim=latent_dim)
    
    plot_reconstruction(model, dataloader, dataset, device)