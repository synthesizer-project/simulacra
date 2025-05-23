import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from grids import SpectralDataset

# === Spectrum Encoder ===
class SpectrumEncoder(nn.Module):
    def __init__(self, spectrum_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(spectrum_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# === Spectrum Decoder (Conditional on Parameters) ===
class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, param_dim, spectrum_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + param_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, spectrum_dim)
        )

    def forward(self, latent, params):
        x = torch.cat([latent, params], dim=1)
        return self.decoder(x)

# === Full Autoencoder + Emulator ===
class SpectrumAutoencoderEmulator(nn.Module):
    def __init__(self, spectrum_dim, latent_dim, param_dim):
        super().__init__()
        self.encoder = SpectrumEncoder(spectrum_dim, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, param_dim, spectrum_dim)

    def encode(self, spectrum):
        return self.encoder(spectrum)

    def decode(self, latent, params):
        return self.decoder(latent, params)

    def forward(self, spectrum, params):
        latent = self.encode(spectrum)
        return self.decode(latent, params)

def train_autoencoder(model, train_loader, val_loader, num_epochs, device, learning_rate=1e-3, patience=10, min_delta=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for param_batch, spectrum_batch in train_loader:
            spectrum_batch = spectrum_batch.to(device)
            param_batch = param_batch.to(device)
            
            optimizer.zero_grad()
            pred_spectrum = model(spectrum_batch, param_batch)
            loss = loss_fn(pred_spectrum, spectrum_batch)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for param_batch, spectrum_batch in val_loader:
                spectrum_batch = spectrum_batch.to(device)
                param_batch = param_batch.to(device)
                
                pred_spectrum = model(spectrum_batch, param_batch)
                loss = loss_fn(pred_spectrum, spectrum_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_autoencoder.pt')
        else:
            no_improve_epochs += 1
            
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"No improvement for {no_improve_epochs} epochs")
        
        # Early stopping
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('models/best_autoencoder.pt'))
    return train_losses, val_losses

if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 128
    batch_size = 32
    num_epochs = 50
    early_stopping_patience = 10
    early_stopping_min_delta = 1e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    grid_dir = '../synthesizer_grids/grids/'
    dataset = SpectralDataset(f'{grid_dir}/bc03-2016-Miles_chabrier-0.1,100.hdf5')
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = SpectrumAutoencoderEmulator(
        spectrum_dim=dataset.n_wavelength,
        latent_dim=latent_dim,
        param_dim=dataset.conditions.shape[1]
    ).to(device)
    
    # Train model
    train_losses, val_losses = train_autoencoder(
        model, train_loader, val_loader, num_epochs, device,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/autoencoder_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
