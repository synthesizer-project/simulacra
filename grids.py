import torch
import h5py
from torch.utils.data import Dataset

class SpectralDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.spectra = torch.tensor(f['spectra/incident'][:], dtype=torch.float32)
            self.wavelength = torch.tensor(f['spectra/wavelength'][:], dtype=torch.float32)
            self.ages = torch.tensor(f['axes/ages'][:], dtype=torch.float32)
            self.metallicities = torch.tensor(f['axes/metallicities'][:], dtype=torch.float32)

        # Filter spectra and wavelength
        mask = (self.wavelength > 1000) & (self.wavelength < 10000)
        self.spectra = self.spectra[:, :, mask]
        self.wavelength = self.wavelength[mask]

        # Get dimensions
        self.n_age, self.n_met, self.n_wavelength = self.spectra.shape
        
        # Normalize parameters
        self.ages = (self.ages - self.ages.mean()) / self.ages.std()
        self.metallicities = (self.metallicities - self.metallicities.mean()) / self.metallicities.std()
        
        # Create all combinations of parameters
        self.conditions = torch.stack(torch.meshgrid(self.ages, self.metallicities, indexing='ij')).reshape(2, -1).T

        # Reshape spectra to match conditions
        self.spectra = self.spectra.reshape(-1, self.n_wavelength)

        # Log and normalize spectra
        self.spectra = torch.log10(self.spectra)
        self.spectra = (self.spectra - self.spectra.mean(dim=1, keepdim=True)) / self.spectra.std(dim=1, keepdim=True)
    
    def __len__(self):
        return len(self.conditions)
    
    def __getitem__(self, idx):
        return self.conditions[idx], self.spectra[idx]
    
