import h5py
import jax.numpy as jnp


class SpectralDatasetJAX:
    def __init__(
        self,
        h5_path='example_grid.hdf5',
        spec_type='incident',
        parent_dataset=None,
        split=None,
        ages='log10age',
        metallicity='metallicity',
    ):

        if parent_dataset is not None:
            self.spectra = parent_dataset.spectra
            self.wavelength = parent_dataset.wavelength
            self.ages = parent_dataset.ages
            self.metallicities = parent_dataset.metallicities
            self.conditions = parent_dataset.conditions
            self.n_wavelength = parent_dataset.n_wavelength
            self.n_age = parent_dataset.n_age
            self.n_met = parent_dataset.n_met
        else:
            with h5py.File(h5_path, 'r') as f:
                self.spectra = jnp.array(f[f'spectra/{spec_type}'][:], dtype=jnp.float32)
                self.wavelength = jnp.array(f['spectra/wavelength'][:], dtype=jnp.float32)
                self.ages = jnp.array(f[f'axes/{ages}'][:], dtype=jnp.float32)
                self.metallicities = jnp.array(f[f'axes/{metallicity}'][:], dtype=jnp.float32)

            # Filter spectra and wavelength
            mask = (self.wavelength > 1000) & (self.wavelength < 10000)
            self.spectra = self.spectra[:, :, mask]
            self.wavelength = self.wavelength[mask]

            # Get dimensions
            self.n_age, self.n_met, self.n_wavelength = self.spectra.shape

        if split is not None:
            self.spectra = self.spectra[split]
            self.ages = self.ages[split]
            self.metallicities = self.metallicities[split]
            self.conditions = self.conditions[split]
        
        if parent_dataset is None:
            # Normalize parameters
            self.ages = (self.ages - self.ages.mean()) / self.ages.std()
            self.metallicities = (self.metallicities - self.metallicities.mean()) / self.metallicities.std()
            
            # Create all combinations of parameters
            self.conditions = jnp.stack(jnp.meshgrid(self.ages, self.metallicities, indexing='ij')).reshape(2, -1).T

            # Reshape spectra to match conditions
            self.spectra = self.spectra.reshape(-1, self.n_wavelength)

            # Log and normalize spectra
            self.spectra = jnp.log10(self.spectra)
            self.spectra = (self.spectra - self.spectra.mean(axis=1, keepdims=True)) / self.spectra.std(axis=1, keepdims=True)
    
    def __len__(self):
        return len(self.conditions)
    
    def __getitem__(self, idx):
        return self.conditions[idx], self.spectra[idx]
    
