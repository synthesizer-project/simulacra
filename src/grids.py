from scipy.stats import qmc
import jax.numpy as jnp
import numpy as np

from synthesizer.grid import Grid
from synthesizer.particle import Stars
from synthesizer.emission_models import IncidentEmission
from unyt import Msun, yr, angstrom


def LHGridSpectra(
    grid_dir,
    grid_name,
    num_samples=1000,
    seed=None,
    wl_min: float | None = None,
    wl_max: float | None = None,
):
    grid = Grid(grid_dir=grid_dir, grid_name=grid_name, read_lines=False)

    N = num_samples
    age_lims = (np.log10(float(grid.ages.min().value)), np.log10(float(grid.ages.max().value)))
    met_lims = (float(grid.metallicities.min()), float(grid.metallicities.max()))

    print("Age limits: ", age_lims)
    print("Metallicity limits: ", met_lims)

    sampler = qmc.LatinHypercube(d=2, seed=seed)
    samples = qmc.scale(sampler.random(n=N), (age_lims[0], met_lims[0]), (age_lims[1], met_lims[1]))
    
    initial_masses = np.ones(N) * Msun

    stars = Stars(
        initial_masses=initial_masses,
        ages=10**samples[:, 0] * yr,
        metallicities=samples[:, 1],
    )

    emodel = IncidentEmission(grid, per_particle=True)
    spec = stars.get_spectra(emodel)

    # Apply wavelength limits (Å). Defaults to [2400, 10000] Å if not provided.
    wl_lo = 2400.0 if wl_min is None else float(wl_min)
    wl_hi = 10000.0 if wl_max is None else float(wl_max)
    # Use inclusive bounds when explicit limits are provided to match saved wavelength grids (e.g., PCA models)
    if wl_min is not None or wl_max is not None:
        mask = (grid.lam >= wl_lo * angstrom) & (grid.lam <= wl_hi * angstrom)
    else:
        mask = (grid.lam > wl_lo * angstrom) & (grid.lam < wl_hi * angstrom)
    spectra = spec.lnu[:, mask]
    wavelength = grid.lam[mask]

    ages = samples[:, 0]
    metallicities = samples[:, 1]

    return spectra, wavelength, ages, metallicities


class SpectralDatasetSynthesizer:
    def __init__(
        self,
        grid_dir=None,
        grid_name=None,
        num_samples=1000,
        parent_dataset=None,
        split=None,
        norm='per-spectra',
        seed=None,
        true_spec_mean=None,
        true_spec_std=None,
        compute_lambda_stats=False,
        zscore_mean_lambda=None,
        zscore_std_lambda=None,
        wl_min: float | None = None,
        wl_max: float | None = None,
    ):

        if parent_dataset is not None:
            # Inherit all data and parameters from the parent dataset
            self.spectra = parent_dataset.spectra
            self.wavelength = parent_dataset.wavelength
            self.ages = parent_dataset.ages
            self.metallicities = parent_dataset.metallicities
            self.conditions = parent_dataset.conditions
            self.n_wavelength = parent_dataset.n_wavelength
            
            # Carry over normalization parameters from the parent for physical params
            self.age_mean = parent_dataset.age_mean
            self.age_std = parent_dataset.age_std
            self.met_mean = parent_dataset.met_mean
            self.met_std = parent_dataset.met_std
            self.true_spec_mean = parent_dataset.true_spec_mean
            self.true_spec_std = parent_dataset.true_spec_std
            
        else:
            # Load raw data if this is a new dataset
            self.spectra, self.wavelength, self.ages, self.metallicities = LHGridSpectra(
                grid_dir,
                grid_name,
                num_samples,
                seed=seed,
                wl_min=wl_min,
                wl_max=wl_max,
            )
            # Persist wavelength limits for downstream consumers
            self.wl_min = float(self.wavelength.min())
            self.wl_max = float(self.wavelength.max())
            self.n_wavelength = self.spectra.shape[1]

            # Reshape and log-transform spectra
            self.spectra = self.spectra.reshape(-1, self.n_wavelength)
            self.spectra = jnp.log10(self.spectra)
            # Track normalization mode used for spectra
            self.norm_mode = norm if norm else 'none'

            # --- Pre-computation before any splitting ---
            # Store normalization params for physical parameters
            self.age_mean, self.age_std = self.ages.mean(), self.ages.std()
            self.met_mean, self.met_std = self.metallicities.mean(), self.metallicities.std()

            # Normalize physical parameters
            norm_ages = (self.ages - self.age_mean) / self.age_std
            norm_mets = (self.metallicities - self.met_mean) / self.met_std
        
            # Create conditions from normalized parameters
            self.conditions = jnp.stack([norm_ages, norm_mets]).T

            if norm:
                if norm == 'per-spectra':
                    # Store the true per-spectrum mean and std for the normalization MLP to learn
                    self.true_spec_mean = self.spectra.mean(axis=1, keepdims=True)
                    self.true_spec_std = self.spectra.std(axis=1, keepdims=True)
                elif norm == 'global':
                    if true_spec_mean is not None and true_spec_std is not None:
                        # Use pre-computed normalization values if provided
                        print("Using pre-computed global normalization values.")
                        self.true_spec_mean = true_spec_mean
                        self.true_spec_std = true_spec_std
                    else:
                        # Otherwise, compute them from the loaded data
                        self.true_spec_mean = self.spectra.mean()  # axis=0, 
                        self.true_spec_std = self.spectra.std()  # axis=0, 
                elif norm == 'zscore':
                    # Two-stage: global scalar normalization, then per-wavelength z-score
                    if true_spec_mean is not None and true_spec_std is not None:
                        print("Using pre-computed global normalization values.")
                        self.true_spec_mean = true_spec_mean
                        self.true_spec_std = true_spec_std
                    else:
                        self.true_spec_mean = self.spectra.mean()
                        self.true_spec_std = self.spectra.std()
                    Xg = (self.spectra - self.true_spec_mean) / self.true_spec_std

                    if zscore_mean_lambda is not None and zscore_std_lambda is not None:
                        self.lambda_mean = zscore_mean_lambda
                        self.lambda_std = zscore_std_lambda
                        print("Using provided per-wavelength z-score parameters.")
                    else:
                        self.lambda_mean = Xg.mean(axis=0)
                        self.lambda_std = Xg.std(axis=0)
                    eps = 1e-6
                    self.lambda_std = jnp.where(self.lambda_std < eps, eps, self.lambda_std)
                    self.spectra = (Xg - self.lambda_mean) / self.lambda_std
                else:
                    raise ValueError(f"Invalid normalization method: {norm}")

                # Perform per-spectrum normalization for the autoencoder
                if norm in ('per-spectra', 'global'):
                    self.spectra = (self.spectra - self.true_spec_mean) / self.true_spec_std
            else:
                self.true_spec_mean = None
                self.true_spec_std = None

            # Optionally compute per-wavelength stats after normalization
            if compute_lambda_stats:
                self.lambda_mean = self.spectra.mean(axis=0)
                self.lambda_std = self.spectra.std(axis=0)


        if split is not None:
            # Apply the split to all relevant arrays
            self.spectra = self.spectra[split]
            self.ages = self.ages[split]
            self.metallicities = self.metallicities[split]
            self.conditions = self.conditions[split]
            # Also split the true mean/std when they are per-sample arrays.
            # In global mode, these may be scalars; keep them as-is.
            if self.true_spec_mean is not None:
                mean_arr = np.asarray(self.true_spec_mean)
                std_arr = np.asarray(self.true_spec_std)
                if mean_arr.ndim == 0:
                    # Global scalar normalization: leave unchanged
                    self.true_spec_mean = float(mean_arr)
                    self.true_spec_std = float(std_arr)
                else:
                    self.true_spec_mean = self.true_spec_mean[split]
                    self.true_spec_std = self.true_spec_std[split]

    def unnormalize_age(self, norm_age):
        """Un-normalizes a single age value."""
        return (norm_age * self.age_std) + self.age_mean

    def unnormalize_metallicity(self, norm_met):
        """Un-normalizes a single metallicity value."""
        return (norm_met * self.met_std) + self.met_mean
    
    def __len__(self):
        return len(self.conditions)
    
    def __getitem__(self, idx):
        return self.conditions[idx], self.spectra[idx]


if __name__ == '__main__':
    spectra, conditions, wavelength, ages, metallicities = LHGridSpectra()

    print(spectra.shape)
    print(conditions.shape)
    print(wavelength.shape)
    print(ages.shape)
    print(metallicities.shape)
