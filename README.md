# Simulacra
Scripts for training and using emulated grids in Synthesizer.

Neural network-based emulation of stellar population synthesis spectra using JAX/Flax. This repository provides fast, accurate alternatives to expensive stellar population synthesis calculations by learning to predict log-flux spectra from physical parameters (age and metallicity).

## Overview

This codebase implements **spectrum emulation for stellar population synthesis models** (specifically BC03 Bruzual & Charlot 2003 models) using machine learning. Instead of running expensive stellar population synthesis calculations, trained neural networks can rapidly predict synthetic spectra for given age and metallicity combinations.

## Architecture

The system uses a sophisticated **three-component neural architecture**:

### 1. Autoencoder Component
- **Purpose**: Learns compact latent representations of normalized log-spectra
- **Architecture**: Dense or convolutional encoder-decoder networks
- **Features**: Configurable latent dimensions, dropout, batch normalization, custom activations
- **Output**: Compressed latent space representation of spectral shapes

### 2. Normalization MLP
- **Purpose**: Predicts per-spectrum normalization parameters (mean, std) from physical conditions
- **Input**: Normalized age and metallicity
- **Output**: Spectrum-level mean and standard deviation for denormalization
- **Architecture**: Small MLP for parameter prediction

### 3. Regression Network
- **Purpose**: Maps physical parameters directly to autoencoder latent space
- **Input**: Normalized age and metallicity
- **Output**: Latent vector that can be decoded to spectrum
- **Training**: Trained against autoencoder-generated latent vectors

## Workflow

The complete spectrum generation pipeline:
1. **Input**: Physical parameters (age, metallicity)
2. **Normalization MLP**: Predicts spectrum normalization parameters
3. **Regressor**: Maps parameters to autoencoder latent space
4. **Autoencoder Decoder**: Reconstructs normalized spectrum from latent vector
5. **Denormalization**: Apply predicted normalization to get final log-flux spectrum

## Key Features

- **JAX/Flax Implementation**: Modern, GPU-accelerated neural networks
- **Multiple Normalization Schemes**: Per-spectra, global, and z-score normalization options
- **Custom Activation Functions**: Parametric gated activations for improved expressivity
- **Comprehensive Evaluation**: Detailed metrics, visualizations, and error analysis
- **Hyperparameter Optimization**: Optuna-based automated tuning
- **Modular Architecture**: Separable training and evaluation of different components
- **Comparison Framework**: PCA + MLP baseline implementation for performance comparison

## Applications

This emulation approach is particularly valuable for:
- **Bayesian Inference**: Fast likelihood evaluations for parameter estimation
- **Mock Catalog Generation**: Rapid generation of large synthetic datasets
- **Parameter Space Exploration**: Efficient sampling of age-metallicity parameter combinations
- **Real-time Analysis**: Interactive spectral analysis and fitting
>>>>>>> 9d3aa50 (update autoencoder)
