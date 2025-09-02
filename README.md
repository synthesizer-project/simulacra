# Simulacra
Scripts for training and using emulated grids in Synthesizer.

Neural network-based emulation of stellar population synthesis spectra using JAX/Flax. This repository provides fast, accurate alternatives to expensive stellar population synthesis calculations by learning to predict log-flux spectra from physical parameters (age and metallicity).

## Overview

This codebase implements **spectrum emulation for stellar population synthesis models** (specifically BC03 Bruzual & Charlot 2003 models) using machine learning. Instead of running expensive stellar population synthesis calculations, trained neural networks can rapidly predict synthetic spectra for given age and metallicity combinations.

## Architecture

## Workflow

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

