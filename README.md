# CWbeta-VAE-Interpretable-Speech-Early-Neurogenerative-Diseases

**This repository accompanies the paper:**

**A Saliency-Mapped CWβ-VAE Framework for Interpretable Speech Biomarkers in Early Parkinson’s Disease and Essential Tremor**

*[Author Names]**[Conference/Journal Name]*, [Year] [![DOI](https://img.shields.io/badge/DOI-xxxxx-blue)](https://doi.org/xxxxx)

## Overview

Differentiating Parkinson's Disease (PD) from Essential Tremor (ET) remains challenging due to overlapping speech impairments, particularly in early stages. This repository provides a complete implementation of an interpretable speech-based framework for digital biomarker discovery combining:

- **Conditional Weighted β-VAE (CWβ-VAE)** for learning disentangled latent representations
- **Cost-sensitive class weights** to address cohort imbalance (14 HC, 41 PD, 35 ET)
- **Sigmoid KL annealing** to prevent posterior collapse
- **Gradient-based saliency mapping** for population-level biomarker identification
- **XGBoost classifier** achieving 96% accuracy for PD vs. ET differentiation
  
## Pipeline Architecture

1. **Acoustic Preprocessing:** Raw speech signals are transformed into unified $224 \times 224$ spectro-temporal slices.
2. **Generative Bottlenecking:** The $CW\beta$-VAE maps spectral densities into a continuous, class-conditioned $d=32$ latent space.
3. **Saliency Attribution:** Backpropagated gradients map structural feature dependencies across five primary bio-bands:
   ```text
   * Rhythm/Tremor ($0$–$10$ Hz)
   * Phonatory Source ($80$–$600$ Hz)
   * Lower Articulatory ($0.6$–$1.8$ kHz)
   * Upper Clarity ($1.8$–$3.0$ kHz)
   * Spectral Noise ($3.0$–$4.0$ kHz)
   ```
5. **Statistical Stratification:** Non-parametric permutation frameworks mapping univariate boundaries vs. global multivariate accuracy anomalies.

## Repository Structure

```text
├── data/                   # Sample data or scripts to download the dataset
├── feature_extraction/     # Scripts for audio processing and feature extraction 
├── models/                 # Machine learning architectures and trained weights
├── notebooks/              # Jupyter notebooks for data exploration and XAI visualization
├── results/                # Output logs, figures, and performance metrics
├── requirements.txt        # Package dependencies
└── main.py                 # Main execution script for training and evaluation

```
### Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate cwbeta-vae
pip install -r requirements.txt
```
GPU is required for training Raw Mel Spectrogram + XGboot

## 📝 Citation
```bibtex
@article{[citation-key],
  title   = {[A Saliency-Mapped CWβ-VAE Framework for Interpretable Speech Biomarkers in Early Parkinson’s Disease and Essential Tremor]},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {[Year]}
}
```
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MIT (see [LICENSE](LICENSE)).
