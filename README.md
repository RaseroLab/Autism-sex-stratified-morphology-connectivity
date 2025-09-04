# Brain Morphology Network Alterations in Adolescents with Autism Spectrum Disorder: A Sex-Stratified Study

This repository contains the scripts and supporting code used in the analysis pipeline for the study:

**Title**: Brain Morphology Network Alterations in Adolescents with Autism Spectrum Disorder: A Sex-Stratified Study. 

**Authors**: Nooshin Safari, Isaac Levy, Aiying Zhang, Chirag Agarwal, Jesus M. Cortes, Kevin A. Pelphrey, John Darrell Van Horn, Javier Rasero.

**Link (biorXiv)**: https://www.biorxiv.org/cgi/content/short/2025.08.28.672884v1


## Summary 

The goal of this project is to characterize sex-specific alterations in brain morphology-based connectivity using multivariate morphometric features derived from T1-weighted MRI.

### Methods

- **Preprocessing and Feature Extraction**: Structural T1-weighted MRI scans were processed with FreeSurfer (v6.0) to extract: Cortical thickness, Surface area, Sulcal depth, Mean curvature, Gray matter volume

- **Connectivity Estimation**: The Morphometric INverse Divergence (MIND) framework was used to estimate inter-regional similarity. Connectivity matrices (68×68 for cortex, 82×82 for whole brain) were computed using symmetrized multivariate KL divergence.

- **Statistical Analysis**: Network-Based Statistics (NBS) was applied separately for males and females to detect clusters of altered connectivity between ASD and typically developing control (TDC) groups. Analyses controlled for age, intracranial volume, and site. A leave-one-feature-out approach assessed the importance of each morphometric feature via changes in AUC in ROC classification. Partial correlations were performed between subnetwork strength and behavioral scores (ADI-R, ADOS-2, CBCL) in the ASD group.

Here a methodological sketch of the project:

![methodology](https://github.com/RaseroLab/Autism-sex-stratified-morphology-connectivity/blob/main/figures/Sup-figures/method-fig.png)
