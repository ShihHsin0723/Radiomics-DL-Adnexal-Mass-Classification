# Radiomics-Incorporated Deep Learning for Adnexal Mass Classification
This project implements a radiomics-enhanced deep learning framework for the classification of benign and malignant adnexal masses using ultrasound images. By integrating radiomic feature maps with a pre-trained EfficientNet-B0 model, the approach aims to improve classification performance and evaluate robustness under varying levels of Speckle and Gaussian noise.

## Overview
- Dataset: 2,658 ultrasound images from 243 patients, each with corresponding segmentation masks.

- Baseline Model: Transfer learning using EfficientNet-B0 with grayscale ultrasound images.

- Radiomics-Incorporated Models: Two variants — one using full-image radiomic feature maps, another using lesion-segmented feature maps.

- Feature Selection: Brute-force search among 40 radiomic features to optimize classification performance.

- Evaluation: 5-fold cross-validation, independent test set analysis, bootstrap-based statistical testing, and noise robustness evaluation.

## Key Findings
- Full-image radiomics incorporation improved classification performance across all metrics compared to a baseline CNN.

- Restricting radiomics to the lesion area enhanced robustness to noise but reduced overall accuracy.

- A trade-off was observed between diagnostic performance under clean conditions and generalizability under noisy imaging scenarios.

## Technologies Used
- Python 3.10

- PyTorch 2.5.1

- PyRadiomics 3.1.0

- NumPy, SciPy, OpenCV

- Ubuntu 22.04, NVIDIA GTX 1080 Ti GPUs

## Future Work
- Expansion to larger and multi-center datasets.

- Exploration of additional radiomic features (e.g., GLSZM, NGTDM).

- Adoption of multi-instance learning frameworks to capture intra-tumoral heterogeneity.

## Link to Report
https://docs.google.com/document/d/1Ch0H9IQ49cAxud1ZAVkgRbZLbzshCWQEQq2X3ezL5TQ/edit?tab=t.0
