# LONDN-MRI: Adaptive Local Neighborhood-based Neural Networks for MR Image Reconstruction

[![IEEE Xplore](https://img.shields.io/badge/IEEE_Xplore-10.1109/TCI.2024.3394770-blue.svg)](https://ieeexplore.ieee.org/abstract/document/10510040)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E1.7.0-ee4c2c.svg)](https://pytorch.org/)

> **Official PyTorch Implementation** for the paper "Adaptive Local Neighborhood-based Neural Networks for MR Image Reconstruction from Undersampled Data" (IEEE TCI 2024).

## 📖 Overview

This repository provides the code for **LONDN-MRI**, a framework for MR image reconstruction using adaptive local neighborhood-based neural networks. The method focuses on learning local dependencies to improve reconstruction quality from undersampled k-space data.

Our implementation builds upon the foundational work of **MODL** and **BLIPS**:
* *Anish Lahiri, et al.* "Blind Primed Supervised (BLIPS) Learning for MR Image Reconstruction." (IEEE TMI 2021). [arXiv:2104.05028](https://arxiv.org/abs/2104.05028).

## ✨ Features

- **Local Learning Strategy**: Implements adaptive local neighborhood-based networks for robust reconstruction.
- **Multi-Coil Support**: capable of handling both single-coil and multi-coil MR measurements.
- **Two-Channel Architecture**: Efficiently processes complex MRI data using a two-channel real/imaginary approach.
- **Transfer Learning**: Includes scripts for transfer learning using DIDN architectures.
- **BART Integration**: Uses [BART](https://mrirecon.github.io/bart/) toolbox for dataset generation and simulation.

## 📂 Repository Structure

The core code is located under the `multi_coil_LONDN/` directory:

```text
📦 LONDN-MRI
 ┣ 📂 multi_coil_LONDN
 ┃ ┣ 📜 make_two_channel_dataset.py          # Script to generate two-channel datasets (Modified from BLIPSrecon)
 ┃ ┣ 📜 global_network_dataset.py            # Dataloader for global multi-coil MR measurements
 ┃ ┣ 📜 local_network_dataset.py             # Dataloader for local noise model case
 ┃ ┣ 📜 local_network_dataset_oracle.py      # Dataloader for local oracle model case
 ┃ ┣ 📜 train_local_unet.py                  # Main script for training and testing Local UNet models
 ┃ ┗ 📜 transfer_learning_local_network.py   # Script for transfer learning using DIDN
 ┗ 📜 requirements.txt                       # Python dependencies
