# Adaptive Local Neighborhood-based Neural Networks for MR Image Reconstruction (LONDN-MRI)

[![IEEE Xplore](https://img.shields.io/badge/IEEE_Xplore-10.1109/TCI.2024.3394770-blue.svg)](https://ieeexplore.ieee.org/abstract/document/10510040)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Official Code Implementation** for the paper "Adaptive Local Neighborhood-based Neural Networks for MR Image Reconstruction from Undersampled Data" (IEEE TCI 2024).

## 📖 Overview

This repository contains the code for testing and reproducing results for the **LONDN-MRI** project. 

The method focuses on **Adaptive Local Neighborhood-based Neural Networks** for MRI reconstruction. The codebase supports varying experimental setups including single-coil and multi-coil studies, leveraging both global and local modeling strategies.

This implementation is based on and extends the **MODL** and **BLIPS** (Blind Primed Supervised Learning) frameworks.

### Key Components
The code is organized into three main study components:
1.  **Single Coil Global MODL**
2.  **Multi-Coil Global MODL**
3.  **Multi-Coil LONDN**

## 📂 Directory Overview

The core implementation is located under the `multi_coil_LONDN/` directory.

```text
📦 LONDN_MRI
 ┣ 📂 data                           # Data directory
 ┣ 📂 models                         # Neural network architectures
 ┃ ┣ 📜 networks.py                  # Common network utilities
 ┃ ┗ 📜 Unet_model_fast_mri.py       # Standard UNet for FastMRI
 ┗ 📂 multi_coil_LONDN               # Main Local LONDN implementation
   ┣ 📂 checkpoints                  # Saved model checkpoints (e.g., exp_0101)
   ┣ 📂 generated_dataset            # Generated image-space datasets and masks
   ┃ ┣ 📂 4acceleration_mask_random3 # Random masks for training (4x accel)
   ┃ ┣ 📂 4acceleration_mask_test2   # Fixed masks for testing (4x accel)
   ┃ ┣ 📂 four_fold_image_shape      # Processed 2-channel complex images
   ┃ ┗ 📂 test_four_fold             # Test set specific data
   ┣ 📜 prepare_data_from_kspace.py  # Script to generate image-space dataset and masks from k-space
   ┣ 📜 local_network_dataset.py     # Data loader for noisy local neighborhood case
   ┗ 📜 train_local_unet.py          # Training/Testing script using UNet for local reconstruction
```

## 🚀 Getting Started

### 1. Prerequisites
* **Python 3.8+**
* **PyTorch > 1.7.0**
* **[BART](https://mrirecon.github.io/bart/)**: Used for generating the initial dataset.

### 2. Environment Setup
We provide an `environment.yml` file to easily configure the Conda environment with all dependencies (including PyTorch, Visdom, Dominate, etc.).

```bash
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate londn_mri
```

### 3. Data Preparation
To reproduce the results, please download the specific k-space datasets used in our experiments.

1.  **Dataset**:  
    - [**fastMRI Dataset website**](https://fastmri.med.nyu.edu/) (Download will take some time)
    - [**Stanford 2D FSE website**](http://mridata.org/list?project=Stanford%202D%20FSE) (or download our copy via [**Google Drive**](https://drive.google.com/drive/folders/1CEI3SH2Amw1wRlQJygeWj1r1TI4jAOjm?usp=sharing))

3.  **Setup**:  
    We **recommend** downloading the **fastMRI dataset** first, as it is the primary dataset used to generate the results in `our_notebook.ipynb`. 
    
    **Download Instructions:**
    * **For fastMRI**: Please visit the official website to obtain the license/agreement and then download the data.
    * **For Stanford 2D FSE**: The **full dataset** is available on the official website. We also provide a **partial dataset** (subset) via [**Google Drive**](https://drive.google.com/drive/folders/1CEI3SH2Amw1wRlQJygeWj1r1TI4jAOjm?usp=sharing) for quick testing.
    
    ⚠️⚠️Once downloaded, unzip the files and place them into the project directory (e.g., inside a folder named `data` or as specified in the notebook).
    
    **Step 1: Configure Path**
    Unzip the downloaded data to your local storage (e.g., `/mnt/DataA/NEW_KSPACE`).
    Open `multi_coil_LONDN/prepare_data_from_kspace.py` and update the `Kspace_data_name` variable:
    
    ```python
    # Inside prepare_data_from_kspace.py.py
    SOURCE_KSPACE_DIR = '/mnt/DataA/NEW_KSPACE'  # <--- Change this to your path
    ```
    
    **Step 2: Generate Dataset**
    Run the script to create the image space data based on the k-space inputs:
    ```bash
    cd multi_coil_LONDN
    python prepare_data_from_kspace.py.py
    ```

## 🏃 Usage

### Local Model Training (UNet)
To train the local model and test reconstruction from undersampled multi-coil k-space measurements:

```bash
python train_local_unet.py
```

### Transfer Learning (DIDN)
To perform transfer learning and reconstruction using the DIDN architecture:

```bash
python transfer_learning_local_network.py
```

*(Note: Ensure you are in the `multi_coil_LONDN` directory when running these scripts.)*

## 📝 Citation

If you find this code useful, please cite our paper:

```bibtex
@ARTICLE{10510040,
  author={Liang, Shijun and Lahiri, Anish and Ravishankar, Saiprasad},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Adaptive Local Neighborhood-Based Neural Networks for MR Image Reconstruction From Undersampled Data}, 
  year={2024},
  volume={10},
  number={},
  pages={1235-1249},
  keywords={Image reconstruction;Magnetic resonance imaging;Training;Deep learning;Adaptation models;Time measurement;Optimization;Compressed sensing;deep learning;machine learning;magnetic resonance imaging;unrolling},
  doi={10.1109/TCI.2024.3394770}}
```

## 📬 Correspondence

For questions regarding the paper or code, please contact:

* **Shijun Liang**: `liangs16@msu.edu`
* **Haijie Yuan**: `hy2786@nyu.edu`
* **Prof. Saiprasad Ravishankar**: `ravisha3@msu.edu`
