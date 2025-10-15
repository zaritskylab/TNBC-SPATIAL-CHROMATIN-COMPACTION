# TNBC-SPATIAL-CHROMATIN-COMPACTION

This repository contains tools and code for analyzing Fluorescence Lifetime Imaging Microscopy (FLIM) data in the context of triple-negative breast cancer (TNBC). It provides pipelines for segmentation, feature extraction, and analysis of FLIM images from tissue samples.

---

## Installation and Setup

### 1. Clone the Repository

You must clone the repository and `cd` into it before installing. The `pyproject.toml` file lives at the root of the repo and is required for correct installation.

```bash
git clone https://github.com/zaritskylab/flim-tnbc
cd flim-tnbc
```

### 2. Choose Your Installation Strategy

>  he installation is split into base and optional components to allow for flexibility and to avoid dependency conflicts - especially when working with GPU-based libraries like stardist and torch.

There are two options, depending on whether you're using GPU or CPU:

* Option 1: If you plan to run segmentation on GPU, you must use separate environments for stardist and torch, as their GPU versions may conflict due to incompatible CUDA requirements.

* Option 2: If you are running segmentation on CPU only, you can safely install everything in a single environment using [all]

#### Option 1: 
 > Use the appropriate environment depending on your task:

* Activate **flim_stardist** when running segmentation.

* Activate **flim_torch** when running graph-based analysis.

When using Option 1, any tasks other than segmentation or graph-based (GNN) analysis can be run in either environment.

##### Environment A: For Segmentation with StarDist
```bash
conda create -n flim_stardist python=3.10.13 -y
conda activate flim_stardist

pip install -e .[stardist]
```

##### Environment B: For Graph-Based Analysis with Torch 
```bash
conda create -n flim_torch python=3.10.13 -y
conda activate flim_torch

pip install -e .[torch]
```

#### Option 2: 
##### Install everything in one environment (only for CPU)
```bash
conda create -n flim_all python=3.10.13 -y
conda activate flim_all

pip install -e .[all]
```

---

## Configuration Check

Before running any analysis, make sure the base directory for your analysis is correctly set.

```python
# config/const.py
base_dir = "/your/full/path/to/analysis"
```
---

## Project Structure

```bash
TNBC_FLIM/
├── config/
├── flim_analysis/
├── notebooks/
├── sbatch/
├── utils/
├── pyproject.toml
└── README.md
```

### Main Components

#### `config/`
Use this folder to manage fixed paths and experiment-level configuration settings.

Contains configuration files:
- `const.py`: path global constants
- `params.py`: experiment parameters

---

#### `flim_analysis/`
Main source code organized by functional domain:

##### `feature_extraction/`
Scripts for extracting FLIM and morphological features at both the patch and full-tissue level.

##### `preprocessing/`
Tissue segmentation and preprocessing workflows to prepare input for feature extraction.

##### `spatial_analysis/`
Contains spatial metrics and related analysis.  
**Data preparation for some notebooks is done in**  
`spatial_analysis/spatial_information.ipynb`

##### `gnn_clssification/`
End-to-end GNN pipeline, subdivided into:
- `build_graphs/`: construct graphs from extracted features
- `create_pytorch_geo_data/`: process graphs into PyTorch Geometric format
- `train_model/`: training and evaluation of GNNs

##### `resection analysis/`
Contains Jupyter notebooks related to resection-based spatial analysis.

---

#### `notebooks/`
Top-level directory for exploratory and paper-figure notebooks.

#### `analysis_paper_result_reproduce/`
Organized by figure number — contains both **main** and **supplementary** figure notebooks:
- `Figure_1.ipynb` – `Figure_4.ipynb`: Main paper figures
- `Supplementary/`: Supplementary figure generation and visualization notebooks
- `_preparation.ipynb` vs `_visualize.ipynb` convention used for clean separation between data generation and plotting

#### `usage_example/`
Contains runnable examples demonstrating how to use the main components of the pipeline.

---
## Data

This analysis is based on a retrospective cohort obtained from triple-negative breast cancer (TNBC) patients prior to neoadjuvant chemotherapy (NACT).

### Download

Download the dataset from [link] and extract it to your local machine.
After extraction, set data_dir in config/const.py to the folder you downloaded.

```python
# config/const.py
data_dir = "/your/full/path/to/data"
```

### Folder layout
```bash
data/
├── raw/
├── segmentations/
├── segmentations_after_qc/
└── metadata.csv
```

- raw/ – original FLIM inputs (intensity and lifetime images).
- segmentations/ – per-image nucleus segmentation outputs.
- segmentations_after_qc/ – segmentation masks after quality control (bad regions removed/fixed).
- metadata.csv – per-sample metadata used across notebooks and pipelines.

### metadata.csv (columns)

**TBD**

### Cohort Overview
#### Core Samples
- **Pre-treatment biopsies**, collected before the initiation of NACT.
- Each sample represents a **small region of the tumor**, typically taken with a core needle.
- Used to assess **chromatin compaction** features prior to therapy.
- Cohort includes:
  - **30 cores from responders**
  - **23 cores from non-responders**

#### Resection Samples
- **Post-treatment surgical specimens**, acquired only for non-responder patients.
- These samples were taken at the time of **surgery** following failed NACT.
- Resections are **substantially larger** than core biopsies and provide more extensive spatial context.
- Available for **17 of the 23 non-responder** patients.

#### Clinical Classification
Patients were classified into **responder** and **non-responder** groups based on their **Residual Cancer Burden (RCB) score**, a metric reflecting the amount of tumor remaining after treatment.

---

#### Note:
The notebooks performs **segmentation and feature extraction** on the FLIM images corresponding to:
- **Core biopsies** (pre-treatment)
- **Resection specimens** (post-treatment for non-responders)

Resection samples are substantially larger than core biopsies, and therefore require significantly more time and memory for segmentation and feature extraction.