# TNBC-SPATIAL-CHROMATIN-COMPACTION

This repository contains tools and code for analyzing Fluorescence Lifetime Imaging Microscopy (FLIM) data in the context of triple-negative breast cancer (TNBC). It provides pipelines for segmentation, feature extraction, and analysis of FLIM images from tissue samples.



## Installation and Setup

### 1. Clone the Repository

You must clone the repository and `cd` into it before installing. The `pyproject.toml` file lives at the root of the repo and is required for correct installation.

```bash
git clone https://github.com/zaritskylab/TNBC-SPATIAL-CHROMATIN-COMPACTION
cd TNBC-SPATIAL-CHROMATIN-COMPACTION
```

### 2. Enviroment installation 

```bash
conda env create -f environment.yml
conda activate tnbc_flim
pip install -e 
```


## Configuration Check

Before running any analysis, make sure the base directory for your analysis is correctly set.

```python
# config/const.py
BASE_DIR = "/your/full/path/to/analysis"
```

## Repository overview

```bash
TNBC-SPATIAL-CHROMATIN-COMPACTION/
├── config/
│   ├── const.py
│   └── params.py
│
├── flim_analysis/
│   ├── distribution_classification/
│   ├── feature_extraction/
│   ├── gnn_classification/
│   ├── preprocessing/
│   ├── resection analysis/
│   └── spatial_analysis/
│
├── notebooks/
│   ├── analysis_paper_result_reproduce/
│   └── usage_example/
│
├── sbatch/
├── utils/
├── pyproject.toml
└── enviroment.yml
```

### Main Components

### `config/`
Use this folder to manage fixed paths and experiment-level configuration settings.

Contains configuration files:
- `const.py`: path global constants
- `params.py`: experiment parameters


### `flim_analysis/`
Main source code organized by functional domain:

#### `preprocessing/`
Tissue segmentation and preprocessing workflows to prepare input for feature extraction.

#### `feature_extraction/`
Scripts for extracting FLIM and morphological features at both the patch and full-tissue level.

#### `distribution_classification/`
Distribution-based classification using cross-validation and robust model evaluation techniques.

#### `gnn_classification/`
End-to-end GNN pipeline, subdivided into:
- `build_graphs/`: construct graphs from extracted features
- `create_pytorch_geo_data/`: process graphs into PyTorch Geometric format
- `train_model/`: training and evaluation of GNNs

#### `spatial_analysis/`
Contains spatial metrics and related analysis.  
**Data preparation for some notebooks is done in**  
`spatial_analysis/spatial_information.ipynb`

#### `resection analysis/`
Contains Jupyter notebooks related to resection-based spatial analysis.

---

### `notebooks/`
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

Download the dataset from [https://doi.org/10.6019/S-BIAD2418] and extract it to your local machine.

> **Note**  
> After extraction, make sure all required input data is located in the directory specified by `const.DATA_DIR` (defined in config/const.py).
>
> ```python
> DATA_DIR = os.path.join(BASE_DIR, "data")
> ```
>
> The scripts assume that all input data is stored inside this directory.

### Folder layout
```bash
data/
├── raw/
├── segmentations/
├── segmentations_after_qc/
└── cohort_metadata.csv
```

- raw/ – original FLIM inputs (intensity and lifetime images).
- segmentations/ – per-image nucleus segmentation outputs.
- segmentations_after_qc/ – segmentation masks after quality control (bad regions removed/fixed).
- cohort_metadata.csv – per-sample metadata. 


## Direct script run
### Preprocessing and segmentation
```bash
python flim_analysis/preprocessing/processing.py

# NOTE: Running this script will overwrite the existing data segmentation outputs in SEG_DIR and SEG_AFTER_QC_DIR.
python flim_analysis/preprocessing/segmentation.py
```
### Core analysis
#### Full tissue analysis
```bash
# Step 1: Feature extraction
python flim_analysis/feature_extraction/extract_features.py core
```

```bash
# Step 2: Create lifetime distribution with 18 bins and median features data 
python flim_analysis/feature_extraction/create_distribution_and_median.py core --max-val 13 --bin-range 0.73 
```

```bash
# Step 3: Lifetime distribution treatment classification tissue wise
python -u -m flim_analysis/distribution_classification/treatment_classification_tissue_wise --dist_csv_name features_lifetime_distribution_data_max_val_13_bins_amount_18_bin_range_0.73.csv --n_seeds 1 --n_permutations 1
```

#### Patch analysis
```bash
# Step 1: Feature extraction (NOTE: Run this only AFTER core feature extraction is complete).
python flim_analysis/feature_extraction/extract_features.py patch --patch-size 1500 --overlap 0.75
```

```bash
# Step 2: Create lifetime distribution with 18 bins
python flim_analysis/feature_extraction/create_distribution_and_median.py patch --patch-size 1500 --overlap 0.75 --max-val 13 --bin-range 0.73      
```

```bash
# Step 3: Lifetime distribution treatment classification patch wise 
python -u -m flim_analysis/distribution_classification/treatment_classification_patch_wise --dist_csv_name features_lifetime_distribution_data_patches_size_1500_overlap_0.75_max_val_13_bins_amount_18_bin_range_0.73.csv --patch_size 1500 --n_seeds 1 --n_permutations 1
```

#### GNN Classification
```bash
# Step 1: Build graphs for GNN training (NOTE: Run this only AFTER patch feature extraction is complete)
python -u flim_analysis/gnn_classification/build_graphs/build_graph_main.py gnn --patch-size 1500 --overlap 0.75 --feature_type 'lifetime' --max_dist 30
```

```bash
# Step 2: Create PyTorch Geometric data objects for GNN training
python -u flim_analysis/gnn_classification/create_pytorch_geo_data/process_data_pytorch_geo_main.py gnn --patch-size 1500 --overlap 0.75 --feature_type 'lifetime' --max_dist 30
```

```bash
# Step 3: GNN training
python -u flim_analysis/gnn_classification/train_model/train_gnn_model_main.py gnn --patch-size 1500 --overlap 0.75 --feature_type 'lifetime' --max_dist 30 --k-fold 5 --model-id 1 --n_seeds 1
```

### Resection analysis
```bash
# Feature extraction
python flim_analysis/feature_extraction/extract_features.py resection
```
 
```bash
# Create median features data frame
python flim_analysis/feature_extraction/create_distribution_and_median.py resection
```
  
