# Spatially distinct chromatin compaction states predict neoadjuvant chemotherapy resistance in Triple Negative Breast Cancer

Reut Mealem<sup>1*</sup>, Thomas. A. Phillips<sup>2*</sup>, Leor Ariel Rose<sup>1*</sup>, Stefania Marcotti<sup>2</sup>, Maddy Parsons<sup>2&</sup>, Assaf Zaritsky<sup>1&</sup>

1. Institute for Interdisciplinary Computational Science, Faculty of Computer and Information Science, Ben-Gurion University of the Negev, Beer-Sheva 84105, Israel
2. Randall Centre for Cell and Molecular Biophysics, King’s College London, Guy’s Campus London, SE1 1UL, UK

- __\*__ Equal contribution
- __\&__ Co-corresponding authorship

Organisation and dynamics of chromatin play a key role in regulation of cell state and function. In cancer, chromatin plasticity is known to be important in control of drug resistance, but the relationship between chromatin compaction and chemotherapy response within complex tissue settings remains unclear. Here, we measured single nuclei chromatin compaction using fluorescence lifetime imaging microscopy (FLIM) in situ in whole biopsies from 53 pre treatment and 14 post surgery tissue samples of human triple-negative breast cancer (TNBC) patients to determine whether single nuclei spatial chromatin compaction state can predict resistance to neoadjuvant chemotherapy (NACT). Bulk chromatin compaction across 53 pre-treatment core biopsies did not predict patient outcome. However, machine learning analysis revealed that a subset of patients exhibited distinct distributions of single nuclei with more open chromatin states, which was predictive of NACT-resistance.

Graph neural network analysis established that the spatial arrangement of chromatin compaction contributed to prediction of NACT resistance and that chromatin compaction signatures were preserved in tissue state transitions from pre- to 14 post-NACT samples. Our findings shed new light on spatial control of chromatin structure and relationship to therapeutic resistance and establish a foundation for further molecular analysis of chromatin states in complex biological tissues.

To read the full research paper go to the following link **[Spatially distinct chromatin compaction states predict neoadjuvant chemotherapy resistance in Triple Negative Breast Cancer](https://doi.org/10.64898/2025.12.04.692131)**

## Dataset Setup

This repo is based on a retrospective cohort obtained from triple-negative breast cancer (TNBC) patients prior to neoadjuvant chemotherapy (NACT). **You must download it and set the paths in the code before running anything**.

### Data overview

```bash
FLIM/
├── metadata/ # Image acquisition metadata
│   ├── LEAP015_slide7_extreme-non-responder_0countthreshold_properties.xml
│   └── ... Other .xml files corresponding to each LEAP ID
├── raw/ # Raw FLIM images
│   ├── LEAP015_slide7_extreme-non-responder_0countthreshold.tif
│   └── ... Other .tif files corresponding to each LEAP ID
├── segmentations/ # Single nuclei segmentation maps (can also be created by the repo using the raw data)
│   ├── LEAP015_segmentation_labels.tif
│   └── ... Other .tif files corresponding to each LEAP ID
├── segmentations_after_qc/ # Single nuclei segmentation maps after quality control (can also be created by the repo using the raw data and segmentation maps)
│   ├── LEAP015_segmentation_labels_qc.tif
│   └── ... Other .tif files corresponding to each LEAP ID
└── cohort_metadata.csv # Clinical metadata
```

### Download

Download the dataset from [BioImage Archive](https://doi.org/10.6019/S-BIAD2418) to your local machine. After downloading, make sure you set the required input data directory specified in [config/const.py](config/const.py) by the `DATA_DIR` variable to where the data is stored on your local machine. Any data related computational outputs (segmentations, segmentations_after_qc) will also be saved in this dir if you decide to run their code.

```python
DATA_DIR = "PATH-TO-THE-DATA"
```

## Repo Installation and Setup

### 1. Clone the Repository

You must clone the repository and change directory into the cloned repository. Run the following commands:

```bash
git clone https://github.com/zaritskylab/TNBC-SPATIAL-CHROMATIN-COMPACTION
cd TNBC-SPATIAL-CHROMATIN-COMPACTION
```

### 2. Environment installation

Make sure the [pyproject.toml](pyproject.toml) file lives at the root of the repo as it is required for correct installation of the Python environment and packages. Run the following commands:

```bash
conda env create -f environment.yml
conda activate tnbc_flim
pip install -e .
```


## Configuration Check

Before running any part of this code, make sure you set the base directory for the analysis specified in [config/const.py](config/const.py) by the `BASE_DIR` variable. This is where all the computational outputs will be saved. 

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
│   ├── resection_analysis/
│   └── spatial_analysis/
│
├── notebooks/
│   ├── analysis_paper_result_reproduce/
│   └── usage_example/
│
├── sbatch/
├── utils/
├── pyproject.toml
└── environment.yml
```

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
Contains spatial metrics and related analysis. Some data preparation is also done in the notebook `spatial_analysis/spatial_information.ipynb`

#### `resection_analysis/`
Contains Jupyter notebooks related to resection-based spatial analysis.

### `notebooks/`
Top-level directory for exploratory and paper-figure creation notebooks.

#### `analysis_paper_result_reproduce/`
Organized by figure number — contains both **main** and **supplementary** figure notebooks:
- `Figure_1.ipynb` – `Figure_4.ipynb`: Main paper figures
- `Supplementary/`: Supplementary figure generation and visualization notebooks
- `_preparation.ipynb` vs `_visualize.ipynb` convention used for clean separation between data generation and plotting

#### `usage_example/`
Contains runnable examples demonstrating how to use the main components of the pipeline.

## Direct script run
**You should run the scripts by the order given here as some depend on others (unless stated that it is not required).**

### Preprocessing
```bash
# Processing of the raw images (not contained in the data directory and should be run)
python flim_analysis/preprocessing/processing.py
```
### Segmentation
```bash
# NOTE: This step is not required as data folder already contains segmentations. Running this script will overwrite the existing data segmentation outputs in SEG_DIR and SEG_AFTER_QC_DIR.
python flim_analysis/preprocessing/segmentation.py
```
### Full tissue analysis
```bash
# Full tissue feature extraction
python flim_analysis/feature_extraction/extract_features.py core
```

```bash
# Lifetime distribution creation with default 18 bins and median features lifetime extraction
python flim_analysis/feature_extraction/create_distribution_and_median.py core --max-val 13 --bin-range 0.73 
```

```bash
# Tissue-wise lifetime distribution treatment classification 
python flim_analysis/distribution_classification/treatment_classification_tissue_wise --dist_csv_name features_lifetime_distribution_data_max_val_13_bins_amount_18_bin_range_0.73.csv --n_seeds 100 --n_permutations 1000
```

### Patch analysis
```bash
# Patch feature extraction
python flim_analysis/feature_extraction/extract_features.py patch --patch-size 1500 --overlap 0.75
```

```bash
# Patch lifetime distribution creation with default 18 bins
python flim_analysis/feature_extraction/create_distribution_and_median.py patch --patch-size 1500 --overlap 0.75 --max-val 13 --bin-range 0.73      
```

```bash
# Patch-wise lifetime distribution treatment classification 
python flim_analysis/distribution_classification/treatment_classification_patch_wise --dist_csv_name features_lifetime_distribution_data_patches_size_1500_overlap_0.75_max_val_13_bins_amount_18_bin_range_0.73.csv --patch_size 1500 --n_seeds 100 --n_permutations 1000
```

### GNN Classification
```bash
# Graphs building for GNN training
python flim_analysis/gnn_classification/build_graphs/build_graph_main.py gnn --patch-size 1500 --overlap 0.75 --feature_type 'lifetime' --max_dist 30
```

```bash
# Conversion of graphs to PyTorch Geometric data objects for GNN training
python flim_analysis/gnn_classification/create_pytorch_geo_data/process_data_pytorch_geo_main.py gnn --patch-size 1500 --overlap 0.75 --feature_type 'lifetime' --max_dist 30
```

```bash
# GNN training
python flim_analysis/gnn_classification/train_model/train_gnn_model_main.py gnn --patch-size 1500 --overlap 0.75 --feature_type 'lifetime' --max_dist 30 --k-fold 5 --model-id 1 --n_seeds 20
```

### Resection analysis
```bash
# Resection feature extraction
python flim_analysis/feature_extraction/extract_features.py resection
```
 
```bash
# Create median features data frame
python flim_analysis/feature_extraction/create_distribution_and_median.py resection
```

## Notebook examples

The `notebooks/usage_example` folder contains three Jupyter notebooks that illustrate three main parts of the analysis workflow:

- `run_example_preprocess_segmentation.ipynb`  
  Demonstrates data preprocessing, segmentation and segmentation quality control.

- `run_example_feature_extraction.ipynb`  
  Shows how to run feature extraction on preprocessed and segmented data.

- `run_example_gnn_build_train.ipynb`  
  Builds graphs and trains the GNN model using extracted features.

Each notebook may depend on earlier processing, and any such dependencies are noted at the beginning.

## License

This repository is released under the Creative Commons Attribution–NonCommercial 4.0 International License. Commercial use is not permitted, and any reuse or modification requires proper attribution to the original authors.

## Citation

If you use this code, please cite:

> Mealem, R. et al. Spatially distinct chromatin compaction states predict neoadjuvant chemotherapy resistance in Triple Negative Breast Cancer. 2025.12.04.692131 Preprint at https://doi.org/10.64898/2025.12.04.692131 (2025). 


## Contact
Please contact <reutme@post.bgu.ac.il> or <assafzar@gmail.com> for bugs or questions regarding this repo.
