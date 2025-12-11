import os

BASE_DIR = "/your/full/path/to/analysis" # Base directory for data, all analysis outputs and intermediate files.

DATA_DIR = os.path.join(BASE_DIR, 'data')   # Root directory for all input data (raw, segmentations, image channels, metadata file).
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')   # Directory containing the original raw input data as downloaded.
SEG_DIR = os.path.join(DATA_DIR, 'segmentations')   # Directory containing nuclei segmentations (before quality control).
SEG_AFTER_QC_DIR = os.path.join(DATA_DIR, 'segmentations_after_qc')   # Directory containing segmentations after applying quality-control filters.
FLUORESCENT_DIR = os.path.join(DATA_DIR, 'fluorescent_channel')   # Directory with the fluorescent intestity channel images.
FLIM_DIR = os.path.join(DATA_DIR, 'flim_channel')   # Directory with the fluorescent lifetime channel images.
RCB_FILE = os.path.join(DATA_DIR, 'cohort_metadata.csv') # Path to the clinical metadata.


PATCH_DIR = os.path.join(BASE_DIR, 'patches_tissue')   # Directory used in patch-wise analyses.     
FULL_TISSUE_DIR = os.path.join(BASE_DIR, 'full_tissue')   # Directory used in tissue-wise analyses.
GNN_DIR = os.path.join(BASE_DIR, 'gnn')   # Directory for graph neural network (GNN) inputs, models, and results.
FLIM_MODEL_PROBABILITY_DIR = os.path.join(BASE_DIR, 'flim_model_probability')   # Directory for FLIM model probability maps.

DISTRIBUTION_RESULTS_BASE_DIR = os.path.join(BASE_DIR, 'distribution_results')   # Base directory for distribution analysis results.

DISTRIBUTION_RESULTS_FULL_TISSUE_DIR = os.path.join(DISTRIBUTION_RESULTS_BASE_DIR, 'tissue_wise')   # Directory for tissue-wise distribution results.
DISTRIBUTION_RESULTS_PATCH_DIR = os.path.join(DISTRIBUTION_RESULTS_BASE_DIR, 'patch_wise')   # Directory for patch-wise distribution results.

DATA_PREPARATION_DIR = os.path.join(BASE_DIR, 'figure_results', 'data_preparation')   # Directory for intermediate data used in figure generation.
os.makedirs(DATA_PREPARATION_DIR, exist_ok=True)

FIGURE_SUPPLEMENTARY_DIR = os.path.join(BASE_DIR, 'figure_results', 'Supplementary')   # Directory for supplementary figure outputs.
os.makedirs(FIGURE_SUPPLEMENTARY_DIR, exist_ok=True)


PRIMARY_SEED = 42   # Primary random seed used across the analysis to ensure reproducibility.
