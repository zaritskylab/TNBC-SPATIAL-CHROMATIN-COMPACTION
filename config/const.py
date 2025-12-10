import os

BASE_DIR = "/your/full/path/to/analysis"

DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
SEG_DIR = os.path.join(DATA_DIR, 'segmentations')
SEG_AFTER_QC_DIR = os.path.join(DATA_DIR, 'segmentations_after_qc')
FLUORESCENT_DIR = os.path.join(DATA_DIR, 'fluorescent_channel')
FLIM_DIR = os.path.join(DATA_DIR, 'flim_channel')
RCB_FILE = os.path.join(DATA_DIR, 'cohort_metadata.csv')


PATCH_DIR = os.path.join(BASE_DIR, 'patches_tissue')
FULL_TISSUE_DIR = os.path.join(BASE_DIR, 'full_tissue')
GNN_DIR = os.path.join(BASE_DIR, 'gnn')
SPATIAL_DIR = os.path.join(BASE_DIR, 'spatial')
FLIM_MODEL_PROBABILITY_DIR = os.path.join(BASE_DIR, 'flim_model_probability')
SINGLE_NUCLEI_LIFETIME_DIR = os.path.join(BASE_DIR, 'single_nuclei_lifetime')


DISTRIBUTION_RESULTS_BASE_DIR = os.path.join(BASE_DIR, 'distribution_results')

DISTRIBUTION_RESULTS_FULL_TISSUE_DIR = os.path.join(DISTRIBUTION_RESULTS_BASE_DIR, 'tissue_wise')
DISTRIBUTION_RESULTS_PATCH_DIR = os.path.join(DISTRIBUTION_RESULTS_BASE_DIR, 'patch_wise')

DATA_PREPARATION_DIR = os.path.join(BASE_DIR, 'figure_results', 'data_preparation')
os.makedirs(DATA_PREPARATION_DIR, exist_ok=True)

FIGURE_SUPPLEMENTARY_DIR = os.path.join(BASE_DIR, 'figure_results', 'Supplementary')
os.makedirs(FIGURE_SUPPLEMENTARY_DIR, exist_ok=True)


PRIMARY_SEED = 42
