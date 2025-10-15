import os

data_dir = r'/sise/assafzar-group/assafzar/leor/delta_tissue_maddy/data/tnbc_organised/4_FLIM'
raw_data_dir = os.path.join(data_dir, 'raw')
seg_dir = os.path.join(data_dir, 'segmentations')
seg_after_qc_dir = os.path.join(data_dir, 'segmentations_after_qc')


base_dir =  r'path/to/your/full/analysis'
# seg_dir = os.path.join(base_dir, 'segmentations')
# seg_after_qc_dir = os.path.join(base_dir, 'segmentations_after_qc')
single_cell_feature_dir = os.path.join(base_dir, 'leap_single_cell_features')
fluorescent_dir = os.path.join(base_dir, 'fluorescent_channel')
flim_dir = os.path.join(base_dir, 'flim_channel')
patch_dir = os.path.join(base_dir, 'patches_tissue')
full_tissue_dir = os.path.join(base_dir, 'full_tissue')
gnn_dir = os.path.join(base_dir, 'gnn')
spatial_dir = os.path.join(base_dir, 'spatial')
rcb_file = os.path.join(base_dir, 'tnbc_dataset.xlsx')
flim_model_probability_dir = os.path.join(base_dir, 'flim_model_probability')
single_nuclei_lifetime_dir = os.path.join(base_dir, 'single_nuclei_lifetime')




distribution_results_base_dir = r'path/to/your/lifetime/distribution/results'

distribution_results_full_tissue_dir = os.path.join(distribution_results_base_dir, '18_bins_new/xgboost', 'tissue_wise')
distribution_results_patch_dir = os.path.join(distribution_results_base_dir, '18_bins_new/xgboost', 'patch_wise')

