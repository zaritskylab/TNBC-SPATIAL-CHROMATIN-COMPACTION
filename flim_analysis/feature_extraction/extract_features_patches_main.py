import sys
from flim_analysis.feature_extraction.extract_features import *
import config.params as params

if __name__ == '__main__':
    task_id = int(sys.argv[1])
 
    patch_size = params.patches_params[task_id]['size']
    overlap = params.patches_params[task_id]['overlap']
    
    create_all_feature_patches_df(patch_size, overlap)

    max_val = params.lifetime_distribution_max_val
    bin_range = params.lifetime_distribution_bin_range

    df_bins_patches = build_lifetime_distribution_patch(patch_size=patch_size, patch_overlap=overlap, max_range=max_val, bin_range=bin_range)



