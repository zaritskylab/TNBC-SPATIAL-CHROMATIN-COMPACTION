import sys

from flim_analysis.gnn_clssification.create_pytorch_geo_data.process_data_pytorch_geo import *
import config.const as const
import config.params as params

if __name__ == '__main__':

 
    patch_size = 1500
    overlap = 0.75
    max_dist = 30
    feature_type = 'lifetime'


    print(f"start create pytorch geo data for feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")

    create_gnn_data_patches_structure(feature_type, max_dist, patch_size, overlap)

    print(f"finish create pytorch geo data for feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")
