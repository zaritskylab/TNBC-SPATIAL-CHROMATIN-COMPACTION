import sys

from flim_analysis.gnn_clssification.create_pytorch_geo_data.process_data_pytorch_geo import *
import config.const as const
import config.params as params

if __name__ == '__main__':

    task_id = int(sys.argv[1])
    feature_type = sys.argv[2]
 
    patch_size = 1500
    overlap = 0.75
    max_dist = 30
    seed = params.shuffling_seed_list[task_id]


    print(f"start create pytorch geo data for seed-{seed}, feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")

    create_gnn_data_patches_shuffling(feature_type, max_dist, patch_size, overlap, seed)

    print(f"finish create pytorch geo data for seed-{seed}, for feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")
