import sys

from flim_analysis.gnn_clssification.create_pytorch_geo_data.process_data_pytorch_geo import *
import config.const as const
import config.params as params

if __name__ == '__main__':
    task_id = int(sys.argv[1])
    feature_type = sys.argv[2]

    patch_size = params.patches_params[task_id]['size']
    overlap = params.patches_params[task_id]['overlap']

    for key, max_dist in params.full_graph_max_dist_params.items():

        print(f"start create pytorch geo data for feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")

        create_gnn_data_patches(feature_type, max_dist, patch_size, overlap)

        print(f"finish create pytorch geo data for feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")
    # max_dist = 30
    # print(f"start create pytorch geo data for feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")

    # create_gnn_data_patches(feature_type, max_dist, patch_size, overlap)

    # print(f"finish create pytorch geo data for feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")