import sys

from flim_analysis.gnn_classification.create_pytorch_geo_data.process_data_pytorch_geo import *
from utils.data_func import *
import config.const as const
import config.params as params
import argparse
import numpy as np
import random


np.random.seed(const.PRIMARY_SEED)
random.seed(const.PRIMARY_SEED)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create PyTorch Geometric data objects for GNN training.')

    # Positional with default (optional via nargs='?')
    parser.add_argument('graph_type', choices=['gnn', 'structure_gnn', 'shuffling_gnn'], nargs='?', default='gnn',
        help=(
        "Type of graph to build. "
        "'gnn' retains original feature values; "
        "'structure_gnn' removes feature values and keeps only structure; "
        "'shuffling_gnn' randomly shuffles feature values across nodes. "
        "Default: 'gnn'."
        )
    )

    parser.add_argument('--patch-size', type=int, default=1500,
        help="Size (in pixels) of the patch used in patch-level processing. Default: 1500."
    )

    parser.add_argument('--overlap', type=float, default=0.75,
        help="Fractional overlap between adjacent patches (0.0 to 1.0). Default: 0.75."
    )

    parser.add_argument('--feature_type',type=str, default='lifetime',
        help="Feature type used for graph construction. Default: 'lifetime'."
    )

    parser.add_argument('--max_dist', type=int, default=30,
        help="Maximum spatial distance (in pixels) for connecting edges in the graph. Default: 30."
    )
    
    parser.add_argument(
      '--n_seeds', type=int, default=20, help="Number of seeds to run"
    )


    args = parser.parse_args()

    patch_size = args.patch_size
    overlap    = args.overlap
    feature_type = args.feature_type
    max_dist = args.max_dist



    if args.graph_type == 'gnn':
        print(f"start create pytorch geo data for feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")

        create_gnn_data_patches(feature_type, max_dist, patch_size, overlap)

        print(f"finish create pytorch geo data for feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")


    elif args.graph_type == 'structure_gnn':
        print(f"start create pytorch geo data for feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")

        create_gnn_data_patches_structure(feature_type, max_dist, patch_size, overlap)

        print(f"finish create pytorch geo data for feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")


    elif args.graph_type == 'shuffling_gnn':
        seeds_num = args.n_seeds
        seeds = [random.randint(1, 10000) for _ in range(seeds_num)]

        for seed in seeds:
            print(f"start create pytorch geo data for seed-{seed}, feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")

            create_gnn_data_patches_shuffling(feature_type, max_dist, patch_size, overlap, seed)

            print(f"finish create pytorch geo data for seed-{seed}, for feature-{feature_type}, max distance-{max_dist} patch size {patch_size}, overlap {overlap}")
