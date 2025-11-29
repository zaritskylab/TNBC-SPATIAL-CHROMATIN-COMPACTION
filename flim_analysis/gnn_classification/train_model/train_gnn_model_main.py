import sys

from flim_analysis.gnn_classification.train_model.train_gnn_model import *
# from utils.data_func import *
import config.params as params
import argparse
import random
import config.const as const

np.random.seed(const.PRIMARY_SEED)
random.seed(const.PRIMARY_SEED)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GNN training.')

    
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

    parser.add_argument('--k-fold', type=int, default=5,
    help="Number of folds to use for k-fold cross-validation. Default: 5."
    )

    parser.add_argument('--model-id', type=int, default=1,
    help="ID of the model hyperparameter set to use from params.model_params. Default: 1."
    )

    parser.add_argument(
      '--n_seeds', type=int, default=20, help="Number of seeds to run"
    )


    args = parser.parse_args()

    patch_size = args.patch_size
    overlap    = args.overlap
    feature_type = args.feature_type
    max_dist = args.max_dist
    k_fold = args.k_fold
    model_id = args.model_id

    model_params_dict = params.model_params[model_id]

    seeds_num = args.n_seeds

    if args.graph_type == 'gnn':

        run_patch_gnn_pipeline_per_one_model_parameters(
            feature_type=feature_type,
            patch_size=patch_size,
            overlap=overlap,
            max_dist=max_dist,
            k_out=k_fold,
            model_params=model_params_dict,
            seeds_amount=seeds_num,
            aggregation=True,
            tissue_resolution='patch_tissue'
        )


    elif args.graph_type == 'structure_gnn':
        seeds = [random.randint(1, 10000) for _ in range(seeds_num)]
        for seed in seeds:
            run_patch_gnn_pipeline_per_one_model_parameters_structure(
                feature_type=feature_type,
                patch_size=patch_size,
                overlap=overlap,
                max_dist=max_dist,
                k_out=k_fold,
                model_params=model_params_dict,
                seed_val=seed,
                aggregation=True,
                tissue_resolution='patch_tissue'
            )

    elif args.graph_type == 'shuffling_gnn':
        seeds = [random.randint(1, 10000) for _ in range(seeds_num)]
        for seed in seeds:
            run_patch_gnn_pipeline_per_one_model_parameters_shuffling(
                feature_type=feature_type,
                patch_size=patch_size,
                overlap=overlap,
                max_dist=max_dist,
                k_out=k_fold,
                model_params=model_params_dict,
                seed_shuffle_val=seed,
                aggregation=True,
                tissue_resolution='patch_tissue'
            )
