import sys
import os
from flim_analysis.gnn_clssification.build_graphs.build_graph import *
from utils.data_func import *
import config.const as const
import config.params as params
import argparse
primary_seed = 42
np.random.seed(primary_seed)
random.seed(primary_seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build graphs for GNN training.')

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
        print(f"start build graphs for patch size {patch_size}, overlap {overlap}")
        df_file_path = os.path.join(const.patch_dir, f'size_{patch_size}_overlap_{overlap}', f"FLIM_features_patches_size_{patch_size}_overlap_{overlap}_after_filter.csv")
        patch_tissue_features_df = pd.read_csv(df_file_path, dtype = {'leap_ID': str})
        
        print(f"start max distance {max_dist}")
        graph_file_dir = os.path.join(const.gnn_dir, feature_type, 'patch_tissue', f"max_distance_{max_dist}", 'graphs_data')
        graphs = create_patches_graphs_parallel(patch_tissue_features_df, ['leap_ID', 'patch_ID'], feature_type, max_dist, save_dir=graph_file_dir, local_params=f'size_{patch_size}_overlap_{overlap}')
        print(f"finish max distance {max_dist}")
        
        print(f"finish build graphs for patch size {patch_size}, overlap {overlap}, max distance {max_dist}")


    elif args.graph_type == 'structure_gnn':
        print(f"start build graphs for patch size {patch_size}, overlap {overlap}")
        df_file_path = os.path.join(const.patch_dir, f'size_{patch_size}_overlap_{overlap}', f"FLIM_features_patches_size_{patch_size}_overlap_{overlap}_after_filter.csv")
        patch_tissue_features_df = pd.read_csv(df_file_path, dtype = {'leap_ID': str})

        patch_tissue_features_df['lifetime_mean'] = 0

        print(f"start max distance {max_dist}")
        graph_file_dir = os.path.join(const.gnn_dir, feature_type, 'patch_tissue', f"max_distance_{max_dist}", 'graphs_data', 'structure_only')

        graphs = create_patches_graphs_parallel(patch_tissue_features_df, ['leap_ID', 'patch_ID'], feature_type, max_dist, save_dir=graph_file_dir, local_params=f'size_{patch_size}_overlap_{overlap}')
        print(f"finish max distance {max_dist}")
        
        print(f"finish build graphs for patch size {patch_size}, overlap {overlap}, max distance {max_dist}")


    elif args.graph_type == 'shuffling_gnn':
        seeds_num = args.n_seeds
        seeds = [random.randint(1, 10000) for _ in range(seeds_num)]
        print(f"start shuffle + build graphs for patch size {patch_size}, overlap {overlap}")
        df_file_path = os.path.join(const.patch_dir, f'size_{patch_size}_overlap_{overlap}', f"FLIM_features_patches_size_{patch_size}_overlap_{overlap}_after_filter.csv")
        patch_tissue_features_df = pd.read_csv(df_file_path, dtype = {'leap_ID': str})

        for seed in seeds:
            print(f"start shuffeling lifetime value seed {seed}")
            rng = np.random.default_rng(seed)
            # Shuffle the column using the seed
            def shuffle_group(x):
                return rng.permutation(x.values)

            shuffling_features_df = patch_tissue_features_df.copy()
            shuffling_features_df['lifetime_mean'] = (
                shuffling_features_df
                .groupby(['leap_ID', 'patch_ID'])['lifetime_mean']
                .transform(shuffle_group)
            )

            shuffling_df_dir_path = os.path.join(const.patch_dir, f'size_{patch_size}_overlap_{overlap}', 'shuffling_lifetime', f'{seed}')
            os.makedirs(shuffling_df_dir_path, exist_ok=True)
            filename = f"FLIM_features_patches_size_{patch_size}_overlap_{overlap}_after_shuffling.parquet"
            file_path = os.path.join(shuffling_df_dir_path, filename)
            shuffling_features_df.to_parquet(file_path, index=False)
            print(f"Saved shuffled DataFrame to: {file_path}")
            print(f"finish shuffeling lifetime value seed {seed}")

            print(f"start build graphs for patch size {patch_size}, overlap {overlap}")
            print(f"start max distance {max_dist}")
            graph_file_dir = os.path.join(const.gnn_dir, feature_type, 'patch_tissue', f"max_distance_{max_dist}", 'graphs_data', 'shuffling_lifetime', f'{seed}')
            graphs = create_patches_graphs_parallel(shuffling_features_df, ['leap_ID', 'patch_ID'], feature_type, max_dist, save_dir=graph_file_dir, local_params=f'size_{patch_size}_overlap_{overlap}')
            print(f"finish max distance {max_dist}")
            print(f"finish build graphs for patch size {patch_size}, overlap {overlap}")

            print(f"finish shuffle + build graphs for patch size {patch_size}, overlap {overlap}, max distance {max_dist} seed {seed}")

