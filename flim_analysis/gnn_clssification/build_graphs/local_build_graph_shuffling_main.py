import sys
import os
from flim_analysis.gnn_clssification.build_graphs.build_graph import *
from utils.data_func import *
import config.const as const
import config.params as params


if __name__ == '__main__':
    task_id = int(sys.argv[1])
    feature_type = sys.argv[2]
 
    patch_size = 1500
    overlap = 0.75
    max_dist = 30
    seed = params.shuffling_seed_list[task_id]
    
    print(f"start shuffle + build graphs for patch size {patch_size}, overlap {overlap}")
    df_file_path = os.path.join(const.patch_dir, f'size_{patch_size}_overlap_{overlap}', f"FLIM_features_patches_size_{patch_size}_overlap_{overlap}_after_filter.csv")
    patch_tissue_features_df = pd.read_csv(df_file_path, dtype = {'leap_ID': str})

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

