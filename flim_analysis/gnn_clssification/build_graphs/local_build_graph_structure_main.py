import sys
import os
from flim_analysis.gnn_clssification.build_graphs.build_graph import *
from utils.data_func import *
import config.const as const
import config.params as params


if __name__ == '__main__':


    feature_type = 'lifetime'
    patch_size = 1500
    overlap = 0.75
    max_dist = 30
    
    print(f"start build graphs for patch size {patch_size}, overlap {overlap}")
    df_file_path = os.path.join(const.patch_dir, f'size_{patch_size}_overlap_{overlap}', f"FLIM_features_patches_size_{patch_size}_overlap_{overlap}_after_filter.csv")
    patch_tissue_features_df = pd.read_csv(df_file_path, dtype = {'leap_ID': str})

    patch_tissue_features_df['lifetime_mean'] = 0


    print(f"start max distance {max_dist}")
    graph_file_dir = os.path.join(const.gnn_dir, feature_type, 'patch_tissue', f"max_distance_{max_dist}", 'graphs_data', 'structure_only')

    graphs = create_patches_graphs_parallel(patch_tissue_features_df, ['leap_ID', 'patch_ID'], feature_type, max_dist, save_dir=graph_file_dir, local_params=f'size_{patch_size}_overlap_{overlap}')
    print(f"finish max distance {max_dist}")
    
    print(f"finish build graphs for patch size {patch_size}, overlap {overlap}, max distance {max_dist}")


