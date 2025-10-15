import sys

from flim_analysis.gnn_clssification.train_model.train_gnn_model import *
from utils.data_func import *
import config.params as params


if __name__ == '__main__':
    task_id = int(sys.argv[1])
    feature_type = sys.argv[2]

    
    patch_size = 1500
    overlap = 0.75
    max_dist=30
    k_fold=5

    # model_params_dict = params.model_params[task_id]
    model_params_dict = params.model_params[1]


    run_patch_gnn_pipeline_per_one_model_parameters(
        feature_type=feature_type,
        patch_size=patch_size,
        overlap=overlap,
        max_dist=max_dist,
        k_out=k_fold,
        model_params=model_params_dict,
        primary_seed=params.primary_seed,
        aggregation=True,
        tissue_resolution='patch_tissue'
    )
