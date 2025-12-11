import numpy as np
import random
import config.const as const

PRIMARY_SEED = const.PRIMARY_SEED
np.random.seed(PRIMARY_SEED)
random.seed(PRIMARY_SEED)

# Patch extraction settings.
# Keys (1, 2, 3) index different patch configurations (e.g. different scales).
#   - 'size':    patch size in pixels (square patches of size x size).
#   - 'overlap': fractional overlap between neighboring patches (0–1).
patches_params = {1:{'size': 1000, 'overlap': 0.75}, 2:{'size': 1500, 'overlap': 0.75}, 3:{'size': 2000, 'overlap': 0.75}}

# Features to extract from per-nucleus measurements.
# Keys are feature groups, values are lists of specific feature names.
features_params = {'lifetime': ['lifetime_mean']}

lifetime_distribution_params = {10: 1.3, 18: 0.73, 23: 0.585, 34:0.383, 42: 0.31}

# Maximum distance thresholds (in pixels) used when constructing graphs, per configuration key.
full_graph_max_dist_params = {1: 30, 2:40, 3:50}

# Global settings for lifetime distribution histograms.
#   - lifetime_distribution_max_val: upper bound of the lifetime value.
#   - lifetime_distribution_bin_range: bin width.
lifetime_distribution_max_val = 13
lifetime_distribution_bin_range = 0.73

# List of fixed seeds used for GNN control models (graph shuffling of lifetimes and graph structure).
shuffling_seed_list = [435, 410, 489, 521, 1425, 1536, 1680, 1825, 2287, 3583, 3658, 3812, 4013, 4507, 6913, 8280, 8936, 9196, 9675, 9864]

# Number of random seeds to generate for repeated experiments.
seeds_amount = 20
seeds = [random.randint(1, 10000) for _ in range(seeds_amount)]

# Base configuration for the GNN model.
model_custom_params = {
    "model_type": "GAT",           # Option: "GAT" (graph attention network)
    "hidden_layers": [64, 128, 64], # List of hidden layer sizes
    "output_size": 1,              # Usually 1 for binary classification
    "pooling": "max",              # Options: "mean" or "max"
    "heads": 1,                    # Attention heads for GAT
    "weight_decay": 0.0001,        # L2 regularization factor
    "epochs": 50,                  # Number of training epochs
    "class_ratio": 1.0             # Positive class weight for imbalance
}

# Model hyperparameter presets by configuration key (1, 2, 3).
# Each entry expands model_custom_params with training-specific settings:
#   - dropout_rate: dropout probability.
#   - batch_size: training batch size.
#   - test_ratio: fraction of data held out for testing.
#   - lr: learning rate.
model_params = {
                  1:{**model_custom_params, "dropout_rate": 0.1, "batch_size": 32, "test_ratio":0.1, "lr": 0.0001}, 
                  2: {**model_custom_params, "dropout_rate": 0.1, "batch_size": 16, "test_ratio":0.1, "lr": 0.0001},
                  3:{**model_custom_params, "dropout_rate": 0.05, "batch_size": 32, "test_ratio":0.1, "lr": 0.0001},             
}
