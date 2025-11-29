import numpy as np
import random
import config.const as const

PRIMARY_SEED = const.PRIMARY_SEED
np.random.seed(PRIMARY_SEED)
random.seed(PRIMARY_SEED)


patches_params = {1:{'size': 1000, 'overlap': 0.75}, 2:{'size': 1500, 'overlap': 0.75}, 3:{'size': 2000, 'overlap': 0.75}}

features_params = {'lifetime': ['lifetime_mean']}

lifetime_distribution_params = {10: 1.3, 18: 0.73, 23: 0.585, 34:0.383, 42: 0.31}

full_graph_max_dist_params = {1: 30, 2:40, 3:50}

lifetime_distribution_max_val = 13
lifetime_distribution_bin_range = 0.73

# shuffling_seed_list = [435, 410, 489, 521, 1425, 1536, 1680, 1825, 2287, 3583, 3658, 3812, 4013, 4507, 6913, 8280, 8936, 9196, 9675, 9864]
shuffling_seed_list = [435, 410]

# Seed handling
# seeds_amount = 20
seeds_amount = 3
seeds = [random.randint(1, 10000) for _ in range(seeds_amount)]


model_custom_params = {
    "model_type": "GAT",           # Options: "GAT" or "GCN"
    "hidden_layers": [64, 128, 64], # List of hidden layer sizes
    "output_size": 1,              # Usually 1 for binary classification
    "pooling": "max",              # Options: "mean" or "max"
    "heads": 1,                    # Attention heads for GAT; 1 for GCN
    "weight_decay": 0.0001,        # L2 regularization
    "epochs": 5,                  # Number of training epochs
    "class_ratio": 1.0             # Positive class weight for imbalance
}


model_params = {
                  1:{**model_custom_params, "dropout_rate": 0.1, "batch_size": 32, "test_ratio":0.1, "lr": 0.0001}, 
                  2: {**model_custom_params, "dropout_rate": 0.1, "batch_size": 16, "test_ratio":0.1, "lr": 0.0001},
                  3:{**model_custom_params, "dropout_rate": 0.05, "batch_size": 32, "test_ratio":0.1, "lr": 0.0001},             
}