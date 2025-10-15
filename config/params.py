import numpy as np
import random

primary_seed = 42
np.random.seed(primary_seed)
random.seed(primary_seed)


patches_params = {1:{'size': 1000, 'overlap': 0.75}, 2:{'size': 1500, 'overlap': 0.75}, 3:{'size': 2000, 'overlap': 0.75}}

morphology_features = ["area","extent", "solidity", "perimeter", "diameter_area", "convex_hull_area", "minor_axis_length",
                "perimeter_crofton","major_axis_length", "orientation", "diameter_max", "eccentricity"]

features_params = {'lifetime': ['lifetime_mean'], 'lifetime_morphology': morphology_features + ['lifetime_mean'], 'morphology': morphology_features}

lifetime_distribution_params = {10: 1.3, 18: 0.73, 23: 0.585, 34:0.383, 42: 0.31}

full_graph_max_dist_params = {1: 30, 2:40, 3:50}

lifetime_distribution_max_val = 13
lifetime_distribution_bin_range = 0.73


shuffling_seed_list = [435, 410, 489, 521, 1425, 1536, 1680, 1825, 2287, 3583, 3658, 3812, 4013, 4507, 6913, 8280, 8936, 9196, 9675, 9864]

# Seed handling
# gnn_iteration = 20
gnn_iteration = 3

gnn_evaluation_seeds = [random.randint(1, 10000) for _ in range(gnn_iteration)]
# gnn_evaluation_seeds = [6687, 35, 971, 1760, 2304, 3039, 4742, 1315, 8187, 952]




model_custom_params = {
    "model_type": "GAT",           # Options: "GAT" or "GCN"
    "hidden_layers": [64, 128, 64], # List of hidden layer sizes
    "output_size": 1,              # Usually 1 for binary classification
    "pooling": "max",              # Options: "mean" or "max"
    "heads": 1,                    # Attention heads for GAT; 1 for GCN
    "weight_decay": 0.0001,        # L2 regularization
    "epochs": 50,                  # Number of training epochs
    # "epochs": 5,                  # Number of training epochs
    "class_ratio": 1.0             # Positive class weight for imbalance
}


# model_params = {
#                   1:{**model_custom_params, "dropout_rate": 0.1, "batch_size": 16, "test_ratio":0.2, "lr": 0.001}, #output_layer
#                   2:{**model_custom_params, "dropout_rate": 0.05, "batch_size": 16, "test_ratio":0.2, "lr": 0.001}, #dropout + output_layer
#                   3:{**model_custom_params, "dropout_rate": 0.1, "batch_size": 32, "test_ratio":0.2, "lr": 0.001}, #batch_size + output_layer
#                   4:{**model_custom_params, "dropout_rate": 0.1, "batch_size": 16, "test_ratio":0.1, "lr": 0.001}, #test_ratio + output_layer
#                   5: {**model_custom_params, "dropout_rate": 0.1, "batch_size": 16, "test_ratio":0.1, "lr": 0.0001}, #test_ratio + lr + output_layer

#                   6:{**model_custom_params, "dropout_rate": 0.1, "batch_size": 16, "test_ratio":0.2, "lr": 0.0001}, #lr + output_layer 
#                   7:{**model_custom_params, "dropout_rate": 0.1, "batch_size": 32, "test_ratio":0.1, "lr": 0.0001}, #batch_size + test_ratio + lr + output_layer
#                   8:{**model_custom_params, "dropout_rate": 0.05, "batch_size": 32, "test_ratio":0.1, "lr": 0.0001} #dropout + batch_size + test_ratio + lr + output_layer
                  
# }

model_params = {
                  1:{**model_custom_params, "dropout_rate": 0.1, "batch_size": 32, "test_ratio":0.1, "lr": 0.0001}, 
                  2: {**model_custom_params, "dropout_rate": 0.1, "batch_size": 16, "test_ratio":0.1, "lr": 0.0001},
                  3:{**model_custom_params, "dropout_rate": 0.05, "batch_size": 32, "test_ratio":0.1, "lr": 0.0001},             
}