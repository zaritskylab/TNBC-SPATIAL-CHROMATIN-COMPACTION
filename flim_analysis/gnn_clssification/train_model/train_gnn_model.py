import os

from utils.data_func import *
import config.const as const
import config.params as params


import torch
import pandas as pd
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.nn import GATConv
import numpy as np
from tqdm import tqdm
import pickle
import random

from torch.nn import BatchNorm1d
from joblib import Parallel, delayed
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ModuleList, Dropout, ReLU
from datetime import datetime


# from model import GCN_Model


class GNNBlock(torch.nn.Module):
    """
    A modular GNN layer block consisting of a GAT or GCN convolution,
    followed by batch normalization, ReLU activation, and dropout.

    Parameters
    ----------
    block_type : str
        Type of GNN block: "GAT" or "GCN".
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    dropout_rate : float
        Dropout rate applied after activation.
    """
    def __init__(self, block_type, in_channels, out_channels, dropout_rate, heads):
        super().__init__()
        # self.conv = GCNConv(in_channels, out_channels)
        if block_type=="GAT":
            self.conv = GATConv(in_channels, out_channels, heads=heads, concat=True, dropout=dropout_rate)
            conv_out_channels = out_channels * heads
        elif block_type=="GCN":
            self.conv = GCNConv(in_channels, out_channels)
            conv_out_channels = out_channels
        else:
            raise ValueError("block_type must be 'GAT' or 'GCN'")

        self.batch_norm = BatchNorm1d(conv_out_channels)
        # self.activation = torch.nn.ELU()
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.out_channels = conv_out_channels


        # self.relu = torch.nn.ReLU()
        # self.batch_norm = BatchNorm1d(out_channels)  # Add batch normalization
        # self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index):

        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x

class GNN_Model(torch.nn.Module):
    """
    Graph Neural Network model with multiple GNN blocks and global pooling.

    Parameters:
        input_size (int): Node input feature dimension (e.g., 1)
        model_type (str): 'GAT' or 'GCN'
        hidden_channels (list[int]): Output dims for each GNN block (before heads)
        output_size (int): Output size (e.g., 1 for binary classification)
        pooling (str): 'mean' or 'max' pooling
        dropout_rate (float): Dropout rate
        heads (int): Number of attention heads in GAT blocks
    """
    # def __init__(self, input_size, model_type="GAT", hidden_channels=[8, 8], output_size=1,
    #          pooling="mean", dropout_rate=0.3, heads=1):

    def __init__(self, input_size, dropout_rate, model_type="GAT", hidden_channels=[64, 128, 64], output_size=1,
            pooling="max",  heads=1):
        
        super().__init__()

        
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.gnn_blocks = torch.nn.ModuleList()
        
        in_channels = input_size

        for out_channels in hidden_channels:
            self.gnn_blocks.append(GNNBlock(model_type, in_channels, out_channels, dropout_rate, heads))
            # Adjust for GAT concatenation
            in_channels = out_channels * heads if model_type == "GAT" else out_channels

        self.output_layer = torch.nn.Linear(in_channels, output_size)


        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(64, output_size)
        )

        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for block in self.gnn_blocks:
            x = block(x, edge_index)

        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        else:
            raise ValueError("Pooling must be either 'mean' or 'max'")

        out = self.mlp_head(x)
        # out = self.output_layer(x)
        return out
    



def train(model, data_dict, train_indices, optimizer, device, batch_size, pos_weight=1):
    """
    Trains a GNN model for one epoch using binary cross-entropy loss.

    Parameters
    ----------
    model : torch.nn.Module
        The GNN model to train.
    data_dict : dict
        Dictionary of graph data (PyG `Data` objects).
    train_indices : list
        Indices of the training graphs in `data_dict`.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    device : torch.device
        Device to perform training on (CPU or GPU).
    batch_size : int, optional
        Batch size for training.
    pos_weight : torch.Tensor, optional
        Weight used to handle class imbalance in BCEWithLogitsLoss.

    Returns
    -------
    float
        Average training loss over all batches.
    model : torch.nn.Module
        Updated model after training.
    """

    # pos_weight=torch.tensor([pos_weight])
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    loss_fn = torch.nn.BCEWithLogitsLoss()


    model.train()
    train_data = [data_dict[idx] for idx in train_indices]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    total_loss = 0
    for batch in train_loader:

        batch = batch.to(device)
        optimizer.zero_grad() 
        out = model(batch)


        loss = loss_fn(out, batch.y.float().view(-1, 1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Move the batch data back to the CPU after processing
        batch = batch.to('cpu')
        
    return total_loss/len(train_loader), model


def evaluate(model, data_dict, val_indices, device, pos_weight=1):
    """
    Evaluate the GNN model on a validation set and return predictions and loss.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GNN model to evaluate.
    data_dict : dict
        Dictionary of PyTorch Geometric Data objects indexed by ID.
    val_indices : list
        List of indices (keys in data_dict) for validation samples.
    device : torch.device
        The device (CPU or GPU) to run evaluation on.
    pos_weight : torch.Tensor, optional
        Class weight to address imbalance in BCEWithLogitsLoss.

    Returns
    -------
    val_loss : torch.Tensor
        Binary cross-entropy loss on the validation set.
    results_df : pd.DataFrame
        DataFrame containing evaluation results per sample with columns:
        - sample_name
        - y_true
        - y_pred_logits
        - y_pred_probs
    """
    model.eval()
    y_true, y_pred_logits, y_pred_probs, sample_names = [], [], [], []
    sigmoid = torch.nn.Sigmoid()

    # pos_weight=torch.tensor([pos_weight])
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for idx in val_indices:
            sample_name = data_dict[idx].name
            data = data_dict[idx].to(device)
            out = model(data)
            # Store required information
            y_true.append(data.y.cpu().float().view(-1, 1))
            y_pred_logits.append(out.cpu().float().view(-1, 1))
            y_pred_probs.append(sigmoid(out.cpu()).float().view(-1, 1)) 
            sample_names.append(sample_name)

            data = data.to('cpu')
        # Concatenate tensors
    y_true = torch.cat(y_true, dim=0)
    y_pred_logits = torch.cat(y_pred_logits, dim=0)
    

    val_loss = loss_fn(y_pred_logits, y_true)

    # Concatenate all results
    y_true = y_true.numpy().flatten()
    y_pred_logits = y_pred_logits.numpy().flatten()
    y_pred_probs = torch.cat(y_pred_probs, dim=0).numpy().flatten()


    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        "sample_name": sample_names,
        "y_true": y_true,
        "y_pred_logits": y_pred_logits,
        "y_pred_probs": y_pred_probs
    })

    return val_loss, results_df

def train_and_validate(
    data_dict,
    model,
    optimizer,
    train_idx,
    val_idx,
    device,
    epochs,
    batch_size,
    pos_weight=1,
    run=None,
    seed_val=None,
    fold_idx=None
    ): 
    """
    Train a GNN model for multiple epochs and evaluate on a validation set.

    Parameters
    ----------
    data_dict : dict
        Dictionary of PyTorch Geometric Data objects.
    model : torch.nn.Module
        GNN model to train and validate.
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    train_idx : list
        List of training sample indices (keys in data_dict).
    val_idx : list
        List of validation sample indices (keys in data_dict).
    device : torch.device
        Device to run the model on.
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Batch size for DataLoader.
    run : neptune.run.Run, optional
        Neptune run object. If provided, logs train/val losses.
    seed_val : int, optional
        Seed number for Neptune hierarchy.
    fold_idx : int, optional
        Fold index for Neptune hierarchy.

    Returns
    -------
    best_model_state : dict
        The state_dict of the best model (lowest validation loss).
    train_losses : list
        List of average training loss per epoch.
    val_losses : list
        List of validation losses per epoch.
    """
    best_val_loss = float('inf')
    best_model_state = None
    # Scheduler for learning rate adjustment
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Lists to track losses
    train_losses = []
    val_losses = []

    train_model = model
    for epoch in range(1, epochs + 1):
        # Training
        avg_train_loss, train_model = train(train_model, data_dict, train_idx, optimizer, device, batch_size=batch_size, pos_weight=pos_weight)
        train_losses.append(avg_train_loss)

        # Validation
        val_loss, _ = evaluate(train_model, data_dict, val_idx, device, pos_weight=pos_weight)
        val_losses.append(val_loss)

        # Step the scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{epochs}, Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Log to Neptune if run is provided
        if run is not None and seed_val is not None and fold_idx is not None:
            run[f"seed_{seed_val}/fold_{fold_idx}/train/loss"].append(avg_train_loss)
            run[f"seed_{seed_val}/fold_{fold_idx}/val/loss"].append(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = train_model.state_dict()

    return best_model_state, train_losses, val_losses


def calculate_auc_and_roc(y_true_all, probs_all):
    """
    Compute AUC and ROC curve for predicted probabilities.

    Parameters
    ----------
    y_true_all : array-like
        Ground truth binary labels.
    probs_all : array-like
        Predicted probabilities (from sigmoid output).

    Returns
    -------
    auc_score : float
        Area under the ROC curve (AUC).
    fpr : array
        False Positive Rate values for the ROC curve.
    tpr : array
        True Positive Rate values for the ROC curve.
    """
    # Calculate AUC score
    auc_score = roc_auc_score(y_true_all, probs_all)

    # Calculate FPR and TPR using the combined results
    fpr, tpr, _ = roc_curve(y_true_all, probs_all)

    print(f"Combined AUC: {auc_score}")

    return auc_score, fpr, tpr


def get_train_val_split(mapping_subgraph_df, train_val_indices, test_size=0.2, random_val=42):
    """
    Perform a stratified train-validation split based on leap_ID and its category.

    Parameters:
    - mapping_subgraph_df (pd.DataFrame): DataFrame containing 'leap_ID', 'category', and 'gnn_data_idx' columns.
    - train_val_indices (list): List of indices corresponding to the training/validation set.
    - test_size (float): Proportion of the validation set in the train-validation split.
    - random_val (int): Random seed for reproducibility.

    Returns:
    - train_gnn_indices (list): List of `gnn_data_idx` for the training split.
    - val_gnn_indices (list): List of `gnn_data_idx` for the validation split.
    """
    # Filter the DataFrame to keep only rows with indices in train_val_indices
    train_mapping_subgraph_df = mapping_subgraph_df[mapping_subgraph_df['gnn_data_idx'].isin(train_val_indices)]

    # Group and label data for stratified split
    group_labels = train_mapping_subgraph_df.groupby('leap_ID')['category'].first()
    original_graph_keys = group_labels.index.to_list()  # List of leap_IDs
    original_graph_labels = group_labels.values  # Corresponding categories

    # Perform a stratified split based on `leap_ID` and its category
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_val)
    train_indices, val_indices = next(stratified_split.split(original_graph_keys, original_graph_labels))

    # Map indices to leap_IDs
    train_leap_ids = [original_graph_keys[i] for i in train_indices]
    val_leap_ids = [original_graph_keys[i] for i in val_indices]

    # Get train and validation `gnn_data_idx` based on the split
    train_gnn_indices = mapping_subgraph_df[mapping_subgraph_df['leap_ID'].isin(train_leap_ids)]['gnn_data_idx'].tolist()
    val_gnn_indices = mapping_subgraph_df[mapping_subgraph_df['leap_ID'].isin(val_leap_ids)]['gnn_data_idx'].tolist()

    return train_gnn_indices, val_gnn_indices


def get_train_val_test_indices(mapping_subgraph_df, k_out=5, seed_val=42):
    """
    Generate train/validation and test indices for each fold using StratifiedKFold.

    Parameters:
    - mapping_subgraph_df (pd.DataFrame): DataFrame containing 'leap_ID', 'category', and 'gnn_data_idx' columns.
    - k_out (int): Number of splits for StratifiedKFold.
    - seed_val (int): Random seed for reproducibility.

    Returns:
    - folds (list of dict): A list of dictionaries, each containing the train_val_indices and test_indices for a fold.
    """
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=k_out, shuffle=True, random_state=seed_val)

    # Group the data by leap_ID and category
    grouped = mapping_subgraph_df.groupby('leap_ID')
    X = grouped.size().index  # Unique leap_IDs
    ys = grouped['category'].first().values  # Corresponding categories

    # Generate splits
    splits_list = skf.split(X, ys)

    # Store indices for each fold
    folds = []

    for split_idx, (train_val_index, test_index) in enumerate(splits_list, start=1):
        # Get leap_IDs for train/val and test splits
        train_val_leap_ids = X[train_val_index]
        test_leap_ids = X[test_index]

        # Map back to gnn_data_idx
        train_val_indices = mapping_subgraph_df[mapping_subgraph_df['leap_ID'].isin(train_val_leap_ids)]['gnn_data_idx'].tolist()
        test_indices = mapping_subgraph_df[mapping_subgraph_df['leap_ID'].isin(test_leap_ids)]['gnn_data_idx'].tolist()

        # Append indices to the folds list
        folds.append({
            'train_val_indices': train_val_indices,
            'test_indices': test_indices
        })

    return folds


def aggregate_results_by_leap_id(subsamples_results):
    """
    Aggregates results by leap_ID by calculating the mean probability of subgraphs, 
    excluding probabilities between 0.4 and 0.6. Defaults to 0.5 if no valid 
    probabilities remain for a leap_ID.

    Parameters:
    - test_pred_results (pd.DataFrame): DataFrame containing subgraph-level predictions 
      with columns 'leap_ID', 'y_true', and 'y_pred_probs'.

    Returns:
    - aggregated_results_df (pd.DataFrame): DataFrame with aggregated results, containing:
        - 'leap_ID': Unique leap_ID.
        - 'y_true': True label for the leap_ID.
        - 'aggregated_prob': Aggregated probability for the leap_ID.
    """
    # Group results by leap_ID
    leap_id_groups = subsamples_results.groupby("leap_ID")
    aggregated_results = []

    for leap_id, group in leap_id_groups:
        # Assume the true label is the same for all subgraphs of a leap_ID
        y_true = group["y_true"].iloc[0]
        fold = group['fold'].iloc[0]
        probs = group["y_pred_probs"]

        mean_prob = probs.mean()

        # Filter out probabilities between 0.4 and 0.6
        filtered_probs = probs[(probs < 0.4) | (probs > 0.6)]

        # If filtered_probs is empty or contains NaN, return the mean_prob as a default value
        if filtered_probs.empty or filtered_probs.isnull().all():
            mean_prob_filter = mean_prob
        else:
            mean_prob_filter = filtered_probs.mean()  
    

        aggregated_results.append({
            "leap_ID": leap_id,
            "fold": fold,
            "y_true": y_true,
            "y_pred_probs_agg": mean_prob,
            "y_pred_probs_agg_filter": mean_prob_filter
        })

    # Convert aggregated results to a DataFrame
    aggregated_results_df = pd.DataFrame(aggregated_results)

    return aggregated_results_df


def gnn_train_test_k_fold(data_dict, mapping_subgraph_df, device, seed_dir, seed_val, k_out, run_params, aggregation=True, neptune_run=None):

    """
    Train and evaluate a GNN model using k-fold cross-validation on graph-based FLIM data.

    This function:
    - Splits LEAP-level data into k folds using StratifiedKFold
    - For each fold: splits train/validation data stratified by LEAP category
    - Trains a GNN model on the training set and evaluates on the test set
    - Aggregates and saves fold-level and LEAP-level predictions
    - Optionally computes performance after filtering out low-confidence subgraph predictions

    Parameters
    ----------
    data_dict : dict
        Dictionary of PyTorch Geometric `Data` objects (indexed by subgraph).
    mapping_subgraph_df : pd.DataFrame
        DataFrame containing graph metadata (e.g., graph_ID, leap_ID, category, gnn_data_idx(=key in data_dict).
    device : torch.device
        Device to run the model on (e.g., `torch.device("cuda")` or `torch.device("cpu")`).
    seed_dir : str
        Path to the directory where model and results should be saved.
    seed_val : int, optional
        Random seed for reproducibility. Default is 42.
    k_out : int, optional
        Number of folds for StratifiedKFold. Default is 5.
    epochs : int, optional
        Number of training epochs per fold. Default is 50.
    model_type : str, optional
        Type of GNN model to use ("GAT" or "GCN"). Default is "GAT".
    batch_size : int, optional
        Mini-batch size for training. Default is 32.
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-3.
    weight_decay : float, optional
        Weight decay (L2 regularization) coefficient for the optimizer. Helps prevent overfitting. Default is 1e-4.
    aggregation : bool, optional
        If True, computes performance metrics using LEAP-level aggregation. Default is True.

    Returns
    -------
    If aggregation is True:
        auc : float
            AUC score from subgraph-level predictions.
        fpr : np.ndarray
            False positive rates for ROC curve.
        tpr : np.ndarray
            True positive rates for ROC curve.
        auc_agg : float
            AUC score from aggregated LEAP-level predictions (mean of subgraph probs).
        fpr_agg : np.ndarray
            FPR for ROC on aggregated predictions.
        tpr_agg : np.ndarray
            TPR for ROC on aggregated predictions.
        auc_agg_filter : float
            AUC after filtering out low-confidence predictions (probs between 0.4 and 0.6).
        fpr_agg_filter : np.ndarray
            FPR for filtered ROC curve.
        tpr_agg_filter : np.ndarray
            TPR for filtered ROC curve.

    If aggregation is False:
        auc : float
            AUC score from subgraph-level predictions.
        fpr : np.ndarray
            False positive rates for ROC curve.
        tpr : np.ndarray
            True positive rates for ROC curve.
    """
    results = []
    models_dict = {}

    folds = get_train_val_test_indices(mapping_subgraph_df, k_out=k_out, seed_val=42)

    # Access train/val and test indices for each fold
    for fold_idx, fold in enumerate(folds, start=1):

        print(f"start with split {fold_idx}")
        train_val_indices = fold['train_val_indices']
        test_indices = fold['test_indices']
        
        #split into train validation
        test_ratio=run_params.get("test_ratio")
        train_gnn_indices, val_gnn_indices = get_train_val_split(mapping_subgraph_df, train_val_indices, test_size=test_ratio, random_val=seed_val)

        # Initialize model and optimizer


        input_size = data_dict[train_gnn_indices[0]].x.shape[1]

        hidden_layers=run_params.get("hidden_layers")
        dropout_rate=run_params.get("dropout_rate")
        model_type=run_params.get("model_type")
        output_size=run_params.get("output_size")
        pooling=run_params.get("pooling")
        heads=run_params.get("heads")

        weight_decay=run_params.get("weight_decay")
        lr=run_params.get("lr")

        epochs=run_params.get("epochs")
        batch_size=run_params.get("batch_size")

        model = GNN_Model(input_size=input_size, dropout_rate=dropout_rate, model_type=model_type, hidden_channels=hidden_layers, output_size=output_size, pooling=pooling, heads=heads).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Train and validate
        best_model_state, train_loss_list, val_loss_list = train_and_validate(data_dict, model, optimizer, train_gnn_indices, val_gnn_indices, device, epochs, batch_size=batch_size, run=neptune_run, seed_val=seed_val, fold_idx=fold_idx)

        # Load best model and evaluate on test data
        model.load_state_dict(best_model_state)
        _, test_pred_results = evaluate(model, data_dict, test_indices, device)
        
        best_model=model
        # Add fold number to test predictions
        test_pred_results['fold'] = fold_idx
        test_pred_results["leap_ID"] = test_pred_results["sample_name"].str[:3]
        results.append(test_pred_results)

        # Save the model and the associated loss lists in the dictionary
        models_dict[fold_idx] = {
            'model_state': best_model.state_dict(),
            'train_loss': train_loss_list,
            'val_loss': val_loss_list
        }

        print(f"finish with split {fold_idx}")
    
    combined_results_df = pd.concat(results, ignore_index=True)


    # Save the combined results to a CSV file
    results_save_path = os.path.join(seed_dir, "k_fold_results.csv")
    combined_results_df.to_csv(results_save_path, index=False)
    print(f"Saved combined fold results at {results_save_path}")

    models_dict_save_path = os.path.join(seed_dir, "models_dict.pth")
    torch.save(models_dict, models_dict_save_path)
    print(f"Saved models dictionary at {models_dict_save_path}")

    auc, fpr, tpr = calculate_auc_and_roc(combined_results_df['y_true'], combined_results_df['y_pred_probs'])

    if aggregation:
        aggregation_results_df = aggregate_results_by_leap_id(combined_results_df)
        # Save the aggregated results to a CSV file
        aggregation_results_save_path = os.path.join(seed_dir, "k_fold_aggregation_results.csv")
        aggregation_results_df.to_csv(aggregation_results_save_path, index=False)
        print(f"Saved aggregated results at {aggregation_results_save_path}")

        auc_agg, fpr_agg, tpr_agg = calculate_auc_and_roc(aggregation_results_df['y_true'], aggregation_results_df['y_pred_probs_agg'])
        auc_agg_filter, fpr_agg_filter, tpr_agg_filter = calculate_auc_and_roc(aggregation_results_df['y_true'], aggregation_results_df['y_pred_probs_agg_filter'])

        if neptune_run is not None:
            neptune_run[f"seed_{seed_val}/val/auc"] = auc
            neptune_run[f"seed_{seed_val}/val/auc_agg"] = auc_agg
            neptune_run[f"seed_{seed_val}/val/auc_agg_filter"] = auc_agg_filter

        return auc, fpr, tpr, auc_agg, fpr_agg, tpr_agg, auc_agg_filter, fpr_agg_filter, tpr_agg_filter

    return auc, fpr, tpr

def calculate_auc_statistics(seed_results_df, auc_column='auc_score'):
    """
    Calculate median AUC, best AUC, best seed, and closest-to-median seed from a DataFrame.

    Parameters:
    - seed_results_df (pd.DataFrame): DataFrame containing results for each seed, including an AUC column.
    - auc_column (str): The column name in the DataFrame representing the AUC score. Default is 'auc_score'.

    Returns:
    - results (dict): A dictionary with the following keys:
        - 'median_auc': The median AUC score.
        - 'best_auc': The highest AUC score.
        - 'best_seed': The seed corresponding to the best AUC score.
        - 'closest_to_median_seed': The seed corresponding to the AUC score closest to the median.
    """
    if seed_results_df.empty:
        raise ValueError("The seed_results_df is empty. Cannot calculate statistics.")

    # Calculate statistics
    median_auc = seed_results_df[auc_column].median()
    best_auc = seed_results_df[auc_column].max()
    best_seed = seed_results_df.loc[seed_results_df[auc_column].idxmax(), 'seed_val']
    closest_to_median_seed = seed_results_df.iloc[
        (seed_results_df[auc_column] - median_auc).abs().idxmin()]['seed_val']

    # Return results as a dictionary
    return {
        'median_auc': median_auc,
        'best_auc': best_auc,
        'best_seed': best_seed,
        'closest_to_median_seed': closest_to_median_seed
    }


def preprocess_argsinh(data_dict):
    """
    Applies the argsinh normalization to the node features (x) of each graph in data_dict.

    Args:
        data_dict (dict): Dictionary of graph data objects.

    Returns:
        dict: Updated data_dict with normalized features.
    """
    for key, data in data_dict.items():
        # Apply argsinh normalization to node features
        data.x = torch.asinh(data.x)  # PyTorch's asinh function
    return data_dict


def preprocess_minmax(data_dict, feature_min=0, feature_max=13):
    """
    Applies min-max normalization to the node features (x) of each graph in data_dict.

    Args:
        data_dict (dict): Dictionary of graph data objects.
        feature_min (float): Minimum possible value of the feature.
        feature_max (float): Maximum possible value of the feature.

    Returns:
        dict: Updated data_dict with normalized features.
    """
    for key, data in data_dict.items():
        data.x = (data.x - feature_min) / (feature_max - feature_min)  # Scale to [0, 1]
    return data_dict


def load_and_preprocess_data(
    subgraphs_path, data_path, preprocess_method="minmax"
):
    """
    Load saved GNN data and associated subgraph metadata, and apply feature normalization.

    Parameters
    ----------
    subgraphs_path : str
        Path to the `.pkl` file containing the subgraph metadata DataFrame.
    data_path : str
        Path to the `.pkl` file containing the GNN dataset (dictionary of PyG `Data` objects).
    preprocess_method : str, optional
        Method to normalize node features. Supported options:
        - "minmax": Min-max normalization to [0, 1]
        - "argsinh": Arcsinh transformation for stabilizing variance
        Default is "minmax".

    Returns
    -------
    preprocessed_data : dict
        Dictionary of PyG `Data` objects with normalized features.
    subgraphs_df : pd.DataFrame
        Metadata DataFrame describing each subgraph.
    """

    # Load subgraphs DataFrame
    with open(subgraphs_path, 'rb') as file:
        subgraphs_df = pickle.load(file)

    # Load GNN data
    with open(data_path, 'rb') as p:
        gnn_data = pickle.load(p)

    # Normalize the data based on the chosen method
    if preprocess_method == "argsinh":
        preprocessed_data = preprocess_argsinh(gnn_data)
    elif preprocess_method == "minmax":
        preprocessed_data = preprocess_minmax(gnn_data)
    else:
        raise ValueError(f"Unknown preprocessing method: {preprocess_method}")

    return preprocessed_data, subgraphs_df


def run_patch_gnn_pipeline_per_one_model_parameters(
    feature_type: str,
    patch_size: int,
    overlap: float,
    max_dist: int,
    k_out: int,
    model_params: dict,
    primary_seed: int = 42,
    aggregation: bool = True, 
    tissue_resolution: str = 'patch_tissue', 
    
    ):
    """
    Full training pipeline for GNN evaluation on patch-level FLIM data.

    Loads preprocessed PyTorch Geometric data, trains multiple models across seeds,
    and evaluates performance with and without aggregation.

    Parameters
    ----------
    feature_type : str
        Type of features to use (e.g., "lifetime").
    patch_size : int
        Size of the image patch in pixels.
    overlap : float
        Patch overlap as a float between 0 and 1.
    max_dist : int
        Max distance threshold for edge creation in graphs.
    k_out : int
        Number of outer folds for cross-validation.
    epochs : int
        Number of training epochs.
    iterations : int
        Number of different seed runs for robustness.
    primary_seed : int
        Main random seed for reproducibility.
    aggregation : bool
        Whether to compute and evaluate aggregated predictions at LEAP level.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    evaluation_seeds = params.gnn_evaluation_seeds
    print(f"Evaluation seeds: {evaluation_seeds}")


    patch_params = f'size_{patch_size}_overlap_{overlap}'
    graph_file_dir = os.path.join(const.gnn_dir, feature_type, tissue_resolution, f"max_distance_{max_dist}", 'pytorch_geo')
    data_pytorch_geo_dir = os.path.join(graph_file_dir, patch_params, 'data')
    mapping_df_file = os.path.join(data_pytorch_geo_dir, "graphs_df_pytorch_geo.pkl")
    data_file = os.path.join(data_pytorch_geo_dir, "data_pytorch_geo.pkl")

    print(f"Loading data from {data_pytorch_geo_dir}...")
    preprocessed_data, subgraphs_df = load_and_preprocess_data(mapping_df_file, data_file)

    date_str = datetime.now().strftime("%d_%m_%Y")
    model_results_dir = os.path.join(graph_file_dir, patch_params, 'results', date_str)
    os.makedirs(model_results_dir, exist_ok=True)
    results_list = []
    
    print("Start processing model parameters: " + ", ".join([f"{k}={v}" for k, v in model_params.items()]))

    model_param_dir = "_".join([
        f"{k}_{str(v).replace('.', 'p')}" for k, v in sorted(model_params.items())
    ])

    results_dir = os.path.join(model_results_dir, model_param_dir)

    seed_results_list = []
    auc_scores, auc_agg_scores, auc_agg_filter_scores = [], [], []


    for seed_val in evaluation_seeds:
        print(f"  Start processing seed={seed_val}")
        seed_dir = os.path.join(results_dir, str(seed_val))
        os.makedirs(seed_dir, exist_ok=True)

        ######## FIX ALL RANDOM SEEDS FOR REPRODUCIBILITY ########
        np.random.seed(seed_val)
        random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        ######## FIX ALL RANDOM SEEDS FOR REPRODUCIBILITY ########


        # ######## neptune initialize start ########
        # # Logging each model+seed combo as a separate run
        # name = " | ".join([f"{k.upper()}: {v}" for k, v in model_params.items()] + [f"SEED: {seed_val}"])
        # tags = [f"{k}:{v}" for k, v in model_params.items()] + [
        #     f"patch_size:{patch_size}",
        #     f"feature_type:{feature_type}"
        # ]

        # run = neptune.init_run(
        #     project=params.neptune_project_name,
        #     api_token=params.neptune_api_token,
        #     name=name,
        #     tags=tags
        # )


        # run["parameters"] = {
        #     **model_params,
        #     "hidden_channels": f"{model_params['hidden_layers']}",
        #     "feature_type": feature_type,
        #     "patch_size": patch_size,
        #     "overlap": overlap,
        #     "max_dist": max_dist,
        #     "k_out": k_out,
        #     "aggregation": aggregation,
        #     "tissue_resolution": tissue_resolution,
        #     "seed": seed_val
        # }

        # ######## neptune initialize finish ########


        # results = gnn_train_test_k_fold(
        #     preprocessed_data, subgraphs_df, device, seed_dir,
        #     seed_val=seed_val, k_out=k_out, run_params=model_params,
        #     aggregation=aggregation, neptune_run=run
        # )

        # auc_score, fpr, tpr, auc_agg, fpr_agg, tpr_agg, auc_agg_filter, fpr_agg_filter, tpr_agg_filter = results

        # run.stop()

        results = gnn_train_test_k_fold(
            preprocessed_data, subgraphs_df, device, seed_dir,
            seed_val=seed_val, k_out=k_out, run_params=model_params,
            aggregation=aggregation
        )

        auc_score, fpr, tpr, auc_agg, fpr_agg, tpr_agg, auc_agg_filter, fpr_agg_filter, tpr_agg_filter = results

        print("\n Finished logging to Neptune")

        seed_results_list.append({
            'seed_val': seed_val,
            'auc_score': auc_score,
            'fpr': fpr,
            'tpr': tpr,
            'auc_agg': auc_agg,
            'fpr_agg': fpr_agg,
            'tpr_agg': tpr_agg,
            'auc_agg_filter': auc_agg_filter,
            'fpr_agg_filter': fpr_agg_filter,
            'tpr_agg_filter': tpr_agg_filter
        })

        auc_scores.append(auc_score)
        auc_agg_scores.append(auc_agg)
        auc_agg_filter_scores.append(auc_agg_filter)
        print(f"  Finished processing seed={seed_val}")

    seed_results_df = pd.DataFrame(seed_results_list)
    seed_results_path = os.path.join(results_dir, "seed_results.pkl")
    seed_results_df.to_pickle(seed_results_path)
    print(f"Saved seed-level results at {seed_results_path}")

    # Compute performance statistics
    subgraph_stats = calculate_auc_statistics(seed_results_df, 'auc_score')
    agg_stats = calculate_auc_statistics(seed_results_df, 'auc_agg')
    agg_filter_stats = calculate_auc_statistics(seed_results_df, 'auc_agg_filter')

    results_list.append({
        'date': date_str,
        'max_dist': max_dist,
        'patch_size': patch_size,
        'overlap': overlap,
        'feature_type': feature_type,
        **model_params,  # dynamically include all model hyperparameters
        **{f"{k}": v for k, v in subgraph_stats.items()},
        **{f"{k}_agg": v for k, v in agg_stats.items()},
        **{f"{k}_agg_filter": v for k, v in agg_filter_stats.items()},
    })

    results_df = pd.DataFrame(results_list)
    output_file = os.path.join(model_results_dir, f"summary_results_{model_param_dir}.pkl")
    results_df.to_pickle(output_file)
    print(f"Results saved to {output_file}")
    print(f"Finished processing max_dist={max_dist}")



# def run_patch_gnn_pipeline(
#     feature_type: str,
#     patch_size: int,
#     overlap: float,
#     max_dist: int = 30,
#     k_out: int = 5,
#     epochs: int = 50,
#     primary_seed: int = 42,
#     aggregation: bool = True, 
#     tissue_resolution: str = 'patch_tissue'
#     ):
#     """
#     Full training pipeline for GNN evaluation on patch-level FLIM data.

#     Loads preprocessed PyTorch Geometric data, trains multiple models across seeds,
#     and evaluates performance with and without aggregation.

#     Parameters
#     ----------
#     feature_type : str
#         Type of features to use (e.g., "lifetime").
#     patch_size : int
#         Size of the image patch in pixels.
#     overlap : float
#         Patch overlap as a float between 0 and 1.
#     max_dist : int
#         Max distance threshold for edge creation in graphs.
#     k_out : int
#         Number of outer folds for cross-validation.
#     epochs : int
#         Number of training epochs.
#     iterations : int
#         Number of different seed runs for robustness.
#     primary_seed : int
#         Main random seed for reproducibility.
#     aggregation : bool
#         Whether to compute and evaluate aggregated predictions at LEAP level.
#     """

#     for _, model_params in params.model_params_dict.items():
#         run_patch_gnn_pipeline_per_one_model_parameters(feature_type, patch_size, overlap, max_dist, k_out, epochs, model_params, aggregation, tissue_resolution)    


def run_patch_gnn_pipeline_per_one_model_parameters_shuffling(
    feature_type: str,
    patch_size: int,
    overlap: float,
    max_dist: int,
    k_out: int,
    model_params: dict,
    seed_shuffle_val: int,
    aggregation: bool = True, 
    tissue_resolution: str = 'patch_tissue', 
    
    ):
    """
    Full training pipeline for GNN evaluation on patch-level FLIM data.

    Loads preprocessed PyTorch Geometric data, trains multiple models across seeds,
    and evaluates performance with and without aggregation.

    Parameters
    ----------
    feature_type : str
        Type of features to use (e.g., "lifetime").
    patch_size : int
        Size of the image patch in pixels.
    overlap : float
        Patch overlap as a float between 0 and 1.
    max_dist : int
        Max distance threshold for edge creation in graphs.
    k_out : int
        Number of outer folds for cross-validation.
    epochs : int
        Number of training epochs.
    iterations : int
        Number of different seed runs for robustness.
    seed_shuffle_val : int
        Main random seed for reproducibility.
    aggregation : bool
        Whether to compute and evaluate aggregated predictions at LEAP level.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    patch_params = f'size_{patch_size}_overlap_{overlap}'
    graph_file_dir = os.path.join(const.gnn_dir, feature_type, tissue_resolution, f"max_distance_{max_dist}", 'pytorch_geo', 'shuffling_lifetime')
    data_pytorch_geo_dir = os.path.join(graph_file_dir, patch_params, 'data', f'{seed_shuffle_val}')
    mapping_df_file = os.path.join(data_pytorch_geo_dir, "graphs_df_pytorch_geo.pkl")
    data_file = os.path.join(data_pytorch_geo_dir, "data_pytorch_geo.pkl")

    print(f"Loading data from {data_pytorch_geo_dir}...")
    preprocessed_data, subgraphs_df = load_and_preprocess_data(mapping_df_file, data_file)

    date_str = datetime.now().strftime("%d_%m_%Y")
    model_results_dir = os.path.join(graph_file_dir, patch_params, 'results')
    os.makedirs(model_results_dir, exist_ok=True)
    
    print("Start processing model parameters: " + ", ".join([f"{k}={v}" for k, v in model_params.items()]))

    model_param_dir = "_".join([
        f"{k}_{str(v).replace('.', 'p')}" for k, v in sorted(model_params.items())
    ])

    results_dir = os.path.join(model_results_dir, model_param_dir)

    seed_results_list = []
    auc_scores, auc_agg_scores, auc_agg_filter_scores = [], [], []


    print(f"  Start processing seed={seed_shuffle_val}")
    seed_dir = os.path.join(results_dir, str(seed_shuffle_val))
    os.makedirs(seed_dir, exist_ok=True)

    ######## FIX ALL RANDOM SEEDS FOR REPRODUCIBILITY ########
    np.random.seed(seed_shuffle_val)
    random.seed(seed_shuffle_val)
    torch.manual_seed(seed_shuffle_val)
    torch.cuda.manual_seed(seed_shuffle_val)
    torch.cuda.manual_seed_all(seed_shuffle_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######## FIX ALL RANDOM SEEDS FOR REPRODUCIBILITY ########


    results = gnn_train_test_k_fold(
        preprocessed_data, subgraphs_df, device, seed_dir,
        seed_val=seed_shuffle_val, k_out=k_out, run_params=model_params,
        aggregation=aggregation
    )

    auc_score, fpr, tpr, auc_agg, fpr_agg, tpr_agg, auc_agg_filter, fpr_agg_filter, tpr_agg_filter = results

    print("\n Finished logging to Neptune")

    seed_results_list.append({
        'seed_val': seed_shuffle_val,
        'auc_score': auc_score,
        'fpr': fpr,
        'tpr': tpr,
        'auc_agg': auc_agg,
        'fpr_agg': fpr_agg,
        'tpr_agg': tpr_agg,
        'auc_agg_filter': auc_agg_filter,
        'fpr_agg_filter': fpr_agg_filter,
        'tpr_agg_filter': tpr_agg_filter
    })

    auc_scores.append(auc_score)
    auc_agg_scores.append(auc_agg)
    auc_agg_filter_scores.append(auc_agg_filter)
    print(f"  Finished processing seed={seed_shuffle_val}")

    seed_results_df = pd.DataFrame(seed_results_list)
    seed_results_path = os.path.join(seed_dir, "seed_results.pkl")
    seed_results_df.to_pickle(seed_results_path)
    print(f"Saved seed-level results at {seed_results_path}")


def run_patch_gnn_pipeline_per_one_model_parameters_structure(
    feature_type: str,
    patch_size: int,
    overlap: float,
    max_dist: int,
    k_out: int,
    model_params: dict,
    seed_val: int,
    aggregation: bool = True, 
    tissue_resolution: str = 'patch_tissue', 
    
    ):
    """
    Full training pipeline for GNN evaluation on patch-level FLIM data.

    Loads preprocessed PyTorch Geometric data, trains multiple models across seeds,
    and evaluates performance with and without aggregation.

    Parameters
    ----------
    feature_type : str
        Type of features to use (e.g., "lifetime").
    patch_size : int
        Size of the image patch in pixels.
    overlap : float
        Patch overlap as a float between 0 and 1.
    max_dist : int
        Max distance threshold for edge creation in graphs.
    k_out : int
        Number of outer folds for cross-validation.
    epochs : int
        Number of training epochs.
    iterations : int
        Number of different seed runs for robustness.
    seed_val : int
        Random seed for reproducibility.
    aggregation : bool
        Whether to compute and evaluate aggregated predictions at LEAP level.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    patch_params = f'size_{patch_size}_overlap_{overlap}'
    graph_file_dir = os.path.join(const.gnn_dir, feature_type, 'patch_tissue', f"max_distance_{max_dist}", 'pytorch_geo', 'structure_only')
    data_pytorch_geo_dir = os.path.join(graph_file_dir, patch_params, 'data')
    mapping_df_file = os.path.join(data_pytorch_geo_dir, "graphs_df_pytorch_geo.pkl")
    data_file = os.path.join(data_pytorch_geo_dir, "data_pytorch_geo.pkl")

    print(f"Loading data from {data_pytorch_geo_dir}...")
    preprocessed_data, subgraphs_df = load_and_preprocess_data(mapping_df_file, data_file)

    date_str = datetime.now().strftime("%d_%m_%Y")
    model_results_dir = os.path.join(graph_file_dir, patch_params, 'results')
    os.makedirs(model_results_dir, exist_ok=True)
    
    print("Start processing model parameters: " + ", ".join([f"{k}={v}" for k, v in model_params.items()]))

    model_param_dir = "_".join([
        f"{k}_{str(v).replace('.', 'p')}" for k, v in sorted(model_params.items())
    ])

    results_dir = os.path.join(model_results_dir, model_param_dir)

    seed_results_list = []
    auc_scores, auc_agg_scores, auc_agg_filter_scores = [], [], []


    print(f"  Start processing seed={seed_val}")
    seed_dir = os.path.join(results_dir, str(seed_val))
    os.makedirs(seed_dir, exist_ok=True)

    ######## FIX ALL RANDOM SEEDS FOR REPRODUCIBILITY ########
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######## FIX ALL RANDOM SEEDS FOR REPRODUCIBILITY ########


    results = gnn_train_test_k_fold(
        preprocessed_data, subgraphs_df, device, seed_dir,
        seed_val=seed_val, k_out=k_out, run_params=model_params,
        aggregation=aggregation
    )

    auc_score, fpr, tpr, auc_agg, fpr_agg, tpr_agg, auc_agg_filter, fpr_agg_filter, tpr_agg_filter = results

    print("\n Finished logging to Neptune")

    seed_results_list.append({
        'seed_val': seed_val,
        'auc_score': auc_score,
        'fpr': fpr,
        'tpr': tpr,
        'auc_agg': auc_agg,
        'fpr_agg': fpr_agg,
        'tpr_agg': tpr_agg,
        'auc_agg_filter': auc_agg_filter,
        'fpr_agg_filter': fpr_agg_filter,
        'tpr_agg_filter': tpr_agg_filter
    })

    auc_scores.append(auc_score)
    auc_agg_scores.append(auc_agg)
    auc_agg_filter_scores.append(auc_agg_filter)
    print(f"  Finished processing seed={seed_val}")

    seed_results_df = pd.DataFrame(seed_results_list)
    seed_results_path = os.path.join(seed_dir, "seed_results.pkl")
    seed_results_df.to_pickle(seed_results_path)
    print(f"Saved seed-level results at {seed_results_path}")


