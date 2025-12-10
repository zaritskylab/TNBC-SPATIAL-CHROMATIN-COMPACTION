
import os
import sys

from utils.data_func import *
import config.const as const
import config.params as params
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import seaborn as sns
import scipy.stats as stats



def process_probs_results(patch_df, prob_results_df, filter_flag=False):
    """
    Merges patch_df with hist_results on 'leap_ID' and 'patch_ID', filters out rows based on 'prob_results',
    and aggregates the data by 'nucleus_label' and 'leap_ID'.

    Parameters
    ----------
    patch_df : pd.DataFrame
        DataFrame containing patch information.
    hist_results : pd.DataFrame
        DataFrame containing histogram results with at least the columns 'leap_ID', 'patch_ID', and 'prob_results'.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame after merging, filtering, and grouping.
    """
    
    # Merge patch_df with hist_results on ['leap_ID', 'patch_ID']
    merged_df = pd.merge(
        patch_df, 
        prob_results_df[['leap_ID', 'patch_ID', 'prob_results']], 
        on=['leap_ID', 'patch_ID'], 
        how='left'
    )
    
    # Check for any NaN values in the merged DataFrame
    nan_rows = merged_df[merged_df.isna().any(axis=1)]
    print(f"Number of rows with NaN values: {len(nan_rows)}")
    # print("Max values in the merged DataFrame:")
    # print(merged_df.max())
    
    # Filter rows based on prob_results criteria
    if filter_flag:
        merged_df = merged_df[
            # (merged_df['prob_results'] <= 0.4) | (merged_df['prob_results'] >= 0.6)
            (merged_df['prob_results'] <= 0.45) | (merged_df['prob_results'] >= 0.55)

        ]
    
    # Group by 'nucleus_label' and 'leap_ID' and aggregate the data
    aggregated_df = merged_df.groupby(['nucleus_label', 'leap_ID']).agg({
        'lifetime_mean': 'first',    # Keep the first occurrence of 'lifetime_mean'
        'X coordinate': 'first',     # Keep the first occurrence of 'X coordinate'
        'Y coordinate': 'first',     # Keep the first occurrence of 'Y coordinate'
        'category': 'first',         # Keep the first occurrence of 'category'
        'prob_results': 'mean'       # Calculate the mean of 'prob_results'
    }).reset_index()
    
    print(f"Aggregated DataFrame shape: {aggregated_df.shape}")
    return aggregated_df


######################################################
######### lifetime distribution - patch_wise #########
######################################################
def get_all_seeds_results(base_dir):
    results = []
    
    # Loop over everything in the base directory
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # We only care about folders that start with 'seed_'
        if folder_name.startswith("seed_") and os.path.isdir(folder_path):
            csv_path = os.path.join(folder_path, "predictions.csv")
            
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                
                # df = df[(df['y_pred'] <= 0.45) | (df['y_pred'] >= 0.55)]
                df_agg = df.groupby('sample_id', as_index=False).agg({
                    'y_true': 'mean',
                    'y_pred': 'mean'
                })
                
                # Calculate AUC on the aggregated DataFrame
                auc_score = roc_auc_score(df_agg["y_true"], df_agg["y_pred"])
                
                # Extract the numeric part of the seed from folder name (e.g., "seed_35" -> 35)
                seed_num = int(folder_name.split("_")[1])
                
                # Store (seed number, AUC score)
                results.append((seed_num, auc_score))
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=["seed", "auc"])

    return results_df

def find_seed_closest_to_median_auc(base_dir):

    results_df = get_all_seeds_results(base_dir)

    if results_df.empty:
        print("No seed folders found with valid predictions.csv files.")
        return None

    median_auc = results_df["auc"].median()
    print(median_auc)
    
    # Compute absolute difference from median
    results_df["distance_from_median"] = (results_df["auc"] - median_auc).abs()
    
    # Find the row with the smallest distance to the median
    closest_idx = results_df["distance_from_median"].idxmin()
    closest_seed = results_df.loc[closest_idx, "seed"]
    closest_auc = results_df.loc[closest_idx, "auc"]
    
    print(f"Median AUC across seeds: {median_auc:.4f}")
    print(f"Seed closest to median AUC: {closest_seed} (AUC={closest_auc:.4f})")
    
    return closest_seed



def find_seed_best_auc(base_dir):
    
    # Convert results to a DataFrame
    results_df = get_all_seeds_results(base_dir)
    
    if results_df.empty:
        print("No seed folders found with valid predictions.csv files.")
        return None

    # Calculate the median AUC
    mean_auc = results_df["auc"].mean()
    print(f"mean_auc :{mean_auc}")
    best_auc = results_df["auc"].max()
    print(f"best_auc: {best_auc}")
    median_auc = results_df["auc"].median()
    print(f"median_auc: {median_auc}")
    
    best_idx = results_df["auc"].idxmax()

    # Extract the seed number from that row
    seed_best = results_df.loc[best_idx, "seed"]
    print("Seed corresponding to best AUC:", seed_best)
    
    return seed_best

def extract_hist_results(hist_base_dir, seed_num):

    # Construct the path to the predictions CSV for the given seed
    seed_results = os.path.join(hist_base_dir, f"seed_{seed_num}", "predictions.csv")
    
    # Read the CSV file
    hist_results = pd.read_csv(seed_results, dtype={'leap_ID': str})
    
    # Rename columns according to the provided mapping
    hist_results.rename(
        columns={
            "sample_id": "leap_ID", 
            "instance_id": "patch_ID", 
            "y_pred": "prob_results"
        }, 
        inplace=True
    )

    # Remove the first 4 characters from the 'leap_ID' column
    hist_results["leap_ID"] = hist_results["leap_ID"].str[4:]

    return hist_results


def process_hist_results(patch_df, hist_base_dir, seed_type='best'):
    """
    Process histogram results from a specified seed and merge with patch_df.
    
    The process includes:
      - Reading the predictions CSV from the seed folder.
      - Renaming columns: "sample_id" -> "leap_ID", "instance_id" -> "patch_ID", "y_pred" -> "prob_results".
      - Removing the first 4 characters from the 'leap_ID' column.
      - Merging with patch_df on ['leap_ID', 'patch_ID'].
      - Printing diagnostics (number of NaN rows and max values) if debug=True.
      - Filtering rows based on prob_results (<=0.4 or >=0.6).
      - Aggregating by 'nucleus_label' and 'leap_ID' (keeping the first occurrence for several columns and the mean for 'prob_results').
    
    Parameters
    ----------
    patch_df : pd.DataFrame
        DataFrame containing patch-level information to merge with histogram results.
    hist_base_dir : str
        Base directory for the histogram results (e.g., including overlap and other parameters).
    seed_num : int or str
    debug : bool, optional
        If True, print diagnostic information. Default is True.
    
    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame after processing the histogram results.
    """
    
    if seed_type=='median':
        seed_num = find_seed_closest_to_median_auc(hist_base_dir)
    if seed_type=='best':
        seed_num = find_seed_best_auc(hist_base_dir)

    hist_results = extract_hist_results(hist_base_dir, seed_num)
    
    hist_aggregated_df = process_probs_results(patch_df, hist_results)

    
    return hist_aggregated_df

######################################################
################## GNN - patch_wise ##################
######################################################

def extract_gnn_results(
    seed_num,
    patch_size,
    date_time,
    overlap=0.75,
    feature_type='lifetime',
    tissue_resolution='patch_tissue',
    max_dist=30,
    model_params="model_type_GAT_batch_size_16_lr_0.001"):
    
    # 1. Build the paths
    patch_params = f"size_{patch_size}_overlap_{overlap}"
    graph_file_dir = os.path.join(
        const.GNN_DIR, feature_type, tissue_resolution, f"max_distance_{max_dist}", "pytorch_geo"
    )
    model_results_pytorch_geo_dir = os.path.join(graph_file_dir, patch_params, "results", date_time)

    # Path to the results CSV for the specified seed
    seed_results = os.path.join(model_results_pytorch_geo_dir, model_params, str(seed_num), "k_fold_results.csv")

    # 2. Read the CSV
    gnn_results = pd.read_csv(seed_results, dtype={"leap_ID": str})

    # 3. Rename columns and trim the first four characters from patch_ID
    gnn_results.rename(columns={"sample_name": "patch_ID", "y_pred_probs": "prob_results"}, inplace=True)
    gnn_results["patch_ID"] = gnn_results["patch_ID"].str[4:]

    return gnn_results

def process_gnn_results(
    patch_df,
    seed_num,
    patch_size,
    date_time,
    filter_flag = True,
    overlap=0.75,
    feature_type='lifetime',
    tissue_resolution='patch_tissue',
    max_dist=30,
    model_params="model_type_GAT_batch_size_16_lr_0.001"):
    """
    Reads the GNN results for a given seed, merges them with the patch DataFrame,
    optionally filters based on probability, and then aggregates the results.

    Parameters
    ----------
    patch_df : pd.DataFrame
        DataFrame containing the patch information to merge with GNN results.
    seed_num : int or str
        Seed number used in the experiment (e.g., 35 for folder "seed_35").
    patch_size : int or float
        Size of the patch used when generating the GNN inputs.
    date_time : str or datetime-like
        Experiment date-time stamp used to locate the corresponding GNN results
        directory.
    filter_flag : bool, optional
        If True, apply probability-based filtering to the GNN predictions before
        aggregation. Defaults to True.
    overlap : float, optional
        Overlap between neighboring patches used when constructing the dataset.
        Defaults to 0.75.
    feature_type : str, optional
        The feature type used to construct the graph (e.g., "lifetime").
        Defaults to "lifetime".
    tissue_resolution : str, optional
        Resolution / level at which tissue information is aggregated
        (e.g., "patch_tissue"). Defaults to "patch_tissue".
    max_dist : int, optional
        Maximum distance parameter used when constructing the graph directory
        (e.g., for edge cutoff). Defaults to 30.
    model_params : str, optional
        Model parameter string identifying the model configuration subfolder,
        e.g. "model_type_GAT_batch_size_16_lr_0.001". Defaults to that value.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame after filtering and grouping by nucleus_label,
        leap_ID.
    """
    gnn_results = extract_gnn_results(seed_num, patch_size, date_time, overlap, feature_type, tissue_resolution, max_dist, model_params)
    # Merge with patch_df
    gnn_aggregated_df = process_probs_results(patch_df, gnn_results, filter_flag)

    return gnn_aggregated_df
