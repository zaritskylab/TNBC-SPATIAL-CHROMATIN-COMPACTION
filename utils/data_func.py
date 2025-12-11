import sys
import os

import tifffile as tiff
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def create_X_y(features_df, feature_name, remain_ID=False, patches=False):
    """
    Prepare feature matrix X and target vector y from feature DataFrame.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing features and metadata.
    
    feature_name : tuple
        A tuple specifying the feature and its parameters:
            - feature_name[0] : str
                Name of the feature (e.g. 'lifetime_mean', 'eccentricity', 'area', 'median').
            - feature_name[1] : float or int
                Maximum value for binning (ignored for 'median').
            - feature_name[2] : float or int
                Bin width (ignored for 'median').

    remain_ID : bool, optional
        If True, retains 'leap_ID' and 'patch_ID' (if applicable) in X. Default is False.

    patches : bool, optional
        Whether to extract features at the patch level (vs. LEAP level). Default is False.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with selected features for classification.

    y : pd.Series
        Binary labels: 1 = responder, 0 = non-responder.

    feature_df_train : pd.DataFrame
        The full processed feature DataFrame including IDs and labels.
    """

    print(f"Start with feature {feature_name[0]}")

    max_range = feature_name[1]
    bins_range = feature_name[2]
    
    if feature_name[0] =='lifetime_mean':
        # 13, 0.585
        column=['lifetime_mean',max_range, bins_range]
        feature_df_train = extract_features_each_leap(features_df.copy() ,column, is_patches=patches)

    elif feature_name[0] =='eccentricity':
        # 1, 0.25
        column=['eccentricity', max_range, bins_range]
        feature_df_train = extract_features_each_leap(features_df.copy() ,column, is_patches=patches)

    elif feature_name[0] =='area':
        # 100, 25
        column=['area', max_range, bins_range]
        feature_df_train = extract_features_each_leap(features_df.copy() ,column, is_patches=patches)

    elif feature_name[0] =='median':
        column=['median']
        feature_df_train = extract_median_each_leap(features_df.copy() ,'lifetime_mean', is_patches=patches)
    
    
    if patches:
        id = ["leap_ID", "patch_ID"]
    else:
        id = ["leap_ID"]

    
    # Encode the 'categories' column as binary labels
    feature_df_train["categories"] = feature_df_train["categories"].map({"responder": 1, "non responder": 0})
    # Define the feature matrix X and the target vector y
    X = feature_df_train.drop(columns=["categories"])
    y = feature_df_train["categories"]

    if not remain_ID:
        X = X.drop(columns=id)

    print(f"Finish with feature {feature_name}")

    return X, y, feature_df_train

 
def extract_core_resection_from_tnbc(path_file, slide_num=False, for_prediction=True):
    """
    Load and filter TNBC cohort metadata for LEAP-based analysis.

    Parameters
    ----------
    path_file : str
        Path to the TNBC cohort metadata CSV file.

    slide_num : bool, optional
        If True, includes 'slide_num' in the returned DataFrame. Default is False.

    for_prediction : bool, optional
        If True, includes only samples from the clean cohort (`clean_cohort == 'v'`).
        If False, includes all samples with FLIM data. Default is True.

    Returns
    -------
    result_df : pd.DataFrame
        Cleaned cohort metadata with selected columns.

    leaps_list : np.ndarray
        Array of LEAP IDs from the filtered cohort.

    result_df_nan_rcb : pd.DataFrame
        Version of the result DataFrame before dropping rows with missing RCB group.
    """
    rcb_df = pd.read_csv(path_file, dtype={'leap_ID': str})
    
    # Extract the numerical part of leap_ID
    rcb_df['leap_ID'] = rcb_df['leap_id'].str.extract(r'(\d+)')
    
    # Filter rows where FLIM is 'v'
    if for_prediction:
        rcb_df = rcb_df[(rcb_df['FLIM_image'] == 'v') & (rcb_df['clean_cohort'] == 'v')]
    else:
        rcb_df = rcb_df[rcb_df['FLIM_image'] == 'v']
                        
    # Replace response categories
    rcb_df['category'] = rcb_df['response'].replace({'Responder': 'responder', 'pCR': 'responder', 'Non-Responder': 'non responder'})
    
    if slide_num:
        result_df = rcb_df[['leap_ID', 'slide_num', 'sample_type', 'category', 'RCB_group']]
    else:
    # Select and rename columns
        result_df = rcb_df[['leap_ID', 'sample_type', 'category', 'RCB_group']]
    leaps_list = result_df['leap_ID'].values
    result_df_nan_rcb = result_df.copy()
    result_df = result_df.dropna(subset=['RCB_group'])

    
    # Calculate statistics
    core_responder_count = len(result_df[(result_df['sample_type'] == 'core') & (result_df['category'] == 'responder')])
    core_non_responder_count = len(result_df[(result_df['sample_type'] == 'core') & (result_df['category'] == 'non responder')])
    resection_count = len(result_df[result_df['sample_type'] == 'resection'])
    
    # Print statistics
    print(f"Core Responder Count: {core_responder_count}")
    print(f"Core Non-Responder Count: {core_non_responder_count}")
    print(f"Resection Count: {resection_count}")
    
    return result_df, leaps_list, result_df_nan_rcb



def preprocess_df_rcb_exist(nuclei_path, categories_df, outliner_lifetime):
    """
    Merge features data with cohort metadata and filter by lifetime threshold.

    Parameters
    ----------
    nuclei_path : str
        Path to the CSV file containing features (must include 'leap_ID' and 'lifetime_mean').

    categories_df : pd.DataFrame
        DataFrame containing at least 'leap_ID', 'RCB Group', and 'category'.

    outliner_lifetime : float
        Threshold to remove nuclei with lifetime_mean greater than this value.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing merged features and cohort info,
        with outlier lifetimes removed.
    """
    nuclei_df = pd.read_csv(nuclei_path, dtype = {'leap_ID': str})
    rcb_df = pd.merge(nuclei_df, categories_df[['leap_ID', 'RCB Group', 'category']], on='leap_ID', how='inner')
    print(rcb_df.shape)
    filtered_df = rcb_df[rcb_df['lifetime_mean'] <= outliner_lifetime]
    print(f'Total amount of nuclei with lifetime more than {outliner_lifetime}: {rcb_df.shape[0] - filtered_df.shape[0]}')
    return filtered_df


def calculate_bin_proportions(group, min_bin, max_bin, interval, column):
    """
    Calculate the proportion of values within fixed-width bins for a given column.

    Parameters
    ----------
    group : pd.DataFrame
        DataFrame containing the values to be binned.

    min_bin : float
        Minimum value of the bin range.

    max_bin : float
        Maximum value of the bin range.

    interval : float
        Width of each bin.

    column : str
        Name of the column to bin.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with proportions of values in each bin.
        Column names follow the format: '<column>_<bin_start>-<bin_end>'.
    """
    # Define bins and bin labels
    bins = np.arange(min_bin, max_bin, interval)
    bin_labels = [f'{column}_{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
    
    # Calculate the proportion of nuclei within each bin
    group['bin'] = pd.cut(group[column], bins=bins, labels=bin_labels, include_lowest=True)
    bin_counts = group['bin'].value_counts(normalize=True).reindex(bin_labels, fill_value=0)
    
    # Convert the bin counts to a DataFrame
    bin_proportions_df = bin_counts.to_frame().T
    bin_proportions_df.reset_index(drop=True, inplace=True)
    
    return bin_proportions_df


def process_group_features(group_key, group, column, id):
    """
    Process a group of samples to compute binned feature proportions.

    Parameters
    ----------
    group_key : tuple
        Group identifier(s), e.g., (leap_ID,) or (leap_ID, patch_ID).

    group : pd.DataFrame
        Grouped DataFrame containing the feature values and category.

    column : list or tuple
        Feature specification:
            - column[0]: str, feature name
            - column[1]: float, upper limit for binning
            - column[2]: float, bin interval

    id : list
        List of column names to assign from group_key (e.g., ['leap_ID'] or ['leap_ID', 'patch_ID']).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with binned proportions and group/category info.
    """
    column_name = column[0]
    min_val = 0
    upper_limit = column[1]
    interval = column[2]
    
    max_val = upper_limit + interval
    bin_df_leap = calculate_bin_proportions(group, min_val, max_val, interval, column_name)

    for i, key in enumerate(id):
        bin_df_leap[key] = group_key[i]
    
    bin_df_leap['categories'] = group['category'].values[0]
    return bin_df_leap


def extract_features_each_leap(df, column, is_patches):
    """
    Extract binned feature distributions per LEAP or per patch.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with per-nucleus measurements, including 'leap_ID',
        the target feature column (e.g., 'lifetime_mean'), and optionally 'patch_ID'.

    column : list or tuple
        Feature specification:
            - column[0]: str, feature name (e.g., 'lifetime_mean')
            - column[1]: float, upper limit for binning
            - column[2]: float, bin interval

    is_patches : bool
        If True, groups by ['leap_ID', 'patch_ID']; otherwise by ['leap_ID'].

    Returns
    -------
    pd.DataFrame
        DataFrame with binned feature proportions for each group (LEAP or LEAP+patch).
    """
    # Rename column if misspelled
    if 'pathc_ID' in df.columns:
        df = df.rename(columns={'pathc_ID': 'patch_ID'})
        
    # Define ID columns based on whether patches are used
    id_columns = ['leap_ID', 'patch_ID'] if is_patches else ['leap_ID']
    
    # Group by the specified ID columns
    grouped = df.groupby(id_columns)
    group_list = [(key, group.copy()) for key, group in grouped]

    # Process groups in parallel
    all_bins_df = Parallel(n_jobs=-1)(
        delayed(process_group_features)(group_key, group, column, id_columns)
        for group_key, group in group_list
    )

    # Concatenate results into a single DataFrame
    final_df = pd.concat(all_bins_df, ignore_index=True)
    return final_df

def extract_median_each_leap(df, column, is_patches):
    """
    Compute the median of a given feature per LEAP or per patch.


    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least 'leap_ID', the target feature column,
        and 'category'. Optionally includes 'patch_ID' if `is_patches=True`.

    column : str
        Name of the feature column to compute the median on.

    is_patches : bool
        If True, groups by ['leap_ID', 'patch_ID']; otherwise by ['leap_ID'].

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per group and columns:
        - 'leap_ID' (and 'patch_ID' if applicable)
        - '<column>_median'
        - 'categories'
    """
    all_medians_df = []
    if is_patches:
        id = ['leap_ID', 'patch_ID']
    else:
        id = ['leap_ID']
    
    grouped = df.groupby(id)
    group_list = list(grouped)

    for group_key, group in group_list:
        df_patch = {}
        for i, key in enumerate(id):
            df_patch[key] = group_key[i]
            
        df_patch[f'{column}_median'] = group[column].median()
        
        df_patch['categories'] = group['category'].iloc[0]
        all_medians_df.append(df_patch)
        
    all_medians_df = pd.DataFrame(all_medians_df)
    return all_medians_df


def create_median_df(df, patches=False):
    """
    Prepare median-based feature matrix and labels for classification.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'leap_ID', 'lifetime_mean', and 'category'.
        If `patches=True`, must also include 'patch_ID'.

    patches : bool, optional
        If True, compute medians per (leap_ID, patch_ID). Otherwise, per leap_ID.
        Default is False.

    Returns
    -------
    X : pd.DataFrame
        DataFrame with median-based features (excluding labels).

    y : pd.Series
        Binary labels: 1 = responder, 0 = non responder.

    median_df : pd.DataFrame
        Full DataFrame including IDs, median feature, and original label.
    """
    median_df = extract_median_each_leap(df, 'lifetime_mean', is_patches=patches)
    final_df = median_df.copy()
    if patches:
        id = ["leap_ID", "patch_ID"]
    else:
        id = ["leap_ID"]

    final_df.drop(columns=id, inplace=True)

    # Encode the 'categories' column as binary labels
    final_df["categories"] = final_df["categories"].map({"responder": 1, "non responder": 0})
    # Define the feature matrix X and the target vector y
    X = final_df.drop(columns=["categories"])
    y = final_df["categories"]
    return X, y, median_df


def create_df(df,column, patches=False):
    """
    Prepare binned feature matrix and labels for classification.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with per-nucleus features, including 'leap_ID',
        'category', and the target feature column. If `patches=True`,
        must also include 'patch_ID'.

    column : list or tuple
        Feature binning specification:
            - column[0]: str, feature name (e.g., 'lifetime_mean')
            - column[1]: float, max bin value
            - column[2]: float, bin width

    patches : bool, optional
        If True, compute features per (leap_ID, patch_ID). Otherwise per leap_ID.
        Default is False.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with binned feature proportions (no labels).

    y : pd.Series
        Binary labels: 1 = responder, 0 = non responder.

    bins_df : pd.DataFrame
        Full DataFrame including IDs, binned features, and label.
    """
    bins_df = extract_features_each_leap(df, column, is_patches=patches)
    final_df = bins_df.copy()
    # Drop the 'leap_ID' column
    if patches:
        id = ["leap_ID", "patch_ID"]
    else:
        id = ["leap_ID"]

    final_df.drop(columns=id, inplace=True)
    

    # Encode the 'categories' column as binary labels
    final_df["categories"] = final_df["categories"].map({"responder": 1, "non responder": 0})
    # Define the feature matrix X and the target vector y
    X = final_df.drop(columns=["categories"])
    y = final_df["categories"]
    return X, y, bins_df


def calcultae_lifetime_correlation_to_feature(df, feature_name):
    """
    Compute per-LEAP Pearson correlation between lifetime_mean and another feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'leap_ID', 'lifetime_mean', and the specified feature.

    feature_name : str
        The name of the feature to correlate with 'lifetime_mean'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'feature_name'
        - 'leap_ID'
        - 'Correlation' (Pearson correlation coefficient)
    """
    correlations_tissue_wise = []
    for leap_id, group in df.groupby('leap_ID'):
        lifetime_mean = group['lifetime_mean']
        another_feature = group[feature_name]
        correlation= np.corrcoef(lifetime_mean, another_feature)[0, 1]

        correlations_tissue_wise.append({'feature_name': feature_name, 'leap_ID': leap_id, 'Correlation': correlation})

    # Convert results into a DataFrame
    correlation_tissue_df = pd.DataFrame(correlations_tissue_wise)
    print(f'Median correlation of {feature_name}: {np.median(correlation_tissue_df["Correlation"]):.3f}')
    return correlation_tissue_df