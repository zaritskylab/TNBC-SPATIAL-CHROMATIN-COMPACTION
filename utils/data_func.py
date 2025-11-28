import sys
import os

import tifffile as tiff
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def create_X_y(features_df, feature_name, remain_ID=False, patches=False):
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
    rcb_df = pd.read_excel(path_file, dtype={'leap_ID': str})
    
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
    nuclei_df = pd.read_csv(nuclei_path, dtype = {'leap_ID': str})
    rcb_df = pd.merge(nuclei_df, categories_df[['leap_ID', 'RCB Group', 'category']], on='leap_ID', how='inner')
    print(rcb_df.shape)
    filtered_df = rcb_df[rcb_df['lifetime_mean'] <= outliner_lifetime]
    print(f'Total amount of nuclei with lifetime more than {outliner_lifetime}: {rcb_df.shape[0] - filtered_df.shape[0]}')
    return filtered_df


def calculate_bin_proportions(group, min_bin, max_bin, interval, column):
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