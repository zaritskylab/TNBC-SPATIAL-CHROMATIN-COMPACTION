"""
FLIM Segmentation Feature Extraction & Quality Control Pipeline

This script provides a modular framework for:
- Extracting single-cell features from segmentation masks and FLIM lifetime images.
- Filtering out segmentation outliers.
- Computing spatial statistics (e.g., MAE by radius).
- Patching images and recomputing features per patch.
- Creating final feature tables for full tissue and patch-level analysis.
"""

import os

from utils.data_func import*
import config.const as const
import config.params as params

import pandas as pd
from tifffile import tifffile 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from skimage.measure import regionprops
import argparse


# --------------------------------------
# File Management and Merging Utilities
# --------------------------------------

def concatenate_all_files(files_dir, file_signature, delete_files=True, save_file_path=False):
    """
    Concatenate all CSV files in a directory matching a given suffix.

    Parameters
    ----------
    files_dir : str
        Path to the directory containing the feature CSV files.
    file_signature : str
        Common suffix of feature files to match (e.g., "single_cell_features.csv").
    delete_files : bool, optional
        If True, deletes the original files after merging. Default is True.
    save_file_path : str or bool, optional
        If provided, saves the merged DataFrame to this path. Default is False.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing all rows from the matched files.
    """
    # Collect all feature file paths
    feature_files = [
        os.path.join(files_dir, f)
        for f in os.listdir(files_dir)
        if f.endswith(file_signature)
    ]
    # Combine all CSV files into one DataFrame
    all_dataframes = []
    for file_path in feature_files:
        df = pd.read_csv(file_path, dtype={'leap_ID': str})
        all_dataframes.append(df)

    combined_df = pd.concat(all_dataframes, ignore_index=True) if all_dataframes else pd.DataFrame()

    if save_file_path:
        combined_df.to_csv(save_file_path, index=False)
    if delete_files:
        for file_path in feature_files:
            os.remove(file_path)

    return combined_df
    

def add_label_to_df(df_category, concatenated_df):
    """
    Merge category labels with feature data and filter out high lifetime values.

    Parameters
    ----------
    df_category : pd.DataFrame
        DataFrame with LEAP ID and category labels.
    concatenated_df : pd.DataFrame
        Feature DataFrame to annotate.

    Returns
    -------
    pd.DataFrame
        Annotated and filtered feature DataFrame.
    """

    merged_df = pd.merge(concatenated_df, df_category[['leap_ID', 'category']], on='leap_ID', how='inner')
    merged_df = merged_df[merged_df['lifetime_mean'] <= 13]
    merged_df = merged_df.drop_duplicates()
    
    return merged_df

# -----------------------------------------------------------
# Feature Extraction from RegionProps and density calculation
# -----------------------------------------------------------

def extract_features_regionprops(seg_image, lifetime_image, morphology=True):
    """
    Extract region-based geometric and intensity features from a segmentation mask.

    Parameters
    ----------
    seg_image : np.ndarray
        Labeled segmentation mask.
    lifetime_image : np.ndarray
        fluorescent lifetime image.
    morphology : bool, optional
        If True, extracts geometric features in addition to lifetime_value(intensity). Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with extracted features per segmented region.
    """
    properties = regionprops(seg_image, intensity_image=lifetime_image)
    
    region_data = []
    if morphology:
        for prop in properties:
            region_data.append({
                "nucleus_label": prop.label,  # Label ID
                "X coordinate": prop.centroid[1],  # x-coordinate of centroid
                "Y coordinate": prop.centroid[0],  # y-coordinate of centroid
                "lifetime_mean": prop.mean_intensity/1000,
                "area": prop.area,  # Area of the region
                "extent": prop.extent,  # Extent (area / bounding box area)
                "solidity": prop.solidity,  # Solidity (area / convex area)
                "perimeter": prop.perimeter,  # Perimeter
                "diameter_area": np.sqrt(4 * prop.area / np.pi),  # Diameter equivalent to the area
                "convex_hull_area": prop.convex_area,  # Convex hull area
                "minor_axis_length": prop.minor_axis_length,  # Length of the minor axis
                "perimeter_crofton": prop.perimeter_crofton,  # Crofton perimeter
                "major_axis_length": prop.major_axis_length,  # Length of the major axis
                "orientation": prop.orientation,  # Orientation of the region
                "diameter_max": prop.feret_diameter_max,  # Maximum Feret diameter
                "eccentricity": prop.eccentricity,  # Eccentricity of the ellipse
            })

    else: 
        for prop in properties:
            region_data.append({
                "nucleus_label": prop.label,  # Label ID
                "lifetime_mean": prop.mean_intensity/1000,
            })

    region_df = pd.DataFrame(region_data)

    return region_df

def calculate_density(group, radius=30):
    """
    Calculate local density: number of neighbors within radius
    divided by area of the circle with that radius.

    Parameters
    ----------
    group : pd.DataFrame
        Data with 'X coordinate' and 'Y coordinate'
    radius : float
        Radius for counting neighbors

    Returns
    -------
    pd.Series
        Density values per point
    """
    coords = group[['X coordinate', 'Y coordinate']].astype('float64').values

    # Fit NearestNeighbors
    nbrs = NearestNeighbors(radius=radius)
    nbrs.fit(coords)
    _, indices = nbrs.radius_neighbors(coords)

    # Count neighbors per point (excluding self)
    neighbor_counts = [len(i) - 1 for i in indices]  # exclude self

    # Area of circle
    area = np.pi * radius**2

    # Local density = neighbors per unit area
    density = [count / area for count in neighbor_counts]
    return pd.Series(density, index=group.index)

# --------------------------------------
# Neighborhood Similarity Features (MAE)
# --------------------------------------

def calculate_neighbors_features(df, radius):
    """
    Find all neighbors within a specified radius using 2D coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'X coordinate' and 'Y coordinate' columns.
    radius : float
        Distance threshold for considering neighbors.

    Returns
    -------
    list of np.ndarray
        Indices of neighbors for each point.
    """
    # Initialize the NearestNeighbors object
    nbrs = NearestNeighbors(radius=radius, algorithm='auto')
    
    # Fit the NearestNeighbors object on the X and Y coordinates
    nbrs.fit(df[['X coordinate', 'Y coordinate']])
    
    # Get the indices and distances of neighbors within the radius for each point
    distances, indices = nbrs.radius_neighbors(df[['X coordinate', 'Y coordinate']], radius)
    return indices


def calculate_mae(df, indices):
    """
    Calculate mean absolute error of lifetime values between each point and its neighbors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'lifetime_mean' column.
    indices : list of list of int
        Indices of neighbors for each data point.

    Returns
    -------
    list of float
        Mean absolute error for each point.
    """
    mae_lifetime = []
    for i, neighbors in enumerate(indices):
        neighbors_without_i = [n for n in neighbors if n != i]
        mae = np.mean(np.abs(df.iloc[neighbors_without_i]['lifetime_mean'] - df.iloc[i]['lifetime_mean']))
        mae_lifetime.append(mae)
    return mae_lifetime


def calculate_all_neighbors(group, radius_list):
    """
    Compute neighbor indices for multiple radius sizes.

    Parameters
    ----------
    group : pd.DataFrame
        Grouped data (typically per LEAP ID).
    radius_list : list of float
        List of radius values.

    Returns
    -------
    dict
        Dictionary mapping each radius to its neighbor indices.
    """
    indices_dict = {radius_size: calculate_neighbors_features(group, radius_size) for radius_size in radius_list}
    return indices_dict


def filter_neighbors(indices_dict, radius_list):
    """
    Generate non-overlapping neighbor rings between successive radius levels.

    Parameters
    ----------
    indices_dict : dict
        Dictionary mapping radius sizes to neighbor index arrays.
    radius_list : list
        Ordered list of radius sizes.

    Returns
    -------
    list of tuple
        Each tuple contains (radius, filtered neighbor indices).
    """
    indices_filter_all_radius = []
    indices_radius_smaller = indices_dict[radius_list[0]]
    indices_filter_all_radius.append((radius_list[0], indices_radius_smaller))
    
    for radius_size in radius_list[1:]:
        indices_radius_bigger = indices_dict[radius_size]
        indices_filter = []
        for nodes_radius_bigger, nodes_radius_smaller in zip(indices_radius_bigger, indices_radius_smaller):
            indices_diff = [i for i in nodes_radius_bigger if i not in nodes_radius_smaller]
            indices_filter.append(indices_diff)
        indices_filter_all_radius.append((radius_size, indices_filter))
        indices_radius_smaller = indices_radius_bigger
        
    return indices_filter_all_radius

def calculate_all_mae(group, indices_filter_all_radius):
    """
    Compute MAE for each ring-defined neighbor set.

    Parameters
    ----------
    group : pd.DataFrame
        Data containing 'lifetime_mean'.
    indices_filter_all_radius : list of tuple
        Radius and corresponding neighbor indices.

    Returns
    -------
    dict
        Dictionary mapping radius to a list of MAE values.
    """
    mae_list = {}
    for radius, indices in indices_filter_all_radius:
        mae_list[f"mae_radius_{radius}"] = calculate_mae(group, indices)
    return mae_list


# Define the function to process each group
def process_group_mae(leap_number, group, radius_list):
    """
    Wrapper function to compute ring-wise MAE features for a LEAP group.

    Parameters
    ----------
    leap_number : str
        The LEAP identifier.
    group : pd.DataFrame
        DataFrame for a single LEAP sample.
    radius_list : list of float
        Radius values for computing MAE.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with new MAE feature columns.
    """
    print(f"start with {leap_number}")
    # Calculate neighbors for all radius sizes in parallel
    indices_dict = calculate_all_neighbors(group, radius_list)
    
    # Filter neighbors and create radius rings
    indices_filter_all_radius = filter_neighbors(indices_dict, radius_list)
    
    # Calculate MAE for each radius ring
    mae_dict = calculate_all_mae(group, indices_filter_all_radius)

    # Add MAE values as new columns to the group DataFrame
    for radius_col, mae_values in mae_dict.items():
        group[radius_col] = mae_values

    features_columns = [col for col in group.columns if col != 'category']
    group = group[features_columns + ['category']]
    print(f"Finish processing leap_ID {leap_number}")
    return group

    
# --------------------------------------
# Whole Image Feature Extraction
# --------------------------------------

def process_leap(leap_number, sample_type, morphology=True):
    """
    Process a single LEAP full tissue image to extract features.

    Parameters
    ----------
    leap_number : str
        Identifier for the LEAP sample.
    sample_type : str
        Type of sample (e.g., 'core' or 'resection').
    morphology : bool, optional
        Whether to extract morphological features. Default is True.

    Returns
    -------
    None
        Saves the extracted feature CSV to the full tissue directory.
    """
    try:
        print(f"Start with leap {leap_number}")

        flim_image_path = os.path.join(const.flim_dir, f'LEAP{leap_number}_flim.tif')
        seg_image_path = os.path.join(const.seg_after_qc_dir, f'LEAP{leap_number}_segmentation_labels_qc.tif')
        
        segmentation_image = tifffile.imread(seg_image_path)
        flim_image = tifffile.imread(flim_image_path)

        props_df = extract_features_regionprops(segmentation_image, flim_image, morphology=morphology)
        props_df['leap_ID'] = leap_number.zfill(3)
        props_df = props_df.reindex(columns=['leap_ID'] + [col for col in props_df.columns if col != 'leap_ID'])

        props_df["density_radius_20"] = calculate_density(props_df, radius=20)
        props_df["density_radius_40"] = calculate_density(props_df, radius=40)
        props_df["density_radius_60"] = calculate_density(props_df, radius=60)
        props_df["density_radius_80"] = calculate_density(props_df, radius=80)

        # Save DataFrame as CSV
        file_name = f"LEAP{leap_number}_single_cell_features.csv"

        save_dir = os.path.join(const.full_tissue_dir, sample_type)
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        save_path = os.path.join(save_dir, file_name)
        props_df.to_csv(save_path, index=False)

        print(f"DataFrame for leap {leap_number} saved as {file_name}")

    except Exception as e:
        print(f"Error processing LEAP {leap_number}: {e}")

def process_all_features_full_tissue_df(df_category, sample_type, delete_concat_files=True, with_mae=True):
    """
    Concatenate features for all LEAPs and compute MAE metrics for full tissue samples.

    Parameters
    ----------
    df_category : pd.DataFrame
        DataFrame containing sample metadata and LEAP IDs.
    sample_type : str
        Either 'core' or 'resection'.
    delete_concat_files : bool, optional
        Whether to delete original CSVs after concatenation. Default is True.
    with_mae : bool, optional
        Whether to compute spatial MAE features. Default is True.

    Returns
    -------
    None
        Saves full tissue feature files (with and without MAE).
    """
    print(f"Start concatenation for full tissue image")
    
    sample_type_directory = os.path.join(const.full_tissue_dir, sample_type)
    combined_df = concatenate_all_files(sample_type_directory, file_signature="single_cell_features.csv", delete_files=delete_concat_files)

    combined_df_with_label = add_label_to_df(df_category, combined_df)
    full_tissue_feature_file = os.path.join(sample_type_directory, f"FLIM_features_full_tissue.csv")
    combined_df_with_label.to_csv(full_tissue_feature_file, index=False)

    if with_mae:
        print(f"Start calculate mae features for full tissue")

        # Group the data by `leap_ID`
        grouped = combined_df_with_label.groupby('leap_ID')
        radius_list = [20, 40, 60, 80, 100, 120]

        processed_groups_mae = Parallel(n_jobs=10, timeout=20000)(
            delayed(process_group_mae)(leap_number, group, radius_list)
            for leap_number, group in grouped
        )
        print(f"Finish calculate mae features for full tissue")
        
        combined_df_with_mae_features = pd.concat(processed_groups_mae)
        
        combined_df_with_mae_features = combined_df_with_mae_features.fillna(0)

        full_tissue_feature_file_with_mae = os.path.join(sample_type_directory, f"FLIM_features_full_tissue_with_mae.csv")
        combined_df_with_mae_features.to_csv(full_tissue_feature_file_with_mae, index=False)
        print(f"Saved file: {full_tissue_feature_file_with_mae}")

        # if os.path.exists(full_tissue_feature_file):
        #     os.remove(full_tissue_feature_file)
        #     print(f"Deleted file: {full_tissue_feature_file}")

    print(f"Finish concatenation for full tissue")

def create_all_feature_core_full_tissue_df(sample_type='core', with_mae=True):
    """
    Create feature tables for all core tissue samples.

    Parameters
    ----------
    sample_type : str, optional
        Sample type, default is 'core'.
    with_mae : bool, optional
        Whether to compute spatial MAE features. Default is True.

    Returns
    -------
    None
    """
    df_category_with_rcb, _, _ = extract_core_resection_from_tnbc(const.rcb_file)
    df_category_with_rcb.head()

    core_leaps_list = df_category_with_rcb[df_category_with_rcb['sample_type'] == sample_type]['leap_ID'].to_list()
    _ = Parallel(n_jobs=10, timeout=10000)(delayed(process_leap)(leap_number, sample_type=sample_type) for leap_number in core_leaps_list)
    process_all_features_full_tissue_df(df_category_with_rcb, sample_type=sample_type, with_mae=with_mae)


def create_all_feature_resection_full_tissue_df(sample_type='resection'):
    """
    Create feature tables for all resection tissue samples.

    Parameters
    ----------
    sample_type : str, optional
        Sample type, default is 'resection'.

    Returns
    -------
    None
    """
    df_category_with_rcb, _, _ = extract_core_resection_from_tnbc(const.rcb_file)
    df_category_with_rcb.head()

    resection_leaps_list = df_category_with_rcb[df_category_with_rcb['sample_type'] == sample_type]['leap_ID'].to_list()
    #remove leap_074
    resection_leaps_list = resection_leaps_list[:11] + resection_leaps_list[12:]
    print(resection_leaps_list)
    _ = Parallel(n_jobs=-1)(delayed(process_leap)(leap_number, sample_type=sample_type) for leap_number in resection_leaps_list)
    process_all_features_full_tissue_df(df_category_with_rcb, sample_type=sample_type, with_mae=False)


# --------------------------------------
# Patch Image Feature Extraction
# --------------------------------------

def create_patches(leap_id, segmentation_image, patch_size=1000, overlap_percentage=0):
    """
    Generate patch metadata (coordinates and IDs) from a segmentation image.

    Parameters
    ----------
    leap_id : str
        LEAP sample identifier.
    segmentation_image : np.ndarray
        2D array of labeled segmentation.
    patch_size : int, optional
        Size of each patch (square). Default is 1000.
    overlap_percentage : float, optional
        Overlap fraction between patches. Default is 0.

    Returns
    -------
    list of dict
        Metadata for each patch.
    """
    patches_info = []
    h, w = segmentation_image.shape

    left, right, top, bottom = find_nonzero_borders(segmentation_image)
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)

    step_size = int(patch_size * (1 - overlap_percentage))

    for y in range(top, bottom, step_size):
        for x in range(left, right, step_size):
            patch = segmentation_image[y:y + patch_size, x:x + patch_size]
            non_zero_pixel = np.count_nonzero(patch)
            condition = non_zero_pixel > 40
            
            if condition:
                left_up_x, left_up_y = x, y
                right_down_x, right_down_y = x + patch_size, y + patch_size
                patches_info.append({
                    'patch_ID': f"{left_up_x}_{left_up_y}_{right_down_x}_{right_down_y}",
                    'leap_ID': leap_id,
                    'left_up_x': left_up_x,
                    'left_up_y': left_up_y,
                    'right_down_x': right_down_x,
                    'right_down_y': right_down_y,
                })
    return patches_info


def find_nonzero_borders(image):
    """
    Find bounding box borders of non-zero content in the image.

    Parameters
    ----------
    image : np.ndarray
        Binary or labeled segmentation mask.

    Returns
    -------
    tuple of int
        (left, right, top, bottom) indices.
    """
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    top, bottom = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    left, right = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    return left, right, top, bottom

def plot_borders(segmentation_image, left, right, top, bottom):
    """
    Plot segmentation image with its non-zero borders.

    Parameters
    ----------
    segmentation_image : np.ndarray
        Labeled segmentation image.
    left, right, top, bottom : int
        Border coordinates.

    Returns
    -------
    None
    """
    plt.imshow(segmentation_image, cmap='gray')
    plt.axhline(y=top, color='r', linestyle='-')
    plt.axhline(y=bottom, color='r', linestyle='-')
    plt.axvline(x=left, color='r', linestyle='-')
    plt.axvline(x=right, color='r', linestyle='-')
    plt.title('Segmentation Image with Non-zero Borders')
    plt.show()

def process_patches_for_leap(leap_number, patch_size, overlap=0, morphology=False):
    """
    Process patch-based feature extraction for one LEAP sample.

    Parameters
    ----------
    leap_number : str
        LEAP sample ID.
    patch_size : int
        Patch size in pixels.
    overlap : float, optional
        Overlap percentage between patches. Default is 0.
    morphology : bool, optional
        Whether to extract geometric features. Default is False.

    Returns
    -------
    None
    """
    print(f"start with leap number {leap_number}")

    flim_image_path = os.path.join(const.flim_dir, f'LEAP{leap_number}_flim.tif')
    seg_image_path = os.path.join(const.seg_after_qc_dir, f'LEAP{leap_number}_segmentation_labels_qc.tif')
    
    segmentation_image = tifffile.imread(seg_image_path)
    flim_image = tifffile.imread(flim_image_path)


    patches_dir = os.path.join(const.patch_dir, f"size_{patch_size}_overlap_{overlap}")

    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)

    patch_info_file_name = f"LEAP{leap_number}_patch_info.csv"
    patch_info_save_path = os.path.join(patches_dir, patch_info_file_name)
    # if os.path.exists(patch_info_save_path):
    #     print(f"Leap number {leap_number} File {patch_info_file_name} already exists at {patches_dir}.")
    #     return
    # else:
    patches_info = create_patches(leap_number, segmentation_image, patch_size, overlap)
    patch_info_df = pd.DataFrame(patches_info)
    patch_info_df.to_csv(patch_info_save_path, index=False)
    

    for patch_info in patches_info:
        left_up_x = patch_info['left_up_x']
        left_up_y = patch_info['left_up_y']
        right_down_x = patch_info['right_down_x']
        right_down_y = patch_info['right_down_y']
        patch_ID = patch_info['patch_ID']

        seg_patch = segmentation_image[left_up_y:right_down_y, left_up_x:right_down_x]
        flim_patch = flim_image[left_up_y:right_down_y, left_up_x:right_down_x]


        props_df = extract_features_regionprops(seg_patch, flim_patch, morphology=morphology)

        props_df['leap_ID'] = leap_number.zfill(3)
        props_df['patch_ID'] = patch_ID
        props_df = props_df.reindex(
            columns=['leap_ID', 'patch_ID'] + [col for col in props_df.columns if col not in ['leap_ID', 'patch_ID']])
                
        file_name = f"LEAP{leap_number}_{patch_ID}_single_cell_features.csv"
        save_path = os.path.join(patches_dir, file_name)
        props_df.to_csv(save_path, index=False)

        print(f"DataFrame for patch {patch_ID} saved as {file_name}")


    print(f"Patch info for leap {leap_number} saved as {patch_info_file_name}")        
    print(f"finish with leap number {leap_number}")


def get_filtered_patch_df(patch_df, filter_threshold=50):
    """
    Filters out rows from the patch_df where the count of 'leap_ID' and 'patch_ID' 
    combinations is less than filter_threshold.

    Parameters:
        patch_df (pd.DataFrame): Input DataFrame containing at least 'leap_ID' and 'patch_ID' columns.

    Returns:
        pd.DataFrame: Filtered DataFrame with rows removed based on the criteria.
    """
    # Group by 'leap_ID' and 'patch_ID' and count occurrences
    counts = patch_df.groupby(['leap_ID', 'patch_ID']).size().reset_index(name='count')

    # Filter for rows where the count is less than filter_threshold
    low_count_rows = counts[counts['count'] < filter_threshold]

    # Get the 'leap_ID' and 'patch_ID' combinations to be removed
    to_remove = low_count_rows[['leap_ID', 'patch_ID']]

    # Merge with the original DataFrame to get a mask of rows to be kept
    merged_df = patch_df.merge(to_remove, on=['leap_ID', 'patch_ID'], how='left', indicator=True)

    # Keep only rows that do not match the 'leap_ID' and 'patch_ID' combinations to be removed
    filtered_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    return filtered_df


def process_all_feature_patches_df(patch_size, overlap, df_category, full_tissue_df):
    """
    Aggregate patch-level features and merge with whole image coordinates.

    Parameters
    ----------
    patch_size : int
        Patch size in pixels.
    overlap : float
        Overlap percentage.
    df_category : pd.DataFrame
        Sample category metadata.
    full_tissue_df : pd.DataFrame
        Full image features (to reference coordinates).

    Returns
    -------
    None
    """
    print(f"Start concatenation for patch size {patch_size}, overlap {overlap}")
    
    patches_dir = os.path.join(const.patch_dir, f"size_{patch_size}_overlap_{overlap}")
    combined_df = concatenate_all_files(patches_dir, file_signature="single_cell_features.csv")

    combined_df_with_label = add_label_to_df(df_category, combined_df)

    ##### Update nuclei X Y coordinates based on the whole image
    patches_updated_coords_df = combined_df_with_label.merge(
    full_tissue_df[['leap_ID', 'nucleus_label', 'X coordinate', 'Y coordinate']],
    on=['leap_ID', 'nucleus_label'],
    how='left')

    patches_feature_file = os.path.join(patches_dir, f"FLIM_features_patches_size_{patch_size}_overlap_{overlap}.csv")
    patches_updated_coords_df.to_csv(patches_feature_file, index=False)
    print(f"Combined DataFrame saved as {patches_feature_file}")

    patches_info_save_file = os.path.join(patches_dir, "patches_info.csv")
    _ = concatenate_all_files(patches_dir, file_signature="patch_info.csv", save_file_path=patches_info_save_file)
    print(f"Patch information DataFrame saved as {patches_feature_file}")

    # === Filtering step ===
    print("Filtering patch-level DataFrame based on nucleus count...")
    filtered_df = get_filtered_patch_df(patches_updated_coords_df)
    filtered_file_path = patches_feature_file.replace(".csv", "_after_filter.csv")
    filtered_df.to_csv(filtered_file_path, index=False)
    print(f"Filtered DataFrame saved to: {filtered_file_path}")


    print(f"Finish concatenation for patch size {patch_size}, overlap {overlap}")


def create_all_feature_patches_df(patch_size, overlap):
    """
    Run patch-based feature extraction for all core samples.

    Parameters
    ----------
    patch_size : int
        Patch size in pixels.
    overlap : float
        Overlap fraction.

    Returns
    -------
    None
    """
    df_category_with_rcb, _, _ = extract_core_resection_from_tnbc(const.rcb_file)
    df_category_with_rcb.head()

    core_leaps_list = df_category_with_rcb[df_category_with_rcb['sample_type'] == 'core']['leap_ID'].to_list()
    
    print(f"start extract features for patch size {patch_size}, overlap {overlap}")

    try:
        _ = Parallel(n_jobs=-1, timeout=10000)(delayed(process_patches_for_leap)(leap_number, patch_size=patch_size, overlap=overlap) for leap_number in core_leaps_list)
        # df_file_path = os.path.join(const.full_tissue_dir, 'core', "FLIM_features_full_tissue_with_mae.csv")
        df_file_path = os.path.join(const.full_tissue_dir, 'core', "FLIM_features_full_tissue.csv")


        full_tissue_features_df = pd.read_csv(df_file_path, dtype = {'leap_ID': str})
        process_all_feature_patches_df(patch_size, overlap, df_category_with_rcb, full_tissue_features_df)

        print(f"finish extract features for patch size {patch_size}, overlap {overlap}")
    
    except Exception as e:
        print(f"Error during patch-based feature extraction: {e}")


# --------------------------------------
#  Feature Transformation
# --------------------------------------

def build_lifetime_distribution_full_tissue(sample_type: str, max_range: float, bin_range: float, feature_file_name: str):
    """
    Build and save the lifetime distribution data based on FLIM features.

    Parameters
    ----------
    sample_type : str
        Type of the tissue sample (e.g., 'core', 'resection').
    max_range : float
        Maximum value for the binning range.
    bin_range : float
        Width of each bin.
    Returns
    -------
    pd.DataFrame
        Lifetime distribution (binned) for full tissue.
    """

    print("Starting lifetime distribution processing...")
    print(f"Sample type         : {sample_type}")
    print(f"Maximum value       : {max_range}")
    print(f"Bin range (width)   : {bin_range}")
    print()

    # Load data
    df_file_path = os.path.join(const.full_tissue_dir, sample_type, feature_file_name)
    print(f"Reading FLIM feature file from: {df_file_path}")
    full_tissue_features_df = pd.read_csv(df_file_path, dtype={'leap_ID': str})
    print("Data loaded successfully.\n")

    # Build distribution
    print("Building lifetime distribution...")
    distribution_params = ['lifetime_mean', max_range, bin_range]
    print(f"Distribution parameters: {distribution_params}")
    _, _, df_bins = create_X_y(full_tissue_features_df, distribution_params)
    bin_amount = len(df_bins.columns) - 2
    print(f"Distribution built. Total bins created: {bin_amount}\n")

    # Save output
    lifetime_distribution_full_tissue = os.path.join(
        const.full_tissue_dir, sample_type,
       f"features_lifetime_distribution_data_max_val_{max_range}_bins_amount_{bin_amount}_bin_range_{bin_range}.csv"
    )

    print(f"Saving bin distribution to: {lifetime_distribution_full_tissue}")
    df_bins.to_csv(lifetime_distribution_full_tissue, index=False)
    print("Bin distribution CSV saved successfully.\n")

    return df_bins

def build_distribution(features_df, feature_name, bins_amount, sample_type):
    """
    Build and save a binned distribution of a selected FLIM feature.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing FLIM feature values.
    feature_name : str
        Name of the feature column to be binned (e.g., 'lifetime_mean').
    bins_amount : int
        Number of bins to divide the feature range into.

    Returns
    -------
    pd.DataFrame
        Binned distribution of the selected feature.
    """

    print(f"Building {feature_name} distribution...")

    # Compute binning parameters
    max_val = features_df[feature_name].max()
    bin_range = max_val / bins_amount

    print(f"{feature_name} histogram will use {bins_amount} bins of size {bin_range} ns, up to {max_val} ns.")
    distribution_params = [feature_name, max_val, bin_range]
    print(f"Distribution parameters: {distribution_params}")

    # Generate binned distribution using create_X_y
    _, _, df_bins = create_X_y(features_df, distribution_params)
    bin_amount = len(df_bins.columns) - 2
    print(f"Distribution built. Total bins created: {bin_amount}\n")

    # Save distribution CSV
    distribution_full_tissue = os.path.join(
        const.full_tissue_dir, sample_type,
        f"features_{feature_name}_distribution_data_max_val_{max_val}_bins_amount_{bin_amount}_bin_range_{bin_range:.3f}.csv"
    )

    print(f"Saving bin distribution to: {distribution_full_tissue}")
    df_bins.to_csv(distribution_full_tissue, index=False)
    print("Bin distribution CSV saved successfully.\n")

    return df_bins


def build_lifetime_distribution_patch(patch_size: int, patch_overlap: float, max_range: float=13, bin_range: float=0.73) -> pd.DataFrame:
    """
    Load filtered patch-level FLIM data and build lifetime distribution.

    Parameters
    ----------
    patch_size : int
        Size (in pixels) of each patch.
    patch_overlap : float
        Overlap percentage between patches.

    Returns
    -------
    pd.DataFrame
        Lifetime distribution (binned) for patches.
    """
    specific_patch_dir = os.path.join(const.patch_dir, f"size_{patch_size}_overlap_{patch_overlap}")
    
    # Construct filtered file path
    filtered_file_name = f"FLIM_features_patches_size_{patch_size}_overlap_{patch_overlap}_after_filter.csv"
    filtered_file_path = os.path.join(specific_patch_dir, filtered_file_name)

    print(f"Reading filtered patch data from: {filtered_file_path}")
    filtered_patch_df = pd.read_csv(filtered_file_path, dtype={'leap_ID': str})
    print("Filtered data loaded.\n")

    # Build lifetime distribution
    print("Building patch-level lifetime distribution...")
    distribution_params = ['lifetime_mean', max_range, bin_range]
    print(f"Distribution parameters: {distribution_params}")
    _, _, df_bins_patches = create_X_y(filtered_patch_df, distribution_params, patches=True)
    bin_amount = len(df_bins_patches.columns) - 3
    print(f"Distribution built. Total bins created: {bin_amount}\n")

    patch_distribution_file_name = f"features_lifetime_distribution_data_patches_size_{patch_size}_overlap_{patch_overlap}_max_val_{max_range}_bins_amount_{bin_amount}_bin_range_{bin_range}.csv"
    patch_distribution_path = os.path.join(specific_patch_dir, patch_distribution_file_name)

    df_bins_patches.to_csv(patch_distribution_path, index=False)
    print(f"Lifetime distribution saved to: {patch_distribution_path}")

    return df_bins_patches


def aggregate_median_features_by_leap(output_dir: str, sample_type: str = 'core') -> pd.DataFrame:
    """
    Load FLIM full-tissue feature data, aggregate median values per LEAP sample,
    and save the result to a CSV file.

    Parameters
    ----------
    output_dir : str
        Base directory where data is stored and output will be saved.
    sample_type : str, optional
        Sample type folder name (e.g., 'core'), default is 'core'.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with median features per LEAP sample.
    """
    # Load the full-tissue feature CSV
    # df_file_path = os.path.join(output_dir, sample_type, "FLIM_features_full_tissue_with_mae.csv")
    df_file_path = os.path.join(output_dir, sample_type, "FLIM_features_full_tissue.csv")

    print(f"Reading FLIM features from: {df_file_path}")
    full_tissue_features_df = pd.read_csv(df_file_path, dtype={'leap_ID': str})
    print("Data loaded successfully.")

    # Drop unwanted columns
    df = full_tissue_features_df.drop(columns=['nucleus_label', 'X coordinate', 'Y coordinate'], errors='ignore')
    print(f"Dropped unwanted columns. Remaining columns: {df.columns.tolist()}")

    # Define aggregation: median for features, 'first' for category
    agg_funcs = {col: 'median' for col in df.columns if col not in ['leap_ID', 'category']}
    agg_funcs['category'] = 'first'

    # Group by 'leap_ID' and aggregate
    median_df = df.groupby('leap_ID', as_index=False).agg(agg_funcs)
    print(f"Aggregation complete. Resulting shape: {median_df.shape}")

    # Save to CSV
    df_median_file_path = os.path.join(output_dir, sample_type, f"features_median_data.csv")
    print(f"Saving aggregated DataFrame to: {df_median_file_path}")
    median_df.to_csv(df_median_file_path, index=False)
    print("File saved successfully.")

    return median_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract features FLIM images.')

    # Positional with default (optional via nargs='?')
    parser.add_argument('sample_type', choices=['core', 'resection', 'patch'], nargs='?', default='core',
        help="Type of sample to process. Choices: 'core', 'resection', 'patch'. Default: 'core'."
    )

    parser.add_argument('--patch-size', type=int, default=1500,
        help="Size (in pixels) of the patch used in patch-level processing. Only used if sample_type is 'patch'. Default: 1500."
    )

    parser.add_argument('--overlap', type=float, default=0.75,
        help="Fractional overlap between adjacent patches (0.0 to 1.0). Only used if sample_type is 'patch'. Default: 0.75."
    )

    args = parser.parse_args()

    if args.sample_type == 'core':
        create_all_feature_core_full_tissue_df()

    elif args.sample_type == 'resection':
        create_all_feature_resection_full_tissue_df()

    elif args.sample_type == 'patch':

        patch_size = args.patch_size
        overlap    = args.overlap

        create_all_feature_patches_df(patch_size, overlap)
