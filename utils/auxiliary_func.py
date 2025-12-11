import config.const as const
from PIL import Image
from skimage import io
import csv
import numpy as np
import os
import re
import pandas as pd
import shutil
from scipy.stats import mannwhitneyu, stats


def csv_to_tiff(csv_file_name, tiff_file_name):
    """
    Convert a CSV file of pixel values into a TIFF image.

    Parameters
    ----------
    csv_file_name : str
        Path to the input CSV file containing image data.
    tiff_file_name : str
        Output path where the TIFF image will be saved.
    """
    image_array = np.genfromtxt(csv_file_name, delimiter=',')
    my_image = Image.fromarray(image_array)
    my_image.save(tiff_file_name)
    

def extract_info_from_filename(filename):
    """
    Extract LEAP number and slide number from a filename.

    Expected format: 'LEAP<id>_slide<id>...'

    Parameters
    ----------
    filename : str
        Filename to parse.

    Returns
    -------
    tuple(str, str) or (None, None)
        Extracted (leap_number, slide_number), or (None, None) if not found.
    """
    match = re.match(r"LEAP(\d+)_slide(\d+)", filename)
    if match:
        leap_number = match.group(1)
        slide_number = match.group(2)
        return leap_number, slide_number
    else:
        return None, None

def get_leap_filenames(directory):
    """
    Extracts filenames starting with 'LEAP' from the given directory.

    Parameters:
    directory (str): Path to the directory.

    Returns:
    list: A list of filenames starting with 'LEAP'.
    """
    try:
        # List all files in the directory
        all_files = os.listdir(directory)
        
        # Filter files that start with "LEAP"
        leap_files = [file for file in all_files if file.startswith("LEAP")]
        
        return leap_files
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def delete_directory(directory_path):
    """
    Delete a directory and all its contents if it exists.

    Parameters
    ----------
    directory_path : str
        Full path to the directory to be deleted.
    """
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        print(f"Deleted: {directory_path}")
    else:
        print(f"Directory does not exist: {directory_path}")

def delete_files(files_dir, file_signature):
    """
    Delete all files in a directory that end with a given signature.

    Parameters
    ----------
    files_dir : str
        Path to the directory containing the files.

    file_signature : str
        File name ending pattern to match (e.g., '.pkl').
    """
    feature_files = [
        os.path.join(files_dir, f)
        for f in os.listdir(files_dir)
        if f.endswith(file_signature)
    ]            # Delete individual files
    for file_path in feature_files:
        os.remove(file_path)


def delete_file_unknown_location(directory, file_name):
    """
    Delete all seed_results.csv files in the given directory and its subdirectories.

    Parameters:
    - directory (str): Path to the base directory to search for seed_results.csv files.
    """
    # Iterate through the directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file == file_name:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


def delete_results_dirs(base_dir, delete_dir):
    """
    Recursively find and delete directories named 'results' in the specified base directory.

    Args:
        base_dir (str): The base directory to start searching from.
    """
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == delete_dir:
                results_dir = os.path.join(root, dir_name)
                print(f"Deleting: {results_dir}")
                shutil.rmtree(results_dir)
                print(f"Deleted: {results_dir}")


def save_fig(save_file_path, save_file_name, format_type, plt_fig, transparent=True):
    """
    Save a matplotlib figure to disk in the specified format.

    Parameters
    ----------
    save_file_path : str
        Directory where the figure should be saved.

    save_file_name : str
        File name without extension.

    format_type : str
        Format to save the figure in (e.g., 'png', 'pdf', 'svg').

    plt_fig : matplotlib.figure.Figure
        The matplotlib figure object to save.

    transparent : bool, optional
        Whether to save the figure with a transparent background. Default is True.
    """
    format_path = f'{save_file_path}/{save_file_name}.{format_type}'
    plt_fig.savefig(format_path, bbox_inches='tight', dpi=1200, transparent=transparent, format=format_type)
    print(f"Figure saved as {format_type} at: {format_path}")


def read_patches_files(patch_size, patch_overlap):
    """
    Load the filtered FLIM patch-level feature file for a given patch config.

    Parameters
    ----------
    patch_size : int
        Patch size used during preprocessing.

    patch_overlap : float
        Overlap value used during patch extraction (e.g., 0.75).

    Returns
    -------
    pd.DataFrame
        DataFrame of patch-level features with 'leap_ID' as string dtype.
    """
    # Construct file and directory paths
    patch_features_file_name = f"FLIM_features_patches_size_{patch_size}_overlap_{patch_overlap}_after_filter.csv"
    specific_patch_dir = os.path.join(const.PATCH_DIR, f"size_{patch_size}_overlap_{patch_overlap}")
    patch_features_path = os.path.join(specific_patch_dir, patch_features_file_name)

    # Read the patch DataFrame
    patch_df = pd.read_csv(patch_features_path, dtype={'leap_ID': str})
    return patch_df

def build_gnn_model_dir_name(model_param_dict):
    """
    Build a directory name string from model parameter dictionary.

    Parameters:
        model_param_dict (dict): Dictionary of model hyperparameters.

    Returns:
        str: A formatted and sanitized directory name string.
    """

    def sanitize(val):
        """Convert float-like values and list items to safe strings for directory names."""
        if isinstance(val, float):
            return f"{val}".replace('.', 'p')
        return str(val)

    # Build directory name from ordered keys
    parts = [
        f"batch_size_{model_param_dict['batch_size']}",
        f"class_ratio_{sanitize(model_param_dict['class_ratio'])}",
        f"dropout_rate_{sanitize(model_param_dict['dropout_rate'])}",
        f"epochs_{model_param_dict['epochs']}",
        f"heads_{model_param_dict['heads']}",
        f"hidden_layers_{sanitize(model_param_dict['hidden_layers'])}",
        f"lr_{sanitize(model_param_dict['lr'])}",
        f"model_type_{model_param_dict['model_type']}",
        f"output_size_{model_param_dict['output_size']}",
        f"pooling_{model_param_dict['pooling']}",
        f"test_ratio_{sanitize(model_param_dict['test_ratio'])}",
        f"weight_decay_{sanitize(model_param_dict['weight_decay'])}",
    ]

    return "_".join(parts)


def concatenate_summary_results(base_directory):
    """
    Concatenate all summary_results.pkl files in the given base directory.

    Parameters:
    - base_directory (str): The base directory to search for summary_results.pkl files.

    Returns:
    - combined_df (pd.DataFrame): A DataFrame containing concatenated results from all summary_results.pkl files.
    """
    summary_results_files = []
    for root, dirs, files in os.walk(base_directory):
        # for file in files:
        #     if file == "summary_results.pkl":
        #         summary_results_files.append(os.path.join(root, file))
        for file in files:
            if file.startswith("summary_results") and file.endswith(".pkl"):
                summary_results_files.append(os.path.join(root, file))
    
    # Check if any files were found
    if not summary_results_files:
        print("No summary_results.pkl files found in the specified directory.")
        return pd.DataFrame()  # Return an empty DataFrame if no files are found
    
    # Load and concatenate all files
    combined_df = pd.concat(
        [pd.read_pickle(file) for file in summary_results_files],
        ignore_index=True
    )
    
    return combined_df


def concatenate_seed_results(base_directory):
    """
    Concatenate all seed_results.pkl files in the given base directory.

    Parameters:
    - base_directory (str): The model base directory to search for all seed_results.pkl files.

    Returns:
    - combined_df (pd.DataFrame): A DataFrame containing concatenated results from all seed_results.pkl files.
    """
    all_seeds_results_files = []
    for root, dirs, files in os.walk(base_directory):

        for file in files:
            if file.startswith("seed_results") and file.endswith(".pkl"):
                all_seeds_results_files.append(os.path.join(root, file))
    
    # Check if any files were found
    if not all_seeds_results_files:
        print("No summary_results.pkl files found in the specified directory.")
        return pd.DataFrame()  # Return an empty DataFrame if no files are found
    
    # Load and concatenate all files
    combined_df = pd.concat(
        [pd.read_pickle(file) for file in all_seeds_results_files],
        ignore_index=True
    )
    
    return combined_df


def upper_one_tailed_u_test(data, mu_0=0.5, significance_level=0.05):
    """
    Perform a one-tailed Mann-Whitney U-test to test whether the data distribution is significantly greater than mu_0.

    Parameters:
        data (list or array-like): Sample data to test.
        mu_0 (float): The value to compare the data distribution against (default is 0.5).
        significance_level (float): The significance level for the test (default is 0.05).

    Returns:
        dict: A dictionary with U-statistic, p-value, adjusted significance level, and conclusion.
    """
    # Create a comparison sample with the value mu_0 repeated to match the size of the data
    comparison_sample = [mu_0] * len(data)

    # Perform the Mann-Whitney U-test (one-tailed, greater alternative)
    u_stat, p_value = mannwhitneyu(data, comparison_sample, alternative='greater')

    # Check if the p-value is less than the significance level
    conclusion = ""
    if p_value < significance_level:
        conclusion = f"Reject the null hypothesis: The distribution is significantly greater than {mu_0}."
    else:
        conclusion = f"Fail to reject the null hypothesis: No significant evidence that the distribution is greater than {mu_0}."

    return {
        "u_statistic": round(u_stat, 4),
        "p_value": round(p_value, 4),
        "significance_level": round(significance_level, 4),
        "conclusion": conclusion
    }


def upper_one_tailed_t_test(data, mu_0=0.5, significance_level=0.05):
    """
    Perform a one-sample upper one-tailed t-test.

    Parameters
    ----------
    data : array-like
        Sample data to test.
    
    mu_0 : float, optional
        Null hypothesis mean. Default is 0.5.
    
    significance_level : float, optional
        Significance threshold (alpha). Default is 0.05.

    Returns
    -------
    dict: A dictionary with t-statistic, p-value, adjusted significance level, and conclusion.
    """

    # Perform a t-test
    t_stat, p_value = stats.ttest_1samp(data, popmean=mu_0, alternative='greater')

    # Adjust p-value for one-tailed test
    # p_value_one_tailed = p_value / 2
    p_value_one_tailed = p_value


    # Adjust significance level for one-tailed test
    # adjusted_significance_level = significance_level / 2
    adjusted_significance_level = significance_level


    # Check if the t-statistic is in the lower tail
    conclusion = ""
    if t_stat > 0:  # t-statistic should be negative for a lower-tail test
        if p_value_one_tailed < adjusted_significance_level:
            conclusion = f"Reject the null hypothesis: The mean is significantly upper than {mu_0}."
        else:
            conclusion = f"Fail to reject the null hypothesis: No significant evidence the mean is upper than {mu_0}."
    else:
        conclusion = "Fail to reject the null hypothesis: t-statistic is not in the upper tail."

    return {
        "t_statistic": round(t_stat, 4),
        "p_value_one_tailed": round(p_value_one_tailed, 4),
        "significance_level": round(adjusted_significance_level, 4),
        "conclusion": conclusion
    }


