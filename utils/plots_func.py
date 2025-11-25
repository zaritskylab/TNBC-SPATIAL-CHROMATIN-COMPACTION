import os

import pandas as pd
import config.params as params
import config.const as const
from utils.auxiliary_func import *

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
from scipy.stats import pearsonr,linregress
from skimage import io, img_as_float
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as patches
from scipy.stats import ttest_ind
from skimage import io
from itertools import cycle
from matplotlib.patches import Circle
from sklearn.neighbors import NearestNeighbors
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import tifffile
from matplotlib import font_manager




def extract_distribution_seed_results(root_dir, seed_prefix='seed_', predictions_filename='predictions.csv', agg_results=False):
    """
    Iterates over subdirectories in `root_dir` whose names start with `seed_prefix`.
    Reads the `predictions_filename` file in each seed directory, computes FPR, TPR, 
    and AUC for the predictions, and appends them to lists.

    Parameters
    ----------
    root_dir : str
        The top-level directory containing seed subdirectories.
    seed_prefix : str, optional
        The prefix for seed directories (default is 'seed_').
    predictions_filename : str, optional
        The name of the predictions file in each seed directory (default is 'predictions.csv').

    Returns
    -------
    fpr_list : list of np.ndarray
        List of false positive rates for each seed.
    tpr_list : list of np.ndarray
        List of true positive rates for each seed.
    auc_data : list of float
        List of AUC values for each seed.
    seeds : list of str
        List of seed directory names (or seed numbers) corresponding to each ROC computation.
    """
    fpr_list = []
    tpr_list = []
    auc_data = []
    seeds = []

    # Loop over all items in the root directory
    for item in os.listdir(root_dir):
        # Check if item is a directory starting with seed_prefix
        if item.startswith(seed_prefix):
            seed_dir = os.path.join(root_dir, item)
            if os.path.isdir(seed_dir):
                # Build the path to the predictions file
                predictions_path = os.path.join(seed_dir, predictions_filename)

                # Check if predictions file exists
                if os.path.exists(predictions_path):
                    # Read the predictions file
                    # If it's an Excel file, replace with pd.read_excel
                    df = pd.read_csv(predictions_path)
                    if agg_results:
                        df = df.groupby('sample_id', as_index=False).agg({
                        'y_true': 'mean',
                        'y_pred': 'mean'
                        })
                
                    # Extract true and predicted values
                    y_true = df['y_true']
                    y_pred = df['y_pred']

                    # Compute FPR, TPR, thresholds, and AUC
                    fpr, tpr, _ = roc_curve(y_true, y_pred)
                    auc_val = roc_auc_score(y_true, y_pred)

                    # Append to the lists
                    fpr_list.append(fpr)
                    tpr_list.append(tpr)
                    auc_data.append(auc_val)

                    # Store the seed directory name or just the numeric portion
                    # For just the number, uncomment the line below and comment out seeds.append(item).
                    seeds.append(item)

    return seeds, auc_data, fpr_list, tpr_list


def extract_gnn_seed_results(directory, auc_column, tpr_column, fpr_column):
    """
    Extract seed results from a given directory.

    Parameters:
    - directory (str): Path to the directory containing the seed_results.csv file.

    Returns:
    - seed_list (list): List of seed values.
    - auc_scores (list): List of AUC scores.
    - fpr_list (list): List of False Positive Rate (FPR) values.
    - tpr_list (list): List of True Positive Rate (TPR) values.
    """
    # Define the path to the seed_results.csv file
    seed_results_file = os.path.join(directory, "seed_results.pkl")

    # Check if the file exists
    if not os.path.exists(seed_results_file):
        raise FileNotFoundError(f"{seed_results_file} does not exist in the specified directory.")

    # Load the pickle file into a DataFrame
    with open(seed_results_file, 'rb') as file:
        seed_results_df = pickle.load(file)
        # print(seed_results_df)

    # Extract the required columns
    seed_list = seed_results_df['seed_val'].tolist()
    auc_scores = seed_results_df[auc_column].tolist()
    fpr_list = seed_results_df[fpr_column].tolist()
    tpr_list = seed_results_df[tpr_column].tolist()

    return seed_list, auc_scores, fpr_list, tpr_list


def plot_mean_auc_roc_lists(
    fpr_list, tpr_list, auc_data, std_num=1,
    save_file_path=False, save_file_name=None,
    title='ROC Curves',
    extra_fpr=None, extra_tpr=None, extra_auc=None, extra_label=None
):
    mean_fpr = np.unique(np.concatenate(fpr_list))
    tpr_interpolated = np.zeros((len(fpr_list), len(mean_fpr)))

    for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
        tpr_interpolated[i, :] = np.interp(mean_fpr, fpr, tpr)

    mean_tpr = np.mean(tpr_interpolated, axis=0)

    std_tpr = np.std(tpr_interpolated, axis=0)

    # Calculate mean ROC-AUC score
    median_auc = np.median(auc_data)
    mean_auc = np.mean(auc_data)
    print("Median AUC:" ,median_auc)

    auc_std = np.std(auc_data)  # Calculate std deviation of AUC

    max_auc = np.max(auc_data)
    min_auc = np.min(auc_data)

    plt.figure(figsize=(8, 6))

    # Plot mean ROC curve (bolded)
    # plt.plot(mean_fpr, mean_tpr, label=f'Median ROC (AUC = {median_auc:.2f})', color='green', linestyle='--', linewidth=4)
    plt.plot(mean_fpr, mean_tpr, 
            label=f'Mean ROC (AUC = {round(mean_auc, 2)} ± {round(auc_std, 2)})',
            color='green', linestyle='--', linewidth=4)

    # Plot standard deviation area
    plt.fill_between(mean_fpr, mean_tpr - std_num*std_tpr, mean_tpr + std_num*std_tpr, color='lightgreen', alpha=0.4, label=f'± {std_num} std. dev.')

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    if title is not None:
        plt.title(title, fontweight='bold', fontsize=16)
    else:
        plt.title(title, fontweight='bold', fontsize=16)
    plt.xticks(fontweight='bold', fontsize=10)
    plt.yticks(fontweight='bold', fontsize=10)
    plt.legend(loc="lower right")
    
    plt.text(0.7, 0.1, f'Best AUC = {round(max_auc, 2)}', fontsize=12, color='green', fontweight='bold')


    # Optionally add an extra ROC curve
    if extra_fpr is not None and extra_tpr is not None and extra_auc is not None:
        label = extra_label if extra_label else f'Extra ROC (AUC = {extra_auc:.2f})'
        plt.plot(extra_fpr, extra_tpr, label=label, color='blue', linestyle='-', linewidth=2)
        
    # Save the figure if save_file_name is provided
    if save_file_path:
        os.makedirs(save_file_path, exist_ok=True)
        # Save as PDF
        pdf_path = f'{save_file_path}/{save_file_name}.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', dpi=1200, transparent=True, format='pdf')
        print(f"Figure saved as PDF at: {pdf_path}")


    plt.show()


def plot_boxplot_by_category(df, col, category_col='category', title=None, ylabel=None, figsize=(4, 4), order=None):
    """
    Plots a boxplot and stripplot for the specified column (col) of the DataFrame,
    grouped by the given category column. Automatically handles any number of categories.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        col (str): The column name for which to create the boxplot.
        category_col (str): The column name used for grouping (default is 'category').
        title (str): Title for the plot (default is f'{col} by {category_col}').
        ylabel (str): Y-axis label (default is col).
        figsize (tuple): Figure size (default is (4,4)).
    """
    plt.figure(figsize=figsize)

    # Define fixed colors: first = blue, second = orange
    fixed_palette = ['#1f77b4', '#ff7f0e']  # blue, orange from matplotlib default
    if order is not None: 
        order=order
    else:
        order = df[category_col].dropna().unique().tolist()

    # Adjust color list to match the number of categories (fallback to default if more than 2)
    if len(order) == 2:
        palette = fixed_palette
    else:
        palette = sns.color_palette("Set2", len(order))

    color_map = dict(zip(order, palette))

    # Boxplot
    sns.boxplot(x=category_col, y=col, data=df, order=order,
                palette=color_map, showfliers=False, boxprops=dict(alpha=.5))

    # Stripplot
    sns.stripplot(x=category_col, y=col, data=df, order=order,
                  palette=color_map, dodge=True, alpha=0.7, jitter=True)

    plt.ylabel(ylabel if ylabel else col)
    plt.title(title if title else f'{col} by {category_col}')
    
    return plt



def plot_histogram(correlation):
    # Plot the correlation distribution
    sns.histplot(correlation, kde=True)

    # Set labels and title
    plt.xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Correlation between Proximity and Similarity in Lifetime', fontsize=16, fontweight='bold')

    # Set tick parameters
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')

    return plt
    

def plot_lifetime_correlation_with_density_map(df, feature):
    """
    Plot a 2D density heatmap between 'lifetime_mean' and a given feature,
    overlay a linear regression line, and compute Pearson correlation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'lifetime_mean' and the selected feature.
    feature : str
        The feature to correlate with 'lifetime_mean'.

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    # Drop rows with NaNs
    df_clean = df[['lifetime_mean', feature]].dropna()

    x = df_clean['lifetime_mean']
    y = df_clean[feature]

    # Pearson correlation
    corr_coef, p_value = pearsonr(x, y)

    # Linear regression
    slope, intercept, _, _, _ = linregress(x, y)
    reg_line = slope * x + intercept

    # Plot
    plt.figure(figsize=(8, 6))
    ax = sns.histplot(
        x=x,
        y=y,
        bins=100,
        pthresh=0,
        cmap='cividis_r',
        # cmap='viridis',
        cbar=True, 
    )

    # Plot regression line
    x_vals = np.linspace(x.min(), x.max(), 500)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='red', label='Linear fit')

    # Labels and title
    plt.xlabel("Lifetime Mean")
    plt.ylabel(feature.replace("_", " ").capitalize())
    plt.title(f'{feature.replace("_", " ").capitalize()} vs Lifetime\n'
              f'Pearson r = {corr_coef:.3f}, p = {p_value:.4f}')

    # Label the colorbar
    if ax.collections:
        cbar = ax.collections[0].colorbar
        cbar.set_label("Nuclei Density")

    plt.legend()
    plt.tight_layout()
    plt.show()

    return corr_coef



    ########### Image plots ###########


def plot_full_image_from_path(img_path, image_type, figure_size=(8, 8), save_path=None, with_scale_bar=True):
    """
    Display the full image with gamma correction and a scale bar.

    Parameters:
        img_path (str): Path to the image file.
        image_type (str): Label for the image (used in title).
        figure_size (tuple): Figure size in inches (default: (8, 8)).
        save_path (str or None): Path to save the figure as PDF (default: None).
    """
    image = tifffile.imread(img_path)
    vmin, vmax = np.min(image), np.max(image)


    fig, ax = plt.subplots(figsize=figure_size)

    ax.imshow(image, cmap='magma', vmin=vmin, vmax=vmax)
    ax.set_title(f"Full Image {image_type}")
    ax.axis('off')
    if with_scale_bar:
        scale_bar = ScaleBar(
            dx=1.139, units="µm", location='lower right',
            color='white', box_alpha=0, scale_loc='bottom'
        )
        scale_bar.set_font_properties({"size": 14, "weight": "bold"})
        ax.add_artist(scale_bar)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=1200, transparent=True, format='png')
    plt.show()


def plot_image_with_zoom_box(img_path, image_type, center=(950, 2500), zoom_size=300, gamma=0.7, figure_size=(8, 8), save_path=None):
    """
    Display the full image with a red rectangle highlighting the zoom region.

    Parameters:
        img_path (str): Path to the image file.
        image_type (str): Label for the image (used in title).
        center (tuple): (x, y) center of the zoom region.
        zoom_size (int): Size of the zoomed square region.
        gamma (float): Gamma correction value.
        figure_size (tuple): Figure size.
        save_path (str or None): Path to save the figure as PDF (default: None).
    """
    image = img_as_float(io.imread(img_path))
    image = np.power(image, gamma)

    x_center, y_center = center
    half = zoom_size // 2
    x_start = x_center - half
    y_start = y_center - half

    fig, ax = plt.subplots(figsize=figure_size)
    ax.imshow(image, cmap='magma')
    ax.set_title(f"{image_type} - zoom area highlighted")
    ax.axis('off')

    rect = patches.Rectangle((x_start, y_start), zoom_size, zoom_size,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    ax.add_artist(ScaleBar(
        dx=1.139, units="µm", location='lower left',
        color='white', box_alpha=0, scale_loc='bottom'
    ))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=1200, transparent=True, format='pdf')
    plt.show()


def plot_zoomed_crop_from_path(img_path, image_type, center=(950, 2500), zoom_size=300, gamma=0.7, figure_size=(6, 6), save_path=None):
    """
    Display only the zoomed-in cropped region of the image.

    Parameters:
        img_path (str): Path to the image file.
        image_type (str): Label for the image (used in title).
        center (tuple): (x, y) center of the zoom region.
        zoom_size (int): Size of the zoomed square region.
        gamma (float): Gamma correction value.
        figure_size (tuple): Figure size.
        save_path (str or None): Path to save the figure as PDF (default: None).
    """
    image = img_as_float(io.imread(img_path))
    image = np.power(image, gamma)

    x_center, y_center = center
    half = zoom_size // 2
    cropped = image[
        y_center - half : y_center + half,
        x_center - half : x_center + half
    ]

    fig, ax = plt.subplots(figsize=figure_size)
    ax.imshow(cropped, cmap='magma')
    ax.set_title(f"{image_type} - zoomed-in region")
    ax.axis('off')

    ax.add_artist(ScaleBar(
        dx=1.139, units="µm", location='lower left',
        color='white', box_alpha=0, scale_loc='bottom'
    ))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=1200, transparent=True, format='pdf')
    plt.show()


def plot_full_image_with_zoom_side_by_side(img_path, image_type, center=(950, 2500), zoom_size=300, gamma=0.7, figure_size=(14, 7), cmap_color='magma', save_path=None):
    """
    Display a side-by-side figure showing:
    - Full image with red zoom rectangle
    - Zoomed-in cropped region

    Parameters:
        img_path (str): Path to the image file.
        image_type (str): Label for the image (used in titles).
        center (tuple): (x, y) center of the zoom region.
        zoom_size (int): Size of the zoomed square region.
        gamma (float): Gamma correction value.
        figure_size (tuple): Figure size.
        save_path (str or None): Path to save the figure as PDF (default: None).
    """
    image = img_as_float(io.imread(img_path))
    image = np.power(image, gamma)

    x_center, y_center = center
    half = zoom_size // 2
    x_start = x_center - half
    y_start = y_center - half

    cropped = image[
        y_center - half : y_center + half,
        x_center - half : x_center + half
    ]

    fig, axes = plt.subplots(1, 2, figsize=figure_size)

    # Full image with zoom box
    ax0 = axes[0]
    ax0.imshow(image, cmap=cmap_color)
    ax0.set_title(f"Full {image_type} - with Zoom Area")
    ax0.axis('off')

    rect = patches.Rectangle(
        (x_start, y_start), zoom_size, zoom_size,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax0.add_patch(rect)

    ax0.add_artist(ScaleBar(
        dx=1.139, units="µm", location='lower left',
        color='white', box_alpha=0, scale_loc='bottom'
    ))

    # Zoomed-in crop
    ax1 = axes[1]
    ax1.imshow(cropped, cmap=cmap_color)
    ax1.set_title(f"{image_type} - Zoomed-In Region")
    ax1.axis('off')

    ax1.add_artist(ScaleBar(
        dx=1.139, units="µm", location='lower left',
        color='white', box_alpha=0, scale_loc='bottom'
    ))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=1200, transparent=True, format='pdf')
    plt.show()


def plot_zoomed_comparison(
    img_path_1=None, img_path_2=None,
    label_1='Image 1', label_2='Image 2',
    center=(950, 2500), zoom_size=300, gamma=0.7,
    figure_size=(12, 6),cmap_color_1='magma', cmap_color_2='magma',
    save_path=None, img1=None, img2=None
    ):
    """
    Plot zoomed-in regions from two images side-by-side for comparison.

    img_path_1 : str
        Path to the first image file (e.g., fluorescent intensity). Ignored if `img1` is provided.
    img_path_2 : str
        Path to the second image file (e.g., FLIM lifetime). Ignored if `img2` is provided.
    label_1 : str, optional
        Title label for the first zoomed-in image (default: 'Image 1').
    label_2 : str, optional
        Title label for the second zoomed-in image (default: 'Image 2').
    center : tuple of int, optional
        (x, y) center pixel coordinates of the zoomed region (default: (950, 2500)).
    zoom_size : int, optional
        Width/height of the square zoom region in pixels (default: 300).
    gamma : float, optional
        Gamma correction to apply before plotting (default: 0.7).
    figure_size : tuple of float, optional
        Size of the output figure in inches (width, height) (default: (12, 6)).
    cmap_color_1 : str, optional
        Matplotlib colormap name for the first image (default: 'magma').
    cmap_color_2 : str, optional
        Matplotlib colormap name for the second image (default: 'magma').
    save_path : str or None, optional
        If provided, save the figure to this path (PDF) using 
        dpi=1200, transparent background, and tight layout.
        Example: 'zoom_comparison.pdf'
    img1 : ndarray or None, optional
        If provided, use this numpy array for the first image instead of reading from `img_path_1`.
    img2 : ndarray or None, optional
        If provided, use this numpy array for the second image instead of reading from `img_path_2`.

    """
    # Load and gamma-correct both images

    if img1 is not None:
        img1 = np.power(img_as_float(img1), gamma)
    else:
        img1 = np.power(img_as_float(io.imread(img_path_1)), gamma)

    if img2 is not None: 
        img2 = np.power(img_as_float(img2), gamma)
    else:
        img2 = np.power(img_as_float(io.imread(img_path_2)), gamma)

    # Crop both images at the same region
    x_center, y_center = center
    half = zoom_size // 2
    y_slice = slice(y_center - half, y_center + half)
    x_slice = slice(x_center - half, x_center + half)

    crop1 = img1[y_slice, x_slice]
    crop2 = img2[y_slice, x_slice]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figure_size)

    # Image 1
    ax0 = axes[0]
    im0 = ax0.imshow(crop1, cmap=cmap_color_1)
    ax0.set_title(label_1)
    ax0.axis('off')
    ax0.add_artist(ScaleBar(
        dx=1.139, units="µm",
        location='lower left', color='white', box_alpha=0, scale_loc='bottom'
    ))
    # # Optionally add colorbar
    # plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # Image 2
    ax1 = axes[1]
    im1 = ax1.imshow(crop2, cmap=cmap_color_2)
    ax1.set_title(label_2)
    ax1.axis('off')
    ax1.add_artist(ScaleBar(
        dx=1.139, units="µm",
        location='lower left', color='white', box_alpha=0, scale_loc='bottom'
    ))
    # plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save if path is given
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=1200, transparent=True, format='pdf')

    plt.show()



def plot_full_and_zoom_separately_with_colorbar(
    img_path,
    center=(770,2330),
    zoom_size=300,
    using_mask=False,
    cmap_col='magma',
    pixel_size_um=1.139,
    save_full_name=None,
    save_zoom_name=None, 
    save_colorbar_name=None,
    save_dir=None,
    color_bar=True,

):
    """
    Display full image and zoom-in region in separate figures,
    each with a scale bar and color bar (no title, clean layout).
    """
    # Load and gamma correct
    image = tifffile.imread(img_path).astype(np.float32)
    vmin, vmax = np.min(image), np.max(image)

    if using_mask:
        masked_image = np.ma.masked_where(image == 0, image)
        image = masked_image
        data = image.compressed()        # drop masked zeros
        vmin, vmax = np.percentile(data, [0.2, 99.75])
    

    cmap_color = plt.colormaps.get_cmap(cmap_col).copy()
    cmap_color.set_bad(color='black')

    x_center, y_center = center
    half = zoom_size // 2
    x0 = max(x_center - half, 0)
    y0 = max(y_center - half, 0)
    x1 = x_center + half
    y1 = y_center + half

    cropped = image[y0:y1, x0:x1]

    # --- Full Image (no colorbar) ---
    fig_full, ax_full = plt.subplots(figsize=(7, 7))
    im_full = ax_full.imshow(image, cmap=cmap_color, vmin=vmin, vmax=vmax)
    ax_full.axis('off')

    rect = patches.Rectangle(
        (x0, y0), zoom_size, zoom_size,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax_full.add_patch(rect)

    scalebar_full = ScaleBar(
        dx=pixel_size_um, units="µm", location='lower right',
        color='white', box_alpha=0, scale_loc='bottom'
    )
    scalebar_full.set_font_properties({"size": 12, "weight": "bold"})


    ax_full.add_artist(scalebar_full)

    if save_full_name:
        save_fig(save_dir, save_full_name, 'png', fig_full, transparent=True)

    # --- Zoomed Image (no colorbar) ---
    fig_zoom, ax_zoom = plt.subplots(figsize=(3, 3))
    im_zoom = ax_zoom.imshow(cropped, cmap=cmap_color, vmin=vmin, vmax=vmax)
    ax_zoom.axis('off')

    scalebar_zoom = ScaleBar(
        dx=pixel_size_um, units="µm", location='lower right',
        color='white', box_alpha=0, scale_loc='bottom'
    )

    scalebar_zoom.set_font_properties({"size": 20, "weight": "bold"})

    ax_zoom.add_artist(scalebar_zoom)

    if save_zoom_name:
        save_fig(save_dir, save_zoom_name, 'png', fig_zoom, transparent=True)


    if color_bar:
        # --- Separate Colorbar Only ---
        fig_cbar, ax_cbar = plt.subplots(figsize=(1.2, 4))
        fig_cbar.subplots_adjust(left=0.5, right=0.8)  # adjust margins for slim layout

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
        sm.set_array([])

        cbar = fig_cbar.colorbar(sm, cax=ax_cbar)
        cbar.set_label("Intensity (a.u.)", rotation=270, labelpad=15)

        if save_colorbar_name:
            save_fig(save_dir, save_colorbar_name, 'pdf', fig_cbar, transparent=True)

    plt.show()

    

def plot_hist_plt(df, title, x_label, bins_amount=30):
  
    plt.figure(figsize=(10, 6))
    sns.histplot(df['area'], bins=bins_amount, kde=True, color='blue')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def plot_hist_sns(df, title, x_label):
    sns.histplot(df, kde=True)

    # Set labels and title
    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')

    # Set tick parameters
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')

    # Display the plot
    plt.show()


def category_seperation_by_feature(df, feature_name, mapping={'non responder': 1, 'responder': 0}, order=None):
    plt_fig = plot_boxplot_by_category(df, feature_name, order=order)
    # Split data into two groups based on 'category'
    non_responders = df[df['category'] == 'non responder'][feature_name]
    responders = df[df['category'] == 'responder'][feature_name]

    # Perform the Mann-Whitney U test
    # u_stat, u_p_value = mannwhitneyu(non_responders, responders, alternative='two-sided')
    t_stat, t_p_value = ttest_ind(non_responders, responders, equal_var=False)  # Welch’s t-test


    print("\nT-statistic:", t_stat)
    print("P-value:", t_p_value)

    # Prepare extra ROC from median_df
    df['binary_label'] = df['category'].map(mapping)
    y_true = df['binary_label']
    y_scores = df[feature_name]

    auc_score = roc_auc_score(y_true, y_scores)

    print(f"{feature_name} auc - {auc_score} \n")

    return plt_fig

    ########### Spatial information plots ###########
def plot_probability_map_one_leap(df, leap_number, category, with_axis=False):
    x = df['X coordinate']
    y = df['Y coordinate']
    density = df['prob_results']

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(x, y, c=density, cmap='viridis', s=0.5, vmin=0, vmax=1)

    cbar = plt.colorbar(scatter, label='Responder probability', fraction=0.046, pad=0.04, shrink=0.6)
    cbar.ax.tick_params(labelsize=14, width=2)  
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.yaxis.label.set_weight('bold')

    # plt.xlabel('X Coordinate', fontsize=12, fontweight='bold')
    # plt.ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    plt.title(f'Leap {leap_number} {category.capitalize()} Probability Map', fontsize=14, fontweight='bold')
    if with_axis:
        plt.xticks(fontsize=12, ticks=np.arange(min(x), max(x)+1, 250),  rotation=90)
        plt.yticks(fontsize=12, ticks=np.arange(min(y), max(y)+1, 250))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    else:
        plt.xticks([])
        plt.yticks([])
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.show()


def plot_probability_map_some_leaps(df, leap_ids, category, row, column):
    fig, axes = plt.subplots(column, row, figsize=(column*4, row*4), constrained_layout=True)
    fig.suptitle(f'{category.capitalize()} Probability Maps', fontsize=24, fontweight='bold')

    scatter_plots = []
    for ax, leap_id in zip(axes.flatten(), leap_ids):
        leap_df = df[(df['leap_ID'] == leap_id) & (df['category'] == category)]
        if leap_df.empty:
            continue
        x = leap_df['X coordinate']
        y = leap_df['Y coordinate']
        density = leap_df['prob_results']

        scatter = ax.scatter(x, y, c=density, cmap='viridis', s=0.5, vmin=0, vmax=1)
        scatter_plots.append(scatter)
        ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')
        ax.set_title(f'Leap {leap_id}', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12, width=2)

    # Adjust layout and place color bar on the right side of the grid
    cbar = fig.colorbar(scatter_plots[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Responder probability', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14, width=2)

    plt.show()


def plot_probability_map_custom(df, leap_ids, category, add_sample_id=False, save_path=None):
    # Determine the number of rows and columns for the subplot grid
    n_leaps = len(leap_ids)
    n_cols = min(6, n_leaps)
    n_rows = (n_leaps + 6) // 6

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), constrained_layout=True)
    fig.suptitle(f'{category.capitalize()} Probability Maps', fontsize=50, fontweight='bold')

    scatter_plots = []
    for i, leap_id in enumerate(leap_ids):
        row, col = divmod(i, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        leap_df = df[(df['leap_ID'] == leap_id) & (df['category'] == category)]
        if leap_df.empty:
            ax.axis('off')
            continue
        x = leap_df['X coordinate']
        y = leap_df['Y coordinate']
        density = leap_df['prob_results']

        scatter = ax.scatter(x, y, c=density, cmap='cividis_r', s=0.5, vmin=0, vmax=1)
        scatter_plots.append(scatter)
        if add_sample_id:
            ax.set_title(f'Leap {leap_id}', fontsize=30, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # Invert the y-axis and set the aspect ratio to be equal
        ax.invert_yaxis()
        # ax.set_aspect('equal', adjustable='box')

    # Turn off any remaining unused subplots
    for j in range(i + 1, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        if n_rows > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')

    # Adjust layout and place color bar on the right side of the grid
    if scatter_plots:
        cbar = fig.colorbar(scatter_plots[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
        # cbar.set_label('Responder probability', fontsize=40, fontweight='bold')
        cbar.ax.tick_params(labelsize=20, width=2)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=1200, transparent=True, format='pdf')

    plt.show()


def probability_maps_r_nr(aggregated_df, show_sample_id=False):
    responder_leaps = aggregated_df[aggregated_df['category'] == 'responder']['leap_ID'].unique()
    non_responder_leaps = aggregated_df[aggregated_df['category'] == 'non responder']['leap_ID'].unique()
    plot_probability_map_custom(aggregated_df, non_responder_leaps, 'non responder', show_sample_id)
    plot_probability_map_custom(aggregated_df, responder_leaps, 'responder', show_sample_id)



def plot_one_transition(leap_id, prob_coords, lifetime_and_prob_df, radius_list, radius_colors_dict, path_colors=['black'], figure_save_dir=None, save_plot=False, val_min=0, val_max=1):
    # prob_coords = pd.read_csv(prob_coords_leap_path)
    # prob_coords['vertex-index'] = prob_coords['vertex-index']+1

    
    flim_prob_path = os.path.join(const.flim_model_probability_dir, f'Leap{leap_id}_flim_probability.tif')
    img = tifffile.imread(flim_prob_path)
    height, width = img.shape[:2]

    # Keep vertex-index to sort each shape's points
    grouped_coords = (
        prob_coords
        .sort_values(['index', 'vertex-index'])
        .groupby('index')
        .agg({
            'axis-0': list,
            'axis-1': list,
            'vertex-index': list
        })
        .rename(columns={'axis-0': 'y_coord', 'axis-1': 'x_coord', 'vertex-index': 'vertex_index'})
        .reset_index()
    )

    leap_df = lifetime_and_prob_df[lifetime_and_prob_df['leap_ID'] == leap_id].copy()

    
    for idx, row in grouped_coords.iterrows():
        y_coord = row['y_coord']
        x_coord = row['x_coord']
        # x_coord = row['x_coord']

        vertex_index = row['vertex_index']
        query_points = np.stack([x_coord, y_coord], axis=1)
        ref_points = leap_df[['X coordinate', 'Y coordinate']].values


        # Plot shape on image
        plt.figure(figsize=(5, 5))
        ax = plt.gca()  # Get the current axes so we can add patches
                # Add padding below each image for the scalebar
        
        img_padded = np.pad(
            img, ((0, 2200), (0, 0)), mode='constant', constant_values=np.nan
        )

        cmap_params = cm.get_cmap("cividis_r").copy()
        cmap_params.set_bad(color='white')  # where NaN → white

        masked_img = np.ma.masked_invalid(img_padded)

        
        ax.imshow(
            masked_img,
            cmap=cmap_params,
            vmax=1.0,
            vmin=0.0,
            interpolation="nearest",
        )

        resolution = 1.139

        scalebar = ScaleBar(
            resolution, "µm", length_fraction=0.3, location="lower right",
            box_alpha=0, width_fraction=0.02, font_properties={'size': 12}
        )
        plt.imshow(img, cmap="cividis_r", vmax=val_max, vmin=val_min, interpolation="nearest")

        ax.add_artist(scalebar)

        path_color = path_colors[idx]
        plt.plot(x_coord, y_coord, color=path_color, linewidth=0.5)

        # Plot radius circles in all 3 colors
        for radius in radius_list:
            for x, y in zip(x_coord, y_coord):
                circle = Circle(
                    (x, y),
                    radius,
                    edgecolor=radius_colors_dict[radius],
                    facecolor='none',
                    linewidth=1,
                    alpha=1
                )
                ax.add_patch(circle)

        plt.axis('off')
        plt.tight_layout()


        if save_plot:
            save_file_name= f"spatial_points_probs_tissue_leap_{leap_id}"
            format_type = 'png'
            save_fig(figure_save_dir, save_file_name, format_type, plt, transparent=False)

        plt.show()

        # Prepare to plot mean_lifetime vs vertex_index for all radii
        plt.figure(figsize=(6, 4))

        for radius in radius_list:
            nn = NearestNeighbors(radius=radius)
            nn.fit(ref_points)
            indices = nn.radius_neighbors(query_points, return_distance=False)

            mean_lifetime = []

            for inds in indices:
                if len(inds) > 0:
                    mean_lifetime.append(leap_df.iloc[inds]['lifetime_mean'].mean())
                else:
                    mean_lifetime.append(np.nan)

            result_df = pd.DataFrame({
                'vertex_index': vertex_index,
                'mean_lifetime': mean_lifetime
            }).dropna()

            if len(result_df) == 0:
                continue

            sorted_df = result_df.sort_values(by='vertex_index')

            plt.plot(
                sorted_df['vertex_index'],
                sorted_df['mean_lifetime'],
                marker='o',
                linestyle='-',
                color=radius_colors_dict[radius],
                label=f'Radius {radius}'
            )

        plt.xticks(sorted_df['vertex_index'])  # discrete ticks
        plt.xlabel('Spatial point')
        plt.ylabel('Mean Lifetime')
        plt.title('Mean Lifetime Across Spatial Transition (Radii Comparison)')
        plt.legend(title='Radius')
        plt.tight_layout()
        plt.ylim(2.2, 3.6)

        if save_plot:
            save_file_name= f"mean_lifetime_across_spatial_transition_by_radius_{leap_id}"
            save_fig(figure_save_dir, save_file_name, 'pdf', plt, transparent=True)

        plt.show()

def plot_transition_2_path_homo_region(leap_id, prob_coords, lifetime_and_prob_df, radius_list, radius_colors_dict, path_colors=['red', 'black'], figure_save_dir=None, save_plot=False, val_min=0, val_max=1):
    # prob_coords = pd.read_csv(prob_coords_leap_path)
    # prob_coords['vertex-index'] = prob_coords['vertex-index']+1

    # Keep vertex-index to sort each shape's points
    grouped_coords = (
        prob_coords
        .sort_values(['index', 'vertex-index'])
        .groupby('index')
        .agg({
            'axis-0': list,
            'axis-1': list,
            'vertex-index': list
        })
        .rename(columns={'axis-0': 'y_coord', 'axis-1': 'x_coord', 'vertex-index': 'vertex_index'})
        .reset_index()
    )

    flim_prob_path = os.path.join(const.flim_model_probability_dir, f'Leap{leap_id}_flim_probability.tif')
    img = tifffile.imread(flim_prob_path)
    height, width = img.shape[:2]

    leap_df = lifetime_and_prob_df[lifetime_and_prob_df['leap_ID'] == leap_id].copy()
    # max_val = leap_df['X coordinate'].max()
    # leap_df['X coordinate'] = max_val - leap_df['X coordinate']

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    plt.imshow(img, cmap="cividis_r", vmax=val_max, vmin=val_min, interpolation="nearest")

    path_colors = ['red', 'black']
    for idx, row in grouped_coords.iterrows():
        y_coord = row['y_coord']
        x_coord = row['x_coord']
        vertex_index = row['vertex_index']

        # Plot path
        path_color = path_colors[idx]
        plt.plot(x_coord, y_coord, color=path_color, linewidth=0.5)


        # Plot circles at each point for each radius
        for radius in radius_list:
            for x, y in zip(x_coord, y_coord):
                circle = Circle(
                    (x, y),
                    radius,
                    edgecolor=radius_colors_dict[radius],
                    facecolor='none',
                    linewidth=1,
                    alpha=1
                )
                ax.add_patch(circle)

    plt.axis('off')
    plt.tight_layout()

    if save_plot:
        save_file_name= f"spatial_points_probs_tissue_leap_{leap_id}_homo_no_transparent"
        save_fig(figure_save_dir, save_file_name, 'png', plt, transparent=False)
    plt.show()

    for idx, row in grouped_coords.iterrows():
        y_coord = row['y_coord']
        x_coord = row['x_coord']
        vertex_index = row['vertex_index']
        query_points = np.stack([x_coord, y_coord], axis=1)
        ref_points = leap_df[['X coordinate', 'Y coordinate']].values

        # Prepare to plot mean_lifetime vs vertex_index for all radii
        plt.figure(figsize=(6, 4))

        for radius in radius_list:
            nn = NearestNeighbors(radius=radius)
            nn.fit(ref_points)
            indices = nn.radius_neighbors(query_points, return_distance=False)

            mean_lifetime = []

            for inds in indices:
                if len(inds) > 0:
                    mean_lifetime.append(leap_df.iloc[inds]['lifetime_mean'].mean())
                else:
                    mean_lifetime.append(np.nan)

            result_df = pd.DataFrame({
                'vertex_index': vertex_index,
                'mean_lifetime': mean_lifetime
            }).dropna()

            if len(result_df) == 0:
                continue

            sorted_df = result_df.sort_values(by='vertex_index')

            plt.plot(
                sorted_df['vertex_index'],
                sorted_df['mean_lifetime'],
                marker='o',
                linestyle='-',
                color=radius_colors_dict[radius],
                label=f'Radius {radius}'
            )

        plt.xticks(sorted_df['vertex_index'])  # discrete ticks
        plt.xlabel('Spatial point')
        plt.ylabel('Mean Lifetime')
        plt.title(f'Mean Lifetime Across Spatial Transition (Radii Comparison) path {idx}')
        plt.legend(title='Radius')
        plt.tight_layout()
        plt.ylim(2.2, 3.6)

        if save_plot:
            save_file_name= f"mean_lifetime_across_spatial_transition_by_radius_{leap_id}_{idx}_homo"
            save_fig(figure_save_dir, save_file_name, 'pdf', plt, transparent=True)

        plt.show()






def truncate_cmap(cmap, minval=0.3, maxval=0.8, n=100):
    """Clip a colormap to avoid very pale/near-white tones."""
    return mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )

def size_for_radius(r, radius_list):
    # Legend circle sizes (points, not data units)
    rmin, rmax = min(radius_list), max(radius_list)
    min_ms, max_ms = 8, 18  # tweak for more/less size contrast
    if rmax == rmin:
        return (min_ms + max_ms) / 2.0
    return min_ms + (r - rmin) * (max_ms - min_ms) / (rmax - rmin)



def plot_transition_2_path_homo_hetro_region(leap_id, prob_coords, lifetime_and_prob_df, radius_list, figure_save_dir=None, save_plot=False, val_min=0, val_max=1):
    print(f"leap - {leap_id}")
    # ---- Load and prep path coordinates ----
    # prob_coords = pd.read_csv(prob_coords_leap_path)
    # prob_coords['vertex-index'] = prob_coords['vertex-index'] + 1

    # ---- Load probability image ----
    flim_prob_path = os.path.join(const.flim_model_probability_dir, f'Leap{leap_id}_flim_probability.tif')
    img = tifffile.imread(flim_prob_path)
    height, width = img.shape[:2]

    grouped_coords = (
        prob_coords
        .sort_values(['index', 'vertex-index'])
        .groupby('index')
        .agg({
            'axis-0': list,
            'axis-1': list,
            'vertex-index': list
        })
        .rename(columns={'axis-0': 'y_coord', 'axis-1': 'x_coord', 'vertex-index': 'vertex_index'})
        .reset_index()
    )

    norm = mcolors.Normalize(vmin=min(radius_list), vmax=max(radius_list))

    # ======================= MAIN FIGURE (NO LEGENDS) =======================
    fig_main, ax_main = plt.subplots(figsize=(6.4, 6.4))
    ax_main.imshow(img, cmap="cividis_r", vmax=val_max, vmin=val_min, interpolation="nearest")
    ax_main.set_title("Mean Lifetime Across Spatial Transition (Radii Comparison)", fontsize=11)

    pixel_size_um = 1.139
    # Add scale bar
    scalebar_main = ScaleBar(
        dx=pixel_size_um, units="µm", location='lower right',
        color='black', box_alpha=0, scale_loc='bottom'
    )

    scalebar_main.set_font_properties({"size": 10, "weight": "bold"})

    ax_main.add_artist(scalebar_main)

    # Use Oranges for Homogeny, Blues for Heterogeny
    orange_cmap = truncate_cmap(cm.Oranges, 0.3, 0.8)
    blue_cmap   = truncate_cmap(cm.Blues,   0.3, 0.8)

    # Path polyline colors to match families
    path_colors = ['darkorange', 'navy']  # 0 -> Homogeny, 1 -> Heterogeny
    
    # Iterate over both path groups
    for p_idx, row in grouped_coords.iterrows():
        y_coord = row['y_coord']
        x_coord = row['x_coord']
        vertex_index = row['vertex_index']

        # Path type + styling
        path_color = path_colors[p_idx % len(path_colors)]
        path_label = 'Homogeny' if (p_idx % 2 == 0) else 'Heterogeny'

        # Plot path line
        ax_main.plot(x_coord, y_coord, color=path_color, linewidth=1.2)

        # Choose cmap
        cmap = orange_cmap if path_label == 'Homogeny' else blue_cmap

        # Circles at each point, shaded by radius
        for r in radius_list:
            shade = cmap(norm(r))
            for x, y in zip(x_coord, y_coord):
                circle = Circle((x, y), r, edgecolor=shade, facecolor='none', linewidth=1, alpha=1)
                ax_main.add_patch(circle)

    ax_main.axis('off')
    fig_main.tight_layout()

    # ---- Save MAIN figure (no legends) ----
    if save_plot:
        main_name = f"spatial_points_probs_tissue_leap_{leap_id}_homogeny_heterogeny_orange_blue"
        save_fig(figure_save_dir, main_name, 'png', fig_main, transparent=False)
    
    plt.show()

    # ======================= LEGEND-ONLY FIGURE =======================
    # Build handles for a clean, separate legend canvas
    path_type_handles = [
        Line2D([0], [0], color=path_colors[0], lw=2, label='Homogeny'),
        Line2D([0], [0], color=path_colors[1], lw=2, label='Heterogeny'),
    ]

    homo_radius_handles, hetero_radius_handles = [], []
    homo_labels, hetero_labels = [], []

    for r in radius_list:
        ms = size_for_radius(r, radius_list)
        homo_shade = orange_cmap(norm(r))
        hetero_shade = blue_cmap(norm(r))

        homo_handle = Line2D([0], [0], marker='o', linestyle='None',
                                markersize=ms, markerfacecolor='none',
                                markeredgecolor=homo_shade, markeredgewidth=2)
        hetero_handle = Line2D([0], [0], marker='o', linestyle='None',
                                markersize=ms, markerfacecolor='none',
                                markeredgecolor=hetero_shade, markeredgewidth=2)

        homo_radius_handles.append(homo_handle)
        hetero_radius_handles.append(hetero_handle)
        homo_labels.append(f"{r} μm")
        hetero_labels.append(f"{r} μm")

    # Create a compact legend-only figure
    fig_leg, ax_leg = plt.subplots(figsize=(7.2, 3.6))
    ax_leg.axis('off')

    # Place the three legends on this separate canvas
    leg1 = ax_leg.legend(handles=path_type_handles, title="Path Type", loc='center left', bbox_to_anchor=(0.02, 0.5), frameon=False)
    ax_leg.add_artist(leg1)

    leg2 = ax_leg.legend(handles=homo_radius_handles, labels=homo_labels,
                            title="Radii (Homogeny)", loc='upper center', bbox_to_anchor=(0.55, 0.70),
                            ncol=len(radius_list), frameon=False)
    ax_leg.add_artist(leg2)

    leg3 = ax_leg.legend(handles=hetero_radius_handles, labels=hetero_labels,
                            title="Radii (Heterogeny)", loc='lower center', bbox_to_anchor=(0.55, 0.30),
                            ncol=len(radius_list), frameon=False)

    fig_leg.tight_layout()

    # ---- Save LEGEND figure ----
    if save_plot:
        legend_name = f"spatial_points_probs_tissue_leap_{leap_id}_legend_orange_blue"
        save_fig(figure_save_dir, legend_name, 'pdf', fig_leg, transparent=True)
    plt.show()


    path_type_for_idx = {0: 'Homogeneous', 1: 'Heterogeneous'}
    cmap_for_type = {
        'Homogeneous': orange_cmap,
        'Heterogeneous': blue_cmap
    }

    leap_df = lifetime_and_prob_df[lifetime_and_prob_df['leap_ID'] == leap_id].copy()
    ref_points = leap_df[['X coordinate', 'Y coordinate']].values

    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    all_xticks = set()

    for idx, row in grouped_coords.iterrows():
        y_coord = row['y_coord']
        x_coord = row['x_coord']
        vertex_index = row['vertex_index']

        path_type = path_type_for_idx.get(idx, 'Heterogeneous' if idx % 2 else 'Homogeneous')
        cmap = cmap_for_type[path_type]
        query_points = np.stack([x_coord, y_coord], axis=1)

        for radius in radius_list:
            nn = NearestNeighbors(radius=radius)
            nn.fit(ref_points)
            indices = nn.radius_neighbors(query_points, return_distance=False)

            mean_lifetime = []
            for inds in indices:
                if len(inds) > 0:
                    mean_lifetime.append(leap_df.iloc[inds]['lifetime_mean'].mean())
                else:
                    mean_lifetime.append(np.nan)

            result_df = pd.DataFrame({
                'vertex_index': vertex_index,
                'mean_lifetime': mean_lifetime
            }).dropna()

            if len(result_df) == 0:
                continue

            sorted_df = result_df.sort_values(by='vertex_index')
            all_xticks.update(sorted_df['vertex_index'].tolist())

            color_shade = cmap(norm(radius))
            ax.plot(
                sorted_df['vertex_index'],
                sorted_df['mean_lifetime'],
                marker='o',
                linestyle='-',
                linewidth=1.5,
                markersize=4,
                color=color_shade,
                label=f'{path_type} r={radius}'
            )

    # Axis labels and title
    if all_xticks:
        ax.set_xticks(sorted(all_xticks))
    ax.set_xlabel('Spatial point')
    ax.set_ylabel('Mean Lifetime')
    ax.set_title(f'Mean Lifetime Across Spatial Transition (Radii Comparison)')

    # Legends with orange and blue shades
    homo_handles = [Line2D([0], [0], color=orange_cmap(norm(r)), lw=2, marker='o', label=f'{r}')
                    for r in radius_list]
    hetero_handles = [Line2D([0], [0], color=blue_cmap(norm(r)), lw=2, marker='o', label=f'{r}')
                    for r in radius_list]

    leg1 = ax.legend(handles=homo_handles, title='Radii (Homogeneous)', loc='upper left', frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=hetero_handles, title='Radii (Heterogeneous)', loc='upper right', frameon=False)

    plt.tight_layout()
    
    if save_plot:

        save_file_name= f"mean_lifetime_across_spatial_transition_by_radius_{leap_id}_homo_hetro_same_plot"
        save_fig(figure_save_dir, save_file_name, 'pdf', plt, transparent=True)

    plt.show()


