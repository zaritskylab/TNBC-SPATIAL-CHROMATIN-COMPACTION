import sys
import os

from utils.data_func import*
from abc import ABC, abstractmethod
import numpy as np
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
from PIL import Image
import config.const as const
import numpy as np
import pandas as pd
from tifffile import tifffile
from skimage import measure, io
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


 
class SegmentationFactory:
    """
    Factory class for creating segmentation model instances.
    Supported models: StarDist, (placeholders for CellPose, DeepCell)
    """

    @staticmethod
    def create_segmentation_model(model_type):
        """
        Instantiate the specified segmentation model.

        Parameters
        ----------
        model_type : str
            The name of the model to create ("StarDist", "CellPose", or "DeepCell")

        Returns
        -------
        SegmentationClass
            An instance of the selected segmentation class
         """
        
        if model_type == "CellPose":
            pass
        elif model_type == "DeepCell":
            pass
        elif model_type == "StarDist":
            return StarDistSeg()
        else:
            raise ValueError("Invalid segmentation model type")


class SegmentationClass(ABC):
    """
    Abstract base class for all segmentation model implementations.
    """

    @abstractmethod
    def perform_segmentation(self, *args, **kwargs):
        """
        Abstract method to perform segmentation on an input image.
        """
        pass



class CellPoseSeg(SegmentationClass):
    def __init__(self):
        # self.model = models.Cellpose(model_type="nuclei")
        pass
 
    def perform_segmentation(self, *args, **kwargs):
        pass


class DeepCellSeg(SegmentationClass):

    def __init__(self):
        # self.model = NuclearSegmentation()
        pass
 
    def perform_segmentation(self, *args, **kwargs):
        pass



class StarDistSeg(SegmentationClass):
    """
    Implements segmentation using a pretrained StarDist2D model.
    """

    def __init__(self):
        self.model = StarDist2D.from_pretrained('2D_versatile_fluo')

    def perform_segmentation(self, *args, **kwargs):
        """
        Perform instance segmentation using StarDist on a fluorescence image.

        Parameters
        ----------
        image : np.ndarray
            Input 2D fluorescence image
        num_leap : str
            LEAP number for naming output files

        Returns
        -------
        np.ndarray
            Labeled segmentation mask
        """

        # Initialize StarDist model
        print("Start StarDist segmentation")
        img = kwargs["image"]
        leap_num = kwargs["num_leap"]

        nms_thresh = 0.6
        scale = 3.0

        # set scale based on image dimensions
        height, width = img.shape[:2]
        print(f"image size is: height= {height}, width= {width}")

        params = {"scale": scale, "nms_thresh": nms_thresh}
        print(f'params: {params.items()}')

        # Perform segmentation
        segment_results = self.model.predict_instances(normalize(img), scale=scale, nms_thresh=nms_thresh,n_tiles=self.model._guess_n_tiles(normalize(img)))
        

        segment_label = segment_results[0]

        dir_leap_segment = const.seg_dir
        os.makedirs(dir_leap_segment, exist_ok=True)

        # Save the segment_label and the segmentation details
        im_tif = Image.fromarray(segment_label)
        im_tif.save(rf'{dir_leap_segment}/LEAP{leap_num}_segmentation_labels.tif')

        print("Finish StarDist segmentation")
        return segment_label
    

def create_segmentation(arg):
    """
    Applies segmentation on a fluorescence image using StarDist.

    Parameters
    ----------
    arg : tuple
        A tuple containing:
            - num_leap (str): LEAP number for identifying the sample
            - fluor_file_path (str): Path to the .tif fluorescence image
            - process_num (str): Index for tracking the segmentaion processing
    """
        
    num_leap = arg[0]
    fluor_file_path = arg[1]
    process_num = arg[2]

    print(f"Starting with {process_num}")

    seg_factory = SegmentationFactory()
    seg_model = seg_factory.create_segmentation_model("StarDist")
    print(fluor_file_path)     
    # Check if the path is a file (not a subdirectory)
    if os.path.isfile(fluor_file_path):
        print(f"start with {fluor_file_path}")
        img = tifffile.imread(fluor_file_path)
        try:
            seg_model.perform_segmentation(image=img, num_leap=num_leap)
        except Exception as e:
            print(f"[ERROR] Segmentation failed for LEAP {num_leap} (process {process_num}): {e}")
        else:
            print(f"Finished segmentation for: {fluor_file_path}, process num: {process_num}")

    else:
        print(f"[WARNING] File not found: {fluor_file_path}")


def run_segmentation(sample=False):
    """
    Run the segmentation process for LEAP samples that lack existing segmentation labels.

    This function identifies LEAP samples requiring segmentation by:
    - Extracting LEAP numbers from the metadata file
    - Verifying the existence of corresponding fluorescent image files
    - Checking whether segmentation label files already exist
    - Calling the segmentation function only for samples that are missing labels
    """

    tup_list = []
    _, leaps_list, _ = extract_core_resection_from_tnbc(const.rcb_file, for_prediction=False)
    if sample:
        leaps_list=leaps_list[:1]
    print(f"Total LEAP samples: {len(leaps_list)}")

    for idx, leap_number in enumerate(leaps_list):
        fluor_file_name = f'LEAP{leap_number}_fluorescent.tif'
        fluor_img_path = os.path.join(const.fluorescent_dir, fluor_file_name)

        if os.path.isfile(fluor_img_path):
            seg_file_name = f'LEAP{leap_number}_segmentation_labels.tif'
            seg_file_path = os.path.join(const.seg_dir, seg_file_name)
            print(seg_file_path)

            if os.path.exists(seg_file_path):
                print(f"Segmentation already exists: {seg_file_path}")
                continue
            else:
                print(f"LEAP {leap_number} requires segmentation")

            tup_list.append((leap_number, fluor_img_path, idx))

    print(f"Total samples needing segmentation: {len(tup_list)}")

    for tup in tup_list:
        create_segmentation(tup)


def remove_outliners_labels(seg_image, seg_qc_path):
    """
    Removes segmentation labels that are outliers based on area thresholds.

    Outliers are defined as:
    - Area < 5th percentile (too small)
    - Area > 100 (too large)

    Parameters
    ----------
    seg_image : np.ndarray
        A labeled 2D segmentation mask where each region has a unique label.
    seg_qc_path : str
        File path to save the cleaned segmentation mask.

    Returns
    -------
    None
        The cleaned segmentation mask is saved to the specified path.
    """
    # Extract region properties (area per label)
    props_df = measure.regionprops_table(seg_image, properties=['label', 'area'])
    feature_df = pd.DataFrame(props_df)

    # Define outlier thresholds
    lower_threshold = np.percentile(feature_df['area'], 5)
    upper_threshold = 100  # fixed based on domain knowledge

    # Keep only labels within the size range
    filtered_df = feature_df[
        (feature_df['area'] >= lower_threshold) & 
        (feature_df['area'] <= upper_threshold)
    ]

    # Create a mask for valid labels
    mask = np.isin(seg_image, filtered_df['label'])

    # Generate filtered segmentation mask
    filtered_label = seg_image.copy()
    filtered_label[~mask] = 0  # Remove outlier regions

    # Save cleaned segmentation
    io.imsave(seg_qc_path, filtered_label)
    print(f"Saved cleaned segmentation to: {seg_qc_path}")


def run_segmentation_qc(sample=False):
    """
    Perform quality control (QC) on segmentation label files for all LEAP samples.

    This function:
    - Iterates through LEAP numbers extracted from the metadata
    - Checks for the presence of segmentation label files
    - Applies `remove_outliners_labels` to generate a QC version if it does not already exist
    """

    os.makedirs(const.seg_after_qc_dir, exist_ok=True)
    _, leaps_list, _ = extract_core_resection_from_tnbc(const.rcb_file, for_prediction=False)
    if sample:
        leaps_list=leaps_list[:1]

    for leap_number in leaps_list:
        leap_name = f"LEAP{leap_number}"
        for file_name in os.listdir(const.seg_dir):
            if file_name == f'{leap_name}_segmentation_labels.tif':
                seg_file_name = file_name
                qc_file_path = os.path.join(const.seg_after_qc_dir, f'{leap_name}_segmentation_labels_qc.tif')

                if os.path.exists(qc_file_path):
                    print(f"QC file already exists: {qc_file_path}")
                else:
                    label_file_path = os.path.join(const.seg_dir, seg_file_name)
                    seg_image = io.imread(label_file_path)
                    remove_outliners_labels(seg_image, qc_file_path)
                    print(f"QC completed for: {leap_name}")

        print(f"Finished with LEAP {leap_number}")



################### Segmentation results validation ###################

# Function to process a single leap number
def process_leap_extract_area(leap_number, seg_dir_path):
    """
    Load the segmentation mask for a specific LEAP sample and return the area of all segmented regions.

    Parameters
    ----------
    leap_number : str
        The LEAP ID number.
    seg_dir_path : str
        Directory path where segmentation files are stored.

    Returns
    -------
    area_df : pd.DataFrame
        DataFrame containing region area measurements.
    """

    leap_name = f"LEAP{leap_number}"
    leap_file_name = f'{leap_name}_segmentation_labels.tif'

    # Find the correct label file
    label_file_name = next(
        (file_name for file_name in os.listdir(seg_dir_path) if file_name == leap_file_name),
        None
    )
    if not label_file_name:
        raise FileNotFoundError(f"No segmentation file found for LEAP {leap_number}")

    label_file_path = os.path.join(seg_dir_path, label_file_name)
    seg_image = io.imread(label_file_path)

    # Calculate region properties for the labels
    props_df = measure.regionprops_table(seg_image, properties=['label','area'])
    area_df = pd.DataFrame(props_df)

    return area_df



def perecent_outliners(seg_image):
    """
    Calculate the number and percentage of outlier segments in a segmentation mask.

    Outliers are defined as:
    - Area less than the 5th percentile (too small)
    - Area greater than 100 (too large)

    Parameters
    ----------
    seg_image : np.ndarray
        Labeled segmentation mask (2D array).

    Returns
    -------
    outliers_lower : int
        Number of segments smaller than the 5th percentile.
    outliers_upper : int
        Number of segments larger than 100.
    total_cell_count : int
        Total number of segments in the image.
    lower_area_list : pd.Series
        Area values of the small segments.
    upper_area_list : pd.Series
        Area values of the large segments.
    """
    
    # Calculate region properties for the labels
    props_df = measure.regionprops_table(seg_image, properties=['label','area'])
    feature_df = pd.DataFrame(props_df)

    total_cell_amount = len(feature_df['area'])
    # Calculate the 5th and 95th percentiles of the 'area' column
    lower_num = np.percentile(feature_df['area'], 5)
    upper_num = 100

    outliners_lower = len(feature_df[feature_df['area'] <= lower_num])
    outliners_up = len(feature_df[feature_df['area'] >= upper_num])
    upper_area_list = feature_df[feature_df['area'] >= upper_num]['area']
    lower_area_list = feature_df[feature_df['area'] <= lower_num]['area']
    
    return outliners_lower, outliners_up, total_cell_amount, lower_area_list, upper_area_list



# Function to process a single leap number
def process_leap_qc(leap_number, seg_dir_path):
    """
    Process a single LEAP segmentation mask and calculate outlier statistics.

    Parameters
    ----------
    leap_number : str or int
        LEAP ID for the sample to be processed.
    seg_dir_path : str
        Directory containing segmentation mask files.

    Returns
    -------
    lower_percent : float
        Fraction of segments below the 5th percentile.
    upper_percent : float
        Fraction of segments above the upper threshold (100).
    lower_count : int
        Number of segments below the lower threshold.
    upper_count : int
        Number of segments above the upper threshold.
    lower_area_list : pd.Series
        Area values of small outlier segments.
    upper_area_list : pd.Series
        Area values of large outlier segments.
    """
        
    leap_name = f"LEAP{leap_number}"
    leap_file_name = f'{leap_name}_segmentation_labels.tif'

    # Find the correct label file
    label_file_name = next(
        (file_name for file_name in os.listdir(seg_dir_path) if file_name == leap_file_name),
        None
    )
    if not label_file_name:
        raise FileNotFoundError(f"No segmentation file found for LEAP {leap_number}")

    label_file_path = os.path.join(seg_dir_path, label_file_name)
    seg_image = io.imread(label_file_path)
    lower, upper, total, l_area_list, u_area_list = perecent_outliners(seg_image)
    lower_percent = lower / total
    upper_percent = upper / total

    # Print results for this leap
    print(f"leap {leap_number}, lower num: {lower}, lower percent: {lower_percent}, upper num: {upper}, upper percent: {upper_percent}")

    return leap_name, lower_percent, upper_percent, lower, upper, l_area_list, u_area_list


if __name__ == '__main__':
    run_segmentation()
    run_segmentation_qc()
