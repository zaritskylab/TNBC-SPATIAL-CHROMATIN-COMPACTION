# %%
import os

import config.const as const
from utils.auxiliary_func import*
from utils.data_func import*


'''
FLIM Data Preprocessing — Channel Splitting

In this step of the TNBC FLIM tissue analysis pipeline, we preprocess the raw `.tif` images by **splitting two-channel FLIM images into separate files**:
    - **Fluorescence intensity channel**
    - **Fluorescence lifetime channel**

Each `.tif` file from the specified directory contains two interleaved image channels.  
The function `split_2_channel_to_1` is applied to **all `.tif` files** in the directory to perform this separation, storing the results in predefined output folders.
'''


def extract_leap_number_and_threshold(file_name):
 
    # Define the regex pattern to extract leap number and threshold
    pattern = r"LEAP(\d+)_.*?(\d+)countthreshold\.tif"

    # Use re.search to find the pattern in the file path
    match = re.search(pattern, file_name)

    # Check if the pattern is found
    if match:
        # Extract leap number and threshold from the matched groups
        leap_number = match.group(1)
        threshold = match.group(2)

        print("Leap Number:", leap_number)
        print("Threshold:", threshold)

        return leap_number, threshold
    

def extract_leap_number(file_name):
    pattern = r"LEAP(\d+)"

    match = re.search(pattern, file_name)

    if match:
        leap_number = match.group(1)
        print("Leap Number:", leap_number)
        return leap_number

    print("No leap number found.")
    return None


def split_2_channel_to_1(directory, file_name):

    leap_num = extract_leap_number(file_name)

    os.makedirs(const.FLUORESCENT_DIR, exist_ok=True)
    print(f"LEAP {leap_num} - start split to fluorescent and flim channels\n")
    fluorescent_path = os.path.join(const.FLUORESCENT_DIR, f'LEAP{leap_num}_fluorescent.tif')
    if os.path.exists(fluorescent_path):
        print(f"File exists: {fluorescent_path}\n")
        return
    
    os.makedirs(const.FLIM_DIR, exist_ok=True)
    flim_path = os.path.join(const.FLIM_DIR, f'LEAP{leap_num}_flim.tif')

    channels_file_path = os.path.join(directory, file_name)
    img = io.imread(channels_file_path)
    # print(f"\nflouroscent channel: \n{img[0]}\n")
    # print(f"\nflim channel: \n{img[1]}\n")
    file_name=file_name[:-4]



    im_tif = Image.fromarray(img[0])
    im_tif.save(fluorescent_path)


    im_tif = Image.fromarray(img[1])
    im_tif.save(flim_path)
    print(f"LEAP {leap_num} - finish split 2 channels\n")



def run_preprocess():
    
    RAW_DATA_DIR = const.RAW_DATA_DIR
    print(f"Files process start from {RAW_DATA_DIR}.")

    _, leaps_list, _ = extract_core_resection_from_tnbc(const.RCB_FILE)
    print(f"Loaded {len(leaps_list)} LEAP(s) from RCB file.")


    # List all .tif files in the directory
    tif_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.tif')]

    for leap_id in leaps_list:
        # Apply the function to each file
        for tif_file in tif_files:
            if leap_id in tif_file:
                split_2_channel_to_1(RAW_DATA_DIR, tif_file)
        
        print(f"Preprocessing complete: {len(tif_files)} files processed from {RAW_DATA_DIR}.")


if __name__ == '__main__':
    run_preprocess()