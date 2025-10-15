from flim_analysis.feature_extraction.extract_features import *
import sys

if __name__ == '__main__':
    sample_type = sys.argv[1]


    if sample_type == 'core':
        create_all_feature_core_full_tissue_df()

    elif sample_type == 'resection':
        create_all_feature_resection_full_tissue_df()




