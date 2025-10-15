from flim_analysis.feature_extraction.extract_features import *
import config.params as params
import config.const as const

if __name__ == '__main__':
    for bin_amount, bin_range in params.lifetime_distribution_params.items():
        sample_type = 'core'
        max_range = params.lifetime_distribution_max_val
        feature_file_name = "FLIM_features_full_tissue.csv"

        build_lifetime_distribution_full_tissue(sample_type, max_range, bin_range, feature_file_name)

    aggregate_median_features_by_leap(const.full_tissue_dir, 'core')