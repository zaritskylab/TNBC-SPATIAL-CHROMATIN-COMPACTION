from flim_analysis.feature_extraction.extract_features import *
import config.params as params
import config.const as const
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create lifetime distributions and medians')
    parser.add_argument('sample_type', choices=['core', 'resection', 'patch'],
                        nargs='?', default='core', help="Type of sample to process. Choices: 'core', 'resection', 'patch'. Default: 'core'.")

    parser.add_argument('--max-val',   type=float, default=13,
                        help="Maximum lifetime value for binning (used in lifetime distribution calculations). Default: 13.")
    parser.add_argument('--bin-range', type=float, default=0.73,
                        help="Range of each bin in the lifetime distribution histogram. Smaller values increase the number of bins. Default: 0.73.")

    parser.add_argument('--patch-size', type=int,   default=1500,
                        help="Size (in pixels) of the patch used in patch-level processing. Only used if sample_type is 'patch'. Default: 1500.")
    parser.add_argument('--overlap',    type=float, default=0.75,
                        help="Fractional overlap between adjacent patches (0.0 to 1.0). Only used if sample_type is 'patch'. Default: 0.75.")

    args = parser.parse_args()

    # ---------- CORE ----------
    if args.sample_type == 'core':

        sample_type = 'core'
        max_range = args.max_val
        bin_range = args.bin_range
        feature_file_name = "FLIM_features_full_tissue.csv"
        build_lifetime_distribution_full_tissue(sample_type, max_range, bin_range, feature_file_name)

        aggregate_median_features_by_leap(const.FULL_TISSUE_DIR, 'core')

    # ---------- RESECTION ----------
    if args.sample_type == 'resection':
        aggregate_median_features_by_leap(const.FULL_TISSUE_DIR, 'resection')

    # ---------- PATCH ----------
    if args.sample_type == 'patch':
        patch_size = args.patch_size
        patch_overlap = args.overlap

        build_lifetime_distribution_patch(
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            max_range=args.max_val,  
            bin_range=args.bin_range
        )