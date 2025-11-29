import os
import argparse
import pandas as pd
import config.params as params
import config.const as const
from pathlib import Path
from flim_analysis.distribution_classification.binary_classifier_trainer import BinaryClassifierTrainer

# Set primary seed for reproducibility
PRIMARY_SEED = params.primary_seed


def load_metadata(metadata_path: Path) -> pd.DataFrame:
  """
  Load and filter metadata from a CSV file.

  Args:
    metadata_path (Path): The path to the metadata CSV file.

  Returns:
    pd.DataFrame: A DataFrame containing filtered metadata.
  
  """
  # Load metadata
  metadata_df = pd.read_csv(metadata_path)

  # Filter non pre test force trial cores
  filtered_df = metadata_df[(
      (
          (metadata_df.force_trial == "v") & (
              ~metadata_df.force_trial_cycle.
              isin(["Post-cycle 1.1", "Post-cycle 2.2"])
          )
      ) | (metadata_df.force_trial != "v")
  ) & ~(metadata_df.preservation_type.str.lower() == "frozen")]

  # Filter for FFPE core samples
  filtered_df = filtered_df[(filtered_df.sample_type == "core")]
  return filtered_df


def preprocess_eccentricity_df(
    metadata_df: pd.DataFrame, eccentricity_distribution: pd.DataFrame
) -> pd.DataFrame:
  """
  Preprocess the eccentricity DataFrame.

  Args:
    metadata_df (pd.DataFrame): The metadata DataFrame.

  Returns:
    pd.DataFrame: The preprocessed eccentricity DataFrame.

  """
  eccentricity_distribution["leap_id"] = eccentricity_distribution["leap_ID"].map(
      lambda x: f"Leap{str(x).zfill(3)}"
  )
  eccentricity_distribution["response"] = eccentricity_distribution["leap_id"].map(
      lambda leap_id: metadata_df[metadata_df["leap_id"] == leap_id]["response"]
      .values[0].lower()
  )
  eccentricity_distribution["response"] = eccentricity_distribution["response"].map(
      lambda x: "non responder" if x == "non-responder" else "responder"
  )
  eccentricity_distribution["analysis_id"] = eccentricity_distribution["leap_id"].map(
      lambda leap_id: metadata_df[metadata_df["leap_id"] == leap_id][
          "analysis_id"].values[0]
  )
  eccentricity_distribution.drop(columns=["categories", "leap_ID"], inplace=True)
  return eccentricity_distribution


def rename_eccentricity_columns(
    eccentricity_distribution: pd.DataFrame, eccentricity_columns: list
) -> tuple:
  """
  Rename eccentricity columns in the DataFrame.

  Args:
    eccentricity_distribution (pd.DataFrame): The eccentricity distribution DataFrame.
    eccentricity_columns (list): List of eccentricity columns to rename.

  Returns:
    tuple: A tuple containing the modified DataFrame and the new column names.
  
  """
  new_eccentricity_columns = []
  for col in eccentricity_columns:
    new_col_name = col.replace("eccentricity_", "")
    bin_start, bin_end = new_col_name.split("-")
    bin_start = round(float(bin_start), 3)
    bin_end = round(float(bin_end), 3)
    new_col_name = f"{bin_start}-{bin_end}"
    eccentricity_distribution.rename(columns={col: new_col_name}, inplace=True)
    new_eccentricity_columns.append(new_col_name)
  return eccentricity_distribution, new_eccentricity_columns


if __name__ == '__main__':
  # Parse command-line arguments
  parser = argparse.ArgumentParser(
      description="FLIM tissue wise classification pipeline using eccentricity features"
  )
  parser.add_argument(
      '--n_seeds', type=int, default=100, help="Number of seeds to run"
  )
  parser.add_argument(
      '--n_permutations', type=int, default=1000,
      help="Number of permutations for the analysis"
  )
  parser.add_argument(
      "--dist_csv_name", type=str, default=
      "features_eccentricity_distribution_data_max_val_1.0_bins_amount_18_bin_range_0.056.csv",
      help="Name of the CSV file containing eccentricity distribution features"
  )
  args = parser.parse_args()

  # Print message
  print(f"Running FLIM tissue-wise classification pipeline using eccentricity features...")

  # Load metadata
  metadata_df = load_metadata(const.RCB_FILE)

  # Load eccentricity distribution data
  distribution_csv_path = Path(
      const.FULL_TISSUE_DIR
  ) / f'core/{args.dist_csv_name}'
  if not distribution_csv_path.exists():
    raise FileNotFoundError(
        f"Distribution CSV file not found: {distribution_csv_path}"
    )
  eccentricity_distribution = pd.read_csv(distribution_csv_path)
  eccentricity_distribution = preprocess_eccentricity_df(
      metadata_df, eccentricity_distribution
  )

  # Get eccentricity columns
  not_eccentricity_columns = [
      col for col in eccentricity_distribution.columns if 'eccentricity' not in col
  ]

  eccentricity_columns = [
      col for col in eccentricity_distribution.columns if 'eccentricity' in col
  ]

  # Rename eccentricity columns
  eccentricity_distribution, eccentricity_columns = rename_eccentricity_columns(
      eccentricity_distribution, eccentricity_columns
  )

  # Create data for classification
  X = eccentricity_distribution.drop(columns=not_eccentricity_columns).values
  y = (eccentricity_distribution["response"].values != "non responder").astype(int)
  leap_ids = eccentricity_distribution["leap_id"].values
  group_ids = eccentricity_distribution["analysis_id"].values
  feature_names = eccentricity_distribution.drop(
      columns=not_eccentricity_columns
  ).columns

  # Create output directory
  cv_strategy = "loocv"
  save_dir = (
      Path(const.DISTRIBUTION_RESULTS_BASE_DIR) / "classification_results_using_eccentricity" /
      "tissue_wise" / distribution_csv_path.stem / cv_strategy
  )
  save_dir.mkdir(parents=True, exist_ok=True)

  # Run classification
  trainer = BinaryClassifierTrainer(
      X, y, patient_ids=group_ids, sample_ids=leap_ids,
      feature_names=feature_names, num_seeds=args.n_seeds,
      num_permutations=args.n_permutations, model_type="xgboost",
      output_dir=save_dir, tune_hyperparameters=True, primary_seed=PRIMARY_SEED,
      cv_strategy=cv_strategy, optuna_cv_strategy='stratified_kfold',
      optuna_cv_n_splits=5
  ).run()
