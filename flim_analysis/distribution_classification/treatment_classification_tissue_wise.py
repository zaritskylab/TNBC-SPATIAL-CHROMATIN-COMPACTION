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


def preprocess_lifetime_df(
    metadata_df: pd.DataFrame, lifetime_distribution: pd.DataFrame
) -> pd.DataFrame:
  """
  Preprocess the lifetime DataFrame.

  Args:
    metadata_df (pd.DataFrame): The metadata DataFrame.

  Returns:
    pd.DataFrame: The preprocessed lifetime DataFrame.
  
  """
  lifetime_distribution["leap_id"] = lifetime_distribution["leap_ID"].map(
      lambda x: f"Leap{str(x).zfill(3)}"
  )
  lifetime_distribution["response"] = lifetime_distribution["leap_id"].map(
      lambda leap_id: metadata_df[metadata_df["leap_id"] == leap_id]["response"]
      .values[0].lower()
  )
  lifetime_distribution["response"] = lifetime_distribution["response"].map(
      lambda x: "non responder" if x == "non-responder" else "responder"
  )
  lifetime_distribution["analysis_id"] = lifetime_distribution["leap_id"].map(
      lambda leap_id: metadata_df[metadata_df["leap_id"] == leap_id][
          "analysis_id"].values[0]
  )
  lifetime_distribution.drop(columns=["categories", "leap_ID"], inplace=True)
  return lifetime_distribution


def rename_lifetime_columns(
    lifetime_distribution: pd.DataFrame, lifetime_columns: list
) -> tuple:
  """
  Rename lifetime columns in the DataFrame.

  Args:
    lifetime_distribution (pd.DataFrame): The lifetime distribution DataFrame.
    lifetime_columns (list): List of lifetime columns to rename.

  Returns:
    tuple: A tuple containing the modified DataFrame and the new column names.
  
  """
  new_lifetime_columns = []
  for col in lifetime_columns:
    new_col_name = col.replace("lifetime_mean_", "")
    bin_start, bin_end = new_col_name.split("-")
    bin_start = round(float(bin_start), 3)
    bin_end = round(float(bin_end), 3)
    new_col_name = f"{bin_start}-{bin_end}"
    lifetime_distribution.rename(columns={col: new_col_name}, inplace=True)
    new_lifetime_columns.append(new_col_name)
  return lifetime_distribution, new_lifetime_columns


if __name__ == '__main__':
  # Parse command-line arguments
  parser = argparse.ArgumentParser(
      description="FLIM tissue wise classification pipeline"
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
      "features_lifetime_distribution_data_max_val_13_bins_amount_18_bin_range_0.73.csv",
      help="Name of the CSV file containing lifetime distribution features"
  )
  args = parser.parse_args()

  # Print message
  print(f"Running FLIM tissue-wise classification pipeline...")

  # Load metadata
  metadata_df = load_metadata(const.rcb_file)

  # Load lifetime distribution data
  distribution_csv_path = Path(
      const.full_tissue_dir
  ) / f'core/{args.dist_csv_name}'
  if not distribution_csv_path.exists():
    raise FileNotFoundError(
        f"Distribution CSV file not found: {distribution_csv_path}"
    )
  lifetime_distribution = pd.read_csv(distribution_csv_path)
  lifetime_distribution = preprocess_lifetime_df(
      metadata_df, lifetime_distribution
  )

  # Get lifetime columns
  not_lifetime_columns = [
      col for col in lifetime_distribution.columns if 'lifetime_mean' not in col
  ]

  lifetime_columns = [
      col for col in lifetime_distribution.columns if 'lifetime_mean' in col
  ]

  # Rename lifetime columns
  lifetime_distribution, lifetime_columns = rename_lifetime_columns(
      lifetime_distribution, lifetime_columns
  )

  # Create data for classification
  X = lifetime_distribution.drop(columns=not_lifetime_columns).values
  y = (lifetime_distribution["response"].values != "non responder").astype(int)
  leap_ids = lifetime_distribution["leap_id"].values
  group_ids = lifetime_distribution["analysis_id"].values
  feature_names = lifetime_distribution.drop(
      columns=not_lifetime_columns
  ).columns

  # Create output directory
  cv_strategy = "loocv"
  save_dir = (
      Path(const.distribution_results_base_dir) / "classification_results" /
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
