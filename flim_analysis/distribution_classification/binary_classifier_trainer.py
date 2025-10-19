"""
This module contains the BinaryClassifierTrainer class for training a binary
classifier using cross-validation end to end including hyperparameter tuning,
permutation tests for model evaluation, SHAP, plotting and more.

"""

import copy
import json
import warnings
import random
import optuna
import shap
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import config.params as params
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Literal, Callable, Iterator, Tuple
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import (
    LeaveOneGroupOut, GroupKFold, StratifiedGroupKFold
)
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from optuna.trial import Trial
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
from flim_analysis.distribution_classification import figure_customizer as fc
from flim_analysis.distribution_classification.figure_customizer import DEFAULT_CONTINUOUS_CMAP, DEFAULT_DISCRETE_CMAP

# Suppress the trial-specific messages that Optuna prints
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set Seaborn style
sns.set_context("notebook", font_scale=1.25)

# set primary seed for reproducibility
PRIMARY_SEED = params.primary_seed


class ModelRegistry:
  """
  A registry for managing predefined models and hyperparameter spaces.
  """

  def __init__(self):
    self.use_gpu = torch.cuda.is_available()

    self.model_definitions = {
        'xgboost': {
            'model': XGBClassifier,
            'hyperparameter_space': self._xgboost_space,
        }
    }

  def _xgboost_space(self, trial) -> dict:
    return {
        'max_depth': trial.suggest_int('max_depth', 3, 10), 'learning_rate':
        trial.suggest_float('learning_rate', 0.01, 0.3, log=True), 'subsample':
        trial.suggest_float('subsample', 0.5, 1.0), 'colsample_bytree':
        trial.suggest_float('colsample_bytree', 0.5,
                            1.0), 'gamma': trial.suggest_float('gamma', 0, 10),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500), 'reg_alpha':
        trial.suggest_float('reg_alpha', 1, 10, log=True), 'reg_lambda':
        trial.suggest_float('reg_lambda', 1, 10, log=True), 'scale_pos_weight':
        trial.suggest_float('scale_pos_weight', 1, 10)
    }

  def get_model_and_space(self, model_type: str) -> tuple:
    """
    Method to retrieve the model and hyperparameter space for a given model 
        type.
    
    Args:
      model_type (str): the type of the model to retrieve.
        
    Returns:
      tuple: a tuple containing the model class and its hyperparameter space 
        function.
    
    Raises:
      ValueError: If the model type is not supported.

    """
    if model_type not in self.model_definitions:
      raise ValueError(f"Unsupported model type: {model_type}")
    return self.model_definitions[model_type]['model'], self.model_definitions[
        model_type]['hyperparameter_space']

  def register_custom_model(
      self, model_name: str, model_class: BaseEstimator,
      hyperparameter_space_func: Callable
  ) -> None:
    """
    Register a custom model and its hyperparameter space.
    
    Args:
      model_name (str): The name of the custom model.
      model_class: The class of the custom model.
      hyperparameter_space_func: A function defining the hyperparameter space 
          for the model.
  
    """
    self.model_definitions[model_name] = {
        'model': model_class,
        'hyperparameter_space': hyperparameter_space_func,
    }


class BinaryClassifierTrainer:
  """ 
  A class to train a binary classifier using Cross-Validation end to end 
      including hyperparameter tuning, permutation tests for model evaluation, 
      SHAP, plotting and more.

  """

  def __init__(
      self, X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray,
      sample_ids: np.ndarray, feature_names: np.ndarray, num_seeds: int,
      num_permutations: int, model_type: str, output_dir: Path,
      tune_hyperparameters: bool = True,
      tune_permutation_hyperparameters: bool = False,
      primary_seed: int = PRIMARY_SEED, n_trials=50, n_jobs: int = -1,
      cv_strategy: Literal["loocv", "kfold",
                           "stratified_kfold"] = "loocv", cv_n_splits: int = 5,
      optuna_cv_strategy: Literal["loocv", "kfold",
                                  "stratified_kfold"] = "stratified_kfold",
      optuna_cv_n_splits: int = 5,
      preprocessing_pipeline: Optional[Pipeline] = None,
      calibrate_proba: bool = False, custom_model_name: Optional[str] = None,
      custom_model: Optional[BaseEstimator] = None,
      custom_hyperparameter_space: Optional[Callable] = None,
      use_gpu: bool = False
  ):
    """
    Initialize the BinaryClassifierTrainer class.

    Args:
      X (np.ndarray): Feature matrix of shape (n_samples, n_features).
      y (np.ndarray): Target vector of shape (n_samples,).
      patient_ids (np.ndarray): Array of patient IDs.
      sample_ids (np.ndarray): Array of sample IDs.
      feature_names (np.ndarray): Array of feature names.
      num_seeds (int): Number of seeds for evaluation.
      num_permutations (int): Number of permutations for permutation tests.
      model_type (str): Type of model to use for training.
      output_dir (Path): Directory to save the results.
      tune_hyperparameters (bool, optional): Flag to tune hyperparameters using
          Optuna. Defaults to True.
      tune_permutation_hyperparameters (bool, optional): Flag to tune 
          hyperparameters for permutation tests using Optuna. Defaults to False.
      primary_seed (int, optional): Random seed for reproducibility. Defaults
          to PRIMARY_SEED.
      n_trials (int, optional): Number of trials for hyperparameter tuning.
          Defaults to 50.
      n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1. 
      cv_strategy (Literal["loocv", "kfold", "stratified_kfold"]): 
          Cross-validation strategy. Defaults to "loocv".
      cv_n_splits (int, optional): Number of splits for cross-validation.
      optuna_cv_strategy (Literal["loocv", "kfold", "stratified_kfold"]): 
          Cross-validation strategy for Optuna. Defaults to "stratified_kfold".
      optuna_cv_n_splits (int, optional): Number of splits for Optuna 
          optimization.
      preprocessing_pipeline (Pipeline, optional): Preprocessing
          pipeline to use before the model. Defaults to None.
      calibrate_proba (bool, optional): Flag to calibrate probabilities using
          Platt scaling. Defaults to False.
      custom_model (BaseEstimator, optional): Custom model to use for
          training. Defaults to None.
      custom_hyperparameter_space (Callable, optional): Custom
          hyperparameter space for the custom model. Defaults to None.
      use_gpu (bool, optional): Flag to use GPU for training. Defaults to False.

    Raises:
      ValueError: If the model type is invalid, the cross-validation strategy is
          invalid, the number of splits is less than 2, or the Optuna
          cross-validation strategy is invalid. 

    """
    # Check if the cross-validation strategy is valid
    cv_strategies = ['loocv', 'kfold', 'stratified_kfold']
    if (cv_strategy
        not in cv_strategies) or (optuna_cv_strategy not in cv_strategies):
      raise ValueError(
          f"Invalid cv_strategy '{cv_strategy}'. Choose either 'loocv', 'kfold'"
          " or 'stratified_kfold'."
      )

    # Check if the cross-validation number of splits is valid
    if cv_n_splits < 2:
      raise ValueError(f"cv_n_splits must be >= 2, got {cv_n_splits}.")
    if optuna_cv_n_splits < 2:
      raise ValueError(
          f"optuna_cv_n_splits must be >= 2, got {optuna_cv_n_splits}."
      )

    # Initialize ModelRegistry and check if the model type is valid
    self.model_registry = ModelRegistry()

    # Register a custom model if provided
    if custom_model_name and custom_model and custom_hyperparameter_space:
      self.model_registry.register_custom_model(
          custom_model_name, custom_model, custom_hyperparameter_space
      )
      self.model_type = custom_model_name

    # Set the attributes
    self.use_gpu = use_gpu and torch.cuda.is_available()
    self.X = X
    self.y = y
    self.patient_ids = patient_ids
    self.sample_ids = sample_ids
    self.feature_names = feature_names
    self.num_seeds = num_seeds
    self.num_permutations = num_permutations
    self.model_type = (
        model_type if custom_model_name is None else custom_model_name
    )
    self.tune_hyperparameters = tune_hyperparameters
    self.tune_permutation_hyperparameters = tune_permutation_hyperparameters
    self.primary_seed = primary_seed
    self.n_trials = n_trials
    self.n_jobs = n_jobs if not self.use_gpu else 1
    self.output_dir = output_dir
    self.cv_strategy = cv_strategy
    self.cv_n_splits = cv_n_splits
    self.optuna_cv_strategy = optuna_cv_strategy
    self.optuna_cv_n_splits = optuna_cv_n_splits
    self.preprocessing_pipeline = preprocessing_pipeline
    self.calibrate_proba = calibrate_proba

    # Set the primary seed for reproducibility
    np.random.seed(self.primary_seed)
    random.seed(self.primary_seed)

    # Create the output directory if it does not exist
    self.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate multiple seeds for evaluation
    self.evaluation_seeds = [self.primary_seed] + [
        int(i) for i in
        np.random.choice(range(10000), size=self.num_seeds - 1, replace=False)
    ]

    # Generate multiple seeds for permutation
    self.permutation_seeds = [self.primary_seed] + [
        int(i) for i in np.random.
        choice(range(10000), size=self.num_permutations - 1, replace=False)
    ]

  def update_pipeline_seed(self, pipeline: Pipeline, seed: int) -> Pipeline:
    """Method to update the random state of each step in the pipeline.

    Args:
      pipeline (Pipeline): scikit-learn pipeline to update.
      seed (int): random seed for reproducibility.

    Returns:
      Pipeline: the updated pipeline with the random state set to the seed.

    """
    # Create a deep copy of the pipeline
    new_pipeline = copy.deepcopy(pipeline)
    # Update the random state of each step in the pipeline
    for _, step in new_pipeline.steps:
      if hasattr(step, 'random_state'):
        setattr(step, 'random_state', seed)
    return new_pipeline

  def get_cross_validation_split(
      self, X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray,
      cv_strategy: Literal["loocv", "kfold",
                           "stratified_kfold"], cv_n_splits: int, seed: int
  ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Method to get the cross-validation split for the data.

    Args:
      X (np.ndarray): the feature matrix.
      y (np.ndarray): the target vector.
      patient_ids (np.ndarray): group labels for stratification or grouping.
      cv_strategy (Literal["loocv", "kfold", "stratified_kfold"]): 
          Cross-validation strategy. Options include 'loocv', 'kfold', 
            'stratified_kfold'.
      cv_n_splits (int): Number of splits for k-fold or stratified k-fold CV.
      seed (int): Random seed for reproducibility.

    Returns:
      Iterator[Tuple[np.ndarray, np.ndarray]]: a tuple containing the training 
          and validation indices.

    """
    # Define the cross-validation strategy
    if cv_strategy == "loocv":
      cv = LeaveOneGroupOut().split(X, y, groups=patient_ids)
    elif cv_strategy == "kfold":
      # Change when xgboost supports new scikit-learn API
      #cv = GroupKFold(n_splits=cv_n_splits, shuffle=True,
      #                random_state=seed).split(X, y, groups=patient_ids)
      cv = GroupKFold(n_splits=cv_n_splits).split(X, y, groups=patient_ids)
    elif cv_strategy == "stratified_kfold":
      cv = StratifiedGroupKFold(
          n_splits=cv_n_splits, shuffle=True, random_state=seed
      ).split(X, y, groups=patient_ids)
    else:
      raise ValueError(f"Invalid cv_strategy '{cv_strategy}'.")
    # Return the cross-validation split
    return cv

  def get_model(self, params: dict, seed: int) -> Pipeline:
    """
    Get the model instance with the provided hyperparameters.

    Args:
      params (dict): Dictionary of hyperparameters for the model.
      seed (int): Random seed for reproducibility.

    Returns:
      Pipeline: The model instance with the provided hyperparameters.

    """
    # Get the model class and hyperparameter space function
    model_class, _ = self.model_registry.get_model_and_space(self.model_type)

    # Set verbose to -1 for LightGBM
    if model_class == LGBMClassifier:
      params['verbose'] = -1

    # Set the device to GPU if available
    if self.use_gpu:
      if model_class == XGBClassifier:
        params['device'] = 'cuda'
        params['tree_method'] = 'hist'
      if model_class == LGBMClassifier:
        params['device'] = 'gpu'

    # Create the model instance
    model = model_class(**params, random_state=seed)

    # Create the pipeline
    if self.preprocessing_pipeline:
      pipeline = self.update_pipeline_seed(self.preprocessing_pipeline, seed)
      pipeline = Pipeline(steps=pipeline.steps + [('model', model)])
    else:
      # Default pipeline: scaling only
      pipeline = Pipeline(steps=[('model', model)])
    return pipeline

  def objective(
      self, trial: Trial, X_train: np.ndarray, y_train: np.ndarray,
      group_ids: np.ndarray, seed: int
  ) -> float:
    """
    Objective function for Optuna to optimize hyperparameters of a model using 
        cross-validation.

    Args:
      trial (optuna.trial.Trial): A trial object that suggests hyperparameters.
      X_train (np.ndarray): Training feature matrix of shape (n_samples, 
          n_features).
      y_train (np.ndarray): Training target vector of shape (n_samples,).
      group_ids (np.ndarray): Array of group IDs used to group samples for 
          cross-validation.
      seed (int): Random seed for reproducibility.

    Returns:
      float: The mean AUC score across the cross-validation folds for the 
          suggested hyperparameters.

    """
    # Retrieve the model and hyperparameter space function
    _, hyperparameter_space_func = self.model_registry.get_model_and_space(
        self.model_type
    )

    # Get hyperparameters
    params = hyperparameter_space_func(trial)

    # Create the model
    model = self.get_model(params, seed)

    # Define cross-validation
    cv = self.get_cross_validation_split(
        X_train, y_train, group_ids, self.optuna_cv_strategy,
        self.optuna_cv_n_splits, seed
    )

    # Define predictions array
    predictions = np.zeros(y_train.shape)

    # Perform cross-validation with the suggested hyperparameters
    for train_idx, val_idx in cv:
      X_tr, X_val = X_train[train_idx].copy(), X_train[val_idx].copy()
      y_tr, _ = y_train[train_idx].copy(), y_train[val_idx].copy()
      predictions[val_idx] = self._train_and_predict(
          X_tr, X_val, y_tr, params, seed
      )
    # Calculate AUC score and return
    return roc_auc_score(y_train, predictions)

  def _get_train_test_split(
      self, test_ids: np.ndarray, y_labels: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets based on patient IDs.

    Args:
      test_ids (np.ndarray): Array of test patient IDs.
      y_labels (np.ndarray): Array of target labels.

    Returns:
      Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, 
          y_train, y_test
    
    """
    # Get the indices of the test IDs
    test_idx = np.where(np.isin(self.patient_ids, test_ids))[0]
    # Get the indices of the training IDs
    train_idx = np.where(~np.isin(self.patient_ids, test_ids))[0]
    # Get the training and test sets
    return (
        self.X[train_idx].copy(), self.X[test_idx].copy(),
        y_labels[train_idx].copy(), y_labels[test_idx].copy()
    )

  def _perform_hyperparameter_tuning(
      self, X_train: np.ndarray, y_train: np.ndarray, train_ids: np.ndarray,
      seed: int
  ) -> dict:
    """
    Perform hyperparameter tuning using Optuna.

    Args:
      X_train (np.ndarray): Training features.
      y_train (np.ndarray): Training labels.
      train_ids (np.ndarray): Training patient IDs.
      seed (int): Random seed.

    Returns:
      dict: Best hyperparameters.
    
    """
    # Create an Optuna study
    study = optuna.create_study(direction='maximize')
    # Optimize hyperparameters
    study.optimize(
        lambda trial: self.objective(trial, X_train, y_train, train_ids, seed),
        n_trials=self.n_trials, n_jobs=self.n_jobs
    )
    return study.best_params

  def _train_and_predict(
      self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray,
      best_params: dict, seed: int
  ) -> np.ndarray:
    """
    Train the model and predict probabilities on the test set.

    Args:
      X_train (np.ndarray): Training features.
      X_test (np.ndarray): Test features.
      y_train (np.ndarray): Training labels.
      best_params (dict): Model hyperparameters.
      seed (int): Random seed.

    Returns:
      np.ndarray: Predicted probabilities for the test set.
    
    """
    # Create the model
    model = self.get_model(best_params, seed)
    model.fit(X_train, y_train)
    # Apply Platt scaling using CalibratedClassifierCV with a sigmoid
    if self.calibrate_proba:
      calibrated_classifier = CalibratedClassifierCV(
          model, method='sigmoid', cv='prefit'
      )
      calibrated_classifier.fit(X_train, y_train)
      return calibrated_classifier.predict_proba(X_test)[:, 1]
    # Return the predicted probabilities
    return model.predict_proba(X_test)[:, 1]

  def _save_predictions_and_params(
      self, predictions: np.ndarray, best_parameters: dict, output_dir: Path,
      seed: int, y_true: np.ndarray = None
  ) -> None:
    """
    Save predictions and best parameters to the output directory.

    Args:
      predictions (np.ndarray): Predicted probabilities.
      best_parameters (dict): Best hyperparameters for each fold.
      output_dir (Path): Output directory.
      seed (int): Random seed.
      y_true (np.ndarray, optional): True labels. Defaults to None.
    
    """
    # Create a folder for the seed
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    # Save predictions
    pd.DataFrame(
        {
            "patient_id": self.patient_ids, "sample_id": self.sample_ids,
            "y_true": self.y if y_true is None else y_true, "y_pred":
            predictions
        }
    ).to_csv(seed_dir / "predictions.csv", index=False)
    # Save best parameters
    with open(seed_dir / "best_parameters.json", 'w') as f:
      json.dump(best_parameters, f)

  def single_seed_cv_train_predict(self, seed: int, output_dir: Path) -> None:
    """
    Perform training and prediction using a single seed with cross validation
        and Optuna for hyperparameter optimization. 

    Args:
      seed (int): Random seed for reproducibility.
      output_dir (Path): Directory where the results of the seed will be saved.

    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Define variables for predictions
    predictions = np.zeros(self.y.shape)

    # Define variables for storing the best parameters
    best_parameters = {}

    # Define cross-validation
    cv = self.get_cross_validation_split(
        self.X, self.y, self.patient_ids, self.cv_strategy, self.cv_n_splits,
        seed
    )

    # Perform Cross-Validation
    for train_idx, test_idx in cv:
      # Get the test IDs
      test_ids = np.unique(self.patient_ids[test_idx])

      # Split data into training and validation sets
      X_train, X_test, y_train, _ = self._get_train_test_split(test_ids, self.y)

      # Get key for best parameters
      key = "_".join(np.sort(np.unique(self.patient_ids[test_idx])).astype(str))

      # Perform hyperparameter tuning with Optuna
      best_params = self._perform_hyperparameter_tuning(
          X_train, y_train, self.patient_ids[train_idx], seed
      ) if self.tune_hyperparameters else {}

      # Train the model and predict probabilities
      predictions[test_idx] = self._train_and_predict(
          X_train, X_test, y_train, best_params, seed
      )

      # Store the best parameters
      best_parameters[key] = best_params

    # Save predictions and best parameters
    self._save_predictions_and_params(
        predictions, best_parameters, output_dir, seed
    )

  def permute_labels(
      self, seed: int, patient_ids: np.ndarray, labels: np.ndarray
  ) -> np.ndarray:
    """
    Permute the labels for permutation tests.

    Args:
      seed (int): Random seed.
      patient_ids (np.ndarray): Array of patient IDs.
      labels (np.ndarray): Array of labels.

    Returns:
      np.ndarray: Permuted labels.

    """
    # Create a new random number generator
    rng = np.random.default_rng(seed)
    # Get unique patient IDs and their corresponding labels
    unique_patient_ids = np.unique(patient_ids)
    unique_patient_ids_labels = np.array(
        [labels[patient_ids == pid][0] for pid in unique_patient_ids]
    )
    # Permute the labels
    permuted_unique_patient_ids_labels = unique_patient_ids_labels.copy()
    while np.array_equal(
        permuted_unique_patient_ids_labels, unique_patient_ids_labels
    ):
      permuted_unique_patient_ids_labels = rng.permutation(
          unique_patient_ids_labels
      )
    # Assign the permuted labels to the patient IDs
    permuted_labels = np.array(
        [
            permuted_unique_patient_ids_labels[np.where(
                unique_patient_ids == pid
            )[0][0]] for pid in patient_ids
        ]
    )
    return permuted_labels

  def single_seed_permutation_test(
      self, seed: int, output_dir: Path, best_auc_parameters: dict = {}
  ):
    """
    Perform permutation tests using a single seed with cross-validation and
          Optuna for hyperparameter optimization.

    Args:
      seed (int): random seed for reproducibility.
      output_dir (Path): directory where the results of the seed will be 
          saved.
      best_auc_parameters (dict, optional): best hyperparameters from AUC
          optimization. Defaults to {}.

    Raises:
      ValueError: best_auc_parameters must be provided for permutation tests
          without hyperparameter tuning.

    """
    # Check if best_auc_parameters is provided for permutation tests without
    # hyperparameter tuning
    if (not self.tune_permutation_hyperparameters
       ) and (best_auc_parameters == {}):
      raise ValueError(
          "best_auc_parameters must be provided for permutation tests without "
          "hyperparameter tuning."
      )
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Permute the labels
    y_permuted = self.permute_labels(
        seed, self.patient_ids.copy(), self.y.copy()
    )

    # Define variables for predictions
    predictions = np.zeros(self.y.shape)

    # Define variables for storing the best parameters
    best_parameters = {}

    # If we are not tuning hyperparameters, use the best parameters from AUC
    if not self.tune_permutation_hyperparameters:
      # Loop over the best parameters
      for key, params in best_auc_parameters.items():
        # Get the test IDs
        test_ids = np.array(list(map(int, key.split("_"))))
        # Split data into training and validation sets
        X_train, X_test, y_train, _ = self._get_train_test_split(
            test_ids, y_permuted
        )
        predictions[
            np.isin(self.patient_ids, test_ids)
        ] = self._train_and_predict(X_train, X_test, y_train, params, seed)
        best_parameters[key] = params
    # Perform hyperparameter tuning for permutation tests
    else:
      # Define cross-validation
      cv = self.get_cross_validation_split(
          self.X, y_permuted, self.patient_ids, self.cv_strategy,
          self.cv_n_splits, seed
      )
      # Perform Cross-Validation
      for train_idx, test_idx in cv:
        # Get the test IDs
        test_ids = np.unique(self.patient_ids[test_idx])

        # Split data into training and validation sets
        X_train, X_test, y_train, _ = self._get_train_test_split(
            test_ids, y_permuted
        )

        # Perform hyperparameter tuning with Optuna
        best_params = self._perform_hyperparameter_tuning(
            X_train, y_train, self.patient_ids[train_idx], seed
        )

        # Train the model and predict probabilities
        predictions[test_idx] = self._train_and_predict(
            X_train, X_test, y_train, best_params, seed
        )

        # Store the best parameters
        key = "_".join(
            np.sort(np.unique(self.patient_ids[test_idx])).astype(str)
        )
        best_parameters[key] = best_params

    # Save predictions and best parameters
    self._save_predictions_and_params(
        predictions, best_parameters, output_dir, seed, y_permuted
    )

  def multiple_seeds_cv_train_predict(self) -> None:
    """
    Perform training and prediction using multiple seeds with Cross-Validation
        and Optuna for hyperparameter optimization.

    """
    # Perform classification with multiple seeds in parallel
    results = [
        r for r in tqdm(
            Parallel(return_as="generator", n_jobs=self.n_jobs)(
                delayed(self.single_seed_cv_train_predict)
                (seed=seed, output_dir=self.output_dir)
                for seed in self.evaluation_seeds
            ), total=self.num_seeds, desc="Classification with multiple seeds"
        )
    ]

  def get_seed_results(self, seed: int, seed_dir: Path) -> pd.DataFrame:
    """
    Get the results of a seed from the predictions.

    Args:
      seed (int): Random seed for reproducibility.
      seed_dir (Path): Directory where the results of the seed are saved.

    Returns:
      pd.DataFrame: A DataFrame containing the predictions of the seed.

    """
    # Load the predictions from the seed
    seed_dir = seed_dir / f"seed_{seed}"
    predictions = pd.read_csv(seed_dir / "predictions.csv")

    return predictions

  def get_seed_roc_curve_and_auc(
      self, seed: int, seed_dir: Path, interp_fpr: Optional[np.ndarray] = None
  ) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Get the ROC curve and AUC score of a seed.

    Args:
      seed (int): Random seed for reproducibility.
      seed_dir (Path): Directory where the results of the seed are saved.
      interp_fpr (np.ndarray, optional): FPR values for interpolation. Defaults 
          to None.

    Returns:
      tuple[float, np.ndarray, np.ndarray]: A tuple containing the AUC score, 
          interpolated FPR values, and TPR values.

    """
    # Get the results of the seed
    predictions = self.get_seed_results(seed, seed_dir)

    # Get the ROC curve
    fpr, tpr, _ = roc_curve(
        predictions['y_true'], predictions['y_pred'], pos_label=1
    )

    # Calculate the AUC score
    auc_score = roc_auc_score(predictions['y_true'], predictions['y_pred'])

    # Interpolate the ROC curve
    if interp_fpr is not None:
      interp_tpr = np.interp(interp_fpr, fpr, tpr)
    else:
      interp_tpr = tpr
      interp_fpr = fpr

    return auc_score, interp_fpr, interp_tpr

  def plot_mean_roc_curve(self, std: int = 1, file_suffix: str = '') -> None:
    """
    Method to plot the mean ROC curve of the evaluation seeds.

    Args:
        std (int, optional): Number of standard deviations to plot in the ROC
            curve plot. Defaults to 1.
        file_name (str, optional): Suffix to add to the file name. Defaults to
            ''.
    
    """
    # Define the roc fpr range
    interp_fpr = np.linspace(0, 1, 100)
    # Define lists to store the ROC curves and AUC scores
    tprs, auc_scores = [], []
    # Iterate over the evaluation seeds
    for seed in self.evaluation_seeds:
      # Get the ROC curve and AUC score of the seed
      auc_score, _, interp_tpr = self.get_seed_roc_curve_and_auc(
          seed, self.output_dir, interp_fpr
      )
      # Store the TPR values
      tprs.append(interp_tpr)
      # Store the AUC scores
      auc_scores.append(auc_score)
    # Calculate the mean and standard deviation of the TPR values
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    # Calculate the mean and standard deviation of the AUC scores
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    # Create the ROC curve plot
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot the default ROC curve
    ax.plot(
        [0, 1], [0, 1], color=fc.DEFAULT_COLOR, lw=fc.DEFAULT_LINE_WIDTH,
        linestyle='--'
    )
    # Plot the mean ROC curve
    ax.plot(
        interp_fpr, mean_tpr, color=DEFAULT_DISCRETE_CMAP[0],
        linewidth=fc.DEFAULT_LINE_WIDTH,
        label=f"Mean ROC Curve (AUC = {mean_auc:.2f} $\pm$ {std * std_auc:.2f})"
    )
    # Plot the standard deviation of the ROC curve
    ax.fill_between(
        interp_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
        color=DEFAULT_DISCRETE_CMAP[0], alpha=0.25
    )
    # Final plot adjustments
    fc.customize_ticks(ax)
    fc.customize_spines(ax)
    fc.set_titles_and_labels(ax, '', 'FPR', 'TPR')
    fc.customize_legend(ax, 'lower right')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(
        self.output_dir / f"{file_suffix}_mean_roc_curve.pdf",
        bbox_inches='tight', dpi=1200, transparent=True
    )
    plt.close()
    # Save the ROC curve data
    pd.DataFrame(
        {'fpr': interp_fpr, 'tpr': mean_tpr}
    ).to_csv(self.output_dir / f"{file_suffix}_roc_curve.csv", index=False)

  def get_params_from_median_auc(self,
                                 seed_dir: Path) -> tuple[dict, int, float]:
    """
    Get the best hyperparameters from the median seed.

    Args:
      seed_dir (Path): Directory where the results of the seeds are saved.

    Returns:
      tuple[dict, int, float]: A tuple containing the best hyperparameters, 
          the seed with the median AUC score, and the median AUC score.

    """
    # Get the AUC scores of the evaluation seeds
    auc_scores = [
        self.get_seed_roc_curve_and_auc(seed, seed_dir)[0]
        for seed in self.evaluation_seeds
    ]
    # Get the index of the median AUC score
    median_idx = np.argsort(auc_scores)[len(auc_scores) // 2]
    # Get the seed with the median AUC score
    median_seed = self.evaluation_seeds[median_idx]
    # Get the median AUC score
    median_seed_auc = auc_scores[median_idx]
    # Load the best parameters from the median seed
    with open(
        self.output_dir / f"seed_{median_seed}/best_parameters.json"
    ) as f:
      best_params = json.load(f)
    # Return the best parameters
    return best_params, median_seed, median_seed_auc

  def get_params_from_max_auc(self, seed_dir: Path) -> tuple[dict, int, float]:
    """
    Get the best hyperparameters from the seed with the maximum AUC score.

    Args:
      seed_dir (Path): Directory where the results of the seeds are saved.

    Returns:
      tuple[dict, int, float]: A tuple containing the best hyperparameters, 
          the seed with the maximum AUC score, and the maximum AUC score.

    """
    # Get the AUC scores of the evaluation seeds
    auc_scores = [
        self.get_seed_roc_curve_and_auc(seed, seed_dir)[0]
        for seed in self.evaluation_seeds
    ]
    # Get the index of the maximum AUC score
    max_idx = np.argmax(auc_scores)
    # Get the seed with the maximum AUC score
    max_seed = self.evaluation_seeds[max_idx]
    # Get the maximum AUC score
    max_seed_auc = auc_scores[max_idx]
    # Load the best parameters from the maximum seed
    with open(self.output_dir / f"seed_{max_seed}/best_parameters.json") as f:
      best_params = json.load(f)
    # Return the best parameters
    return best_params, max_seed, max_seed_auc

  def permutation_classification_with_parallel(self, best_params: dict) -> None:
    """
    Method to perform permutation tests with parallel processing.

    Args:
      best_params (dict): Dictionary of best hyperparameters from the best AUC
          seed.
    
    """
    # Create a folder for the seed
    output_dir = self.output_dir / "permutation"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Set the best parameters as empty if we are tuning the permutation
    # hyperparameters
    if self.tune_permutation_hyperparameters:
      best_params = {}

    # Use joblib to run the function in parallel for each seed
    results = [
        r for r in tqdm(
            Parallel(return_as="generator", n_jobs=self.n_jobs)(
                delayed(self.single_seed_permutation_test)(
                    seed=seed, output_dir=output_dir,
                    best_auc_parameters=best_params
                ) for seed in self.permutation_seeds
            ), total=self.num_permutations,
            desc="Running Permutation for multiple seeds"
        )
    ]

  def plot_permutation_distribution(
      self, real_auc: float, file_suffix: str = ''
  ) -> None:
    """
    Method to plot the permutation distribution and calculate the p-value.

    Args:
      real_auc (float): The real AUC score to compare with the permutation
          distribution.
      file_suffix (str, optional): Suffix to add to the file name. Defaults to
          ''.

    """
    # Create a folder for the seed
    output_dir = self.output_dir / "permutation"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Define the roc fpr range
    interp_fpr = np.linspace(0, 1, 100)
    # Define lists to store the ROC curves and AUC scores
    permutation_auc_scores = []
    # Iterate over the evaluation seeds
    for seed in self.permutation_seeds:
      # Get the ROC curve and AUC score of the seed
      auc_score, _, _ = self.get_seed_roc_curve_and_auc(
          seed, output_dir, interp_fpr
      )
      # Store the AUC scores
      permutation_auc_scores.append(auc_score)
    # Convert the list to a numpy array
    permutation_auc_scores = np.array(permutation_auc_scores)
    # Calculate p-value
    p_value = (np.sum(permutation_auc_scores >= real_auc) +
               1) / (len(permutation_auc_scores) + 1)
    # Plot the permutation test histogram
    ax = sns.histplot(
        permutation_auc_scores, bins=30, edgecolor='k', alpha=0.7,
        label='Permutation AUCs', color=DEFAULT_DISCRETE_CMAP[0], stat="count"
    )
    # Plot the true AUC score
    ax.axvline(
        real_auc, color=DEFAULT_DISCRETE_CMAP[1], linestyle='dashed',
        linewidth=fc.DEFAULT_LINE_WIDTH, label=f'True AUC ({real_auc:.3f})'
    )
    fc.set_titles_and_labels(ax, '', 'AUC Scores', 'Count')
    fc.customize_spines(ax)
    fc.customize_ticks(ax)
    # Add the legend
    l = ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True,
        prop={"size": fc.DEFAULT_FONT_SIZE, "weight": fc.DEFAULT_FONT_WEIGHT}
    )
    for text in l.get_texts():
      text.set_color(fc.DEFAULT_COLOR)
    # Add p-value annotation above the red line
    ax.text(
        real_auc,
        ax.get_ylim()[1], f"p = ({p_value:.6f})", ha='center', va='bottom',
        color=DEFAULT_DISCRETE_CMAP[1], fontsize=fc.DEFAULT_FONT_SIZE,
        fontweight=fc.DEFAULT_FONT_WEIGHT
    )
    # Define the x-axis limits
    ax.set_xlim([-0.05, 1.05])
    #
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{file_suffix}_permutation_distribution.pdf",
        bbox_inches='tight', dpi=1200, transparent=True
    )
    plt.close()
    pd.DataFrame(
        {'seed': self.permutation_seeds, 'auc': permutation_auc_scores}
    ).to_csv(
        output_dir / f'{file_suffix}_permutation_auc_scores.csv', index=False
    )

  def get_shap_values(self, seed: int, best_auc_parameters: dict) -> np.ndarray:
    """
    Get the SHAP values for feature importance.

    Args:
      seed (int): Random seed for reproducibility.
      best_auc_parameters (dict): Dictionary of best hyperparameters from the 
          best AUC seed.

    Returns:
      np.ndarray: SHAP values for feature importance.

    """
    # Define object to store SHAP values
    shap_values_all = np.zeros((self.X.shape[0], self.X.shape[1]))

    # Define cross-validation
    cv = self.get_cross_validation_split(
        self.X, self.y, self.patient_ids, self.cv_strategy, self.cv_n_splits,
        seed
    )

    # Perform Cross-Validation
    for idx, (train_idx, test_idx) in enumerate(cv):
      # Split data into training and validation sets
      X_train, X_test = self.X[train_idx], self.X[test_idx]
      y_train, _ = self.y[train_idx], self.y[test_idx]
      # Get the best model key
      key = np.unique(self.patient_ids[test_idx])
      if isinstance(key, np.integer):
        key = str(int(key))
      if isinstance(key, np.ndarray):
        key = "_".join(key.astype(str))
      # Get the best parameters
      best_params = best_auc_parameters[key]
      # Create the best model for Logistic Regression
      best_model = self.get_model(best_params, seed)
      # Fit the best model
      best_model.fit(X_train, y_train)
      if len(best_model.steps[:-1]) > 0:
        # Remove the final step (the model) from the pipeline
        preprocessor = Pipeline(best_model.steps[:-1])
        # Transform the data
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
      else:
        X_train_transformed = X_train
        X_test_transformed = X_test
      # Get the best model without the pipeline
      final_model = best_model.named_steps['model']
      # Get the SHAP explainer
      if isinstance(final_model, LogisticRegression):
        explainer = shap.LinearExplainer(final_model, X_train_transformed)
      elif isinstance(
          final_model, (
              DecisionTreeClassifier, RandomForestClassifier, XGBClassifier,
              LGBMClassifier
          )
      ):
        explainer = shap.TreeExplainer(final_model, X_train_transformed)
      else:
        explainer = shap.Explainer(final_model, X_train_transformed)
      # Get the SHAP values
      shap_values = explainer(X_test_transformed)
      if shap_values.values.ndim == 3:
        shap_values_all[test_idx, :] = shap_values.values[:, :, 1]
      else:
        shap_values_all[test_idx, :] = shap_values.values
    return shap_values_all

  def plot_shap_values(self, shap_values: np.ndarray, max_display=20) -> None:
    """
    Plot the SHAP values for feature importance.

    Args:
      shap_values (np.ndarray): SHAP values for feature importance.
      max_display (int, optional): Maximum number of features to display. 
          Defaults to 20.

    """
    # Plot the SHAP summary plot
    shap.plots.beeswarm(
        shap.Explanation(
            values=shap_values, data=self.X, feature_names=self.feature_names
        ), show=False, max_display=max_display, color=DEFAULT_CONTINUOUS_CMAP
    )
    # Retrieve the current figure
    fig, ax = plt.gcf(), plt.gca()
    # Set the figure size
    fig.set_size_inches(3, 9)
    # Customize the plot
    fc.set_titles_and_labels(ax, '', 'SHAP value', '')
    fc.customize_ticks(ax)
    # fc.customize_colorbar(fig.colorbar)
    ax.spines["bottom"].set_linewidth(fc.DEFAULT_LINE_WIDTH)
    ax.spines["bottom"].set_color(fc.DEFAULT_COLOR)
    # Customize the colorbar
    cbar = None
    for axis in fig.axes:
      if hasattr(axis, 'collections') and axis.collections:
        cbar = axis
    cbar.set_ylabel(
        'Feature value', fontsize=fc.DEFAULT_FONT_SIZE,
        fontweight=fc.DEFAULT_FONT_WEIGHT, color=fc.DEFAULT_COLOR
    )
    for label in cbar.get_yticklabels():
      label.set_fontweight(fc.DEFAULT_FONT_WEIGHT)
      label.set_fontsize(fc.DEFAULT_FONT_SIZE)
      label.set_color(fc.DEFAULT_COLOR)
    for spine in cbar.spines.values():
      spine.set_linewidth(fc.DEFAULT_LINE_WIDTH)
      spine.set_color(fc.DEFAULT_COLOR)
    # Save the plot
    plt.savefig(
        self.output_dir / f"shap_beeswarm.pdf", bbox_inches='tight', dpi=1200,
        transparent=True
    )
    plt.close()
    # Plot the SHAP bar plot
    shap.plots.bar(
        shap.Explanation(values=shap_values, feature_names=self.feature_names),
        show=False, max_display=max_display
    )
    # Retrieve the current figure
    fig, ax = plt.gcf(), plt.gca()
    # Set the figure size
    fig.set_size_inches(3, 9)
    # Customize the plot
    fc.set_titles_and_labels(ax, '', 'mean(|SHAP value|)', '')
    fc.customize_ticks(ax, rotate_x_ticks=45)
    fc.customize_spines(ax)
    plt.savefig(
        self.output_dir / f"shap_bar.pdf", bbox_inches='tight', dpi=1200,
        transparent=True
    )
    plt.close()
    pd.DataFrame(shap_values, columns=self.feature_names
                ).to_csv(self.output_dir / 'shap_values.csv', index=False)

  def run(self) -> 'BinaryClassifierTrainer':
    """
    Run the full training, evaluation, and plotting pipeline.

    Workflow:
      - Perform binary classificationn training with multiple seeds using cross 
          validation and hyperameter tuning.
      - Perform permutation tests to evaluate significance compared to the 
          median AUC seed.
      - Compute Shapley values for feature importance.
      - Plot the mean +- std ROC curve.
      - Plot the permutation distribution and calculate p-value.
      - Plot the SHAP values
    
    Returns:
      BinaryClassifierTrainer: The trained binary classifier.

    """
    # Perform binary classification training with multiple seeds using cross
    # validation and hyperparameter tuning
    self.multiple_seeds_cv_train_predict()
    # Create ROC-AUC plot
    self.plot_mean_roc_curve()
    # Get the best hyperparameters from the median auc seed
    median_best_params, _, median_seed_auc = self.get_params_from_median_auc(
        self.output_dir
    )
    # Get the best hyperparameters from the max auc seed
    max_best_params, max_seed, _ = self.get_params_from_max_auc(self.output_dir)
    # Perform permutation tests
    self.permutation_classification_with_parallel(median_best_params)
    # Plot permutation distribution
    self.plot_permutation_distribution(median_seed_auc)
    # Get SHAP values
    shap_values = self.get_shap_values(max_seed, max_best_params)
    # Plot SHAP values
    self.plot_shap_values(shap_values, max_display=10)
    # Return the trained binary classifier
    return self


class WeakSupervisionBinaryClassifierTrainer:
  """ 
   A class to train a binary classifier where samples have weak labels from
    the instance level using Cross-Validation end to end including 
    hyperparameter tuning, permutation tests for model evaluation, 
    SHAP, plotting and more.

  """

  def __init__(
      self, X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray,
      sample_ids: np.ndarray, instance_ids: np.ndarray,
      feature_names: np.ndarray, num_seeds: int, num_permutations: int,
      model_type: str, output_dir: Path, tune_hyperparameters: bool = True,
      tune_permutation_hyperparameters: bool = False,
      primary_seed: int = PRIMARY_SEED, n_trials=50, n_jobs: int = -1,
      cv_strategy: Literal["loocv", "kfold",
                           "stratified_kfold"] = "loocv", cv_n_splits: int = 5,
      optuna_cv_strategy: Literal["loocv", "kfold",
                                  "stratified_kfold"] = "stratified_kfold",
      optuna_cv_n_splits: int = 5,
      preprocessing_pipeline: Optional[Pipeline] = None,
      calibrate_proba: bool = False, custom_model_name: Optional[str] = None,
      custom_model: Optional[BaseEstimator] = None,
      custom_hyperparameter_space: Optional[Callable] = None,
      use_gpu: bool = False, filter_prob_neg_lst: list[float] = [0.5, 0.4, 0.3],
      filter_prob_pos_lst: list[float] = [0.5, 0.6, 0.7]
  ):

    self.instance_ids = instance_ids
    self.output_dir = output_dir

    self.filter_prob_neg_lst = filter_prob_neg_lst
    self.filter_prob_pos_lst = filter_prob_pos_lst

    self.filter_prob_neg = filter_prob_neg_lst[0]
    self.filter_prob_pos = filter_prob_pos_lst[0]

    # Train on instance-level data
    self.binary_trainer = BinaryClassifierTrainer(
        X=X, y=y, patient_ids=patient_ids, sample_ids=sample_ids,
        feature_names=feature_names, num_seeds=num_seeds,
        num_permutations=num_permutations, model_type=model_type,
        output_dir=output_dir, tune_hyperparameters=tune_hyperparameters,
        tune_permutation_hyperparameters=tune_permutation_hyperparameters,
        primary_seed=primary_seed, n_trials=n_trials, n_jobs=n_jobs,
        cv_strategy=cv_strategy, cv_n_splits=cv_n_splits,
        optuna_cv_strategy=optuna_cv_strategy,
        optuna_cv_n_splits=optuna_cv_n_splits,
        preprocessing_pipeline=preprocessing_pipeline,
        calibrate_proba=calibrate_proba, custom_model_name=custom_model_name,
        custom_model=custom_model,
        custom_hyperparameter_space=custom_hyperparameter_space, use_gpu=use_gpu
    )

  def get_seed_aggregated_roc_curve_and_auc(
      self, seed: int, seed_dir: Path, interp_fpr: Optional[np.ndarray] = None
  ) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Get the ROC curve and AUC score of a seed.

    Args:
      seed (int): Random seed for reproducibility.
      seed_dir (Path): Directory where the results of the seed are saved.
      interp_fpr (np.ndarray, optional): FPR values for interpolation. Defaults 
          to None.

    Returns:
      tuple[float, np.ndarray, np.ndarray]: A tuple containing the AUC score, 
          interpolated FPR values, and TPR values.

    """
    # Get the results of the seed
    predictions = self.binary_trainer.get_seed_results(seed, seed_dir)

    # Filter the predictions based on the threshold probabilities
    predictions_filtered = predictions[
        (predictions['y_pred'] <= self.filter_prob_neg) |
        (predictions['y_pred'] >= self.filter_prob_pos)]

    # check if there are sample_ids not in the predictions filtered
    missing_sample_ids = np.setdiff1d(
        predictions['sample_id'], predictions_filtered['sample_id']
    )

    # add the missing sample_ids to the predictions_filtered with y_pred = 0.5
    for missing_sample_id in missing_sample_ids:
      print(f"Missing sample_id: {missing_sample_id}")
      y_true = predictions[predictions['sample_id'] == missing_sample_id
                          ]['y_true'].values[0]
      predictions_filtered = pd.concat(
          [
              predictions_filtered,
              pd.DataFrame(
                  [
                      {
                          'sample_id': missing_sample_id, 'y_pred': 0.5,
                          'y_true': y_true
                      }
                  ]
              )
          ], ignore_index=True
      )

    # aggreagte the predictions by sample_id
    predictions_aggregated = predictions_filtered.groupby('sample_id').agg(
        y_true=('y_true', 'max'), y_pred=('y_pred', 'mean')
    ).reset_index()

    # Get the ROC curve
    fpr, tpr, _ = roc_curve(
        predictions_aggregated['y_true'], predictions_aggregated['y_pred'],
        pos_label=1
    )

    # Calculate the AUC score
    auc_score = roc_auc_score(
        predictions_aggregated['y_true'], predictions_aggregated['y_pred']
    )

    # Interpolate the ROC curve
    if interp_fpr is not None:
      interp_tpr = np.interp(interp_fpr, fpr, tpr)
    else:
      interp_tpr = tpr
      interp_fpr = fpr

    return auc_score, interp_fpr, interp_tpr

  def run(self) -> 'MILBinaryClassifierTrainer':
    # Run the BinaryClassifierTrainer
    self.binary_trainer.run()

    # Update the predictions with the instance ids
    for seed in self.binary_trainer.evaluation_seeds:
      predictions = self.binary_trainer.get_seed_results(seed, self.output_dir)
      predictions['instance_id'] = self.instance_ids
      predictions.to_csv(
          self.output_dir / f"seed_{seed}" / "predictions.csv", index=False
      )

    # Update the predictions with the instance ids in the permutation folder
    for seed in self.binary_trainer.permutation_seeds:
      predictions = self.binary_trainer.get_seed_results(
          seed, self.output_dir / "permutation"
      )
      predictions['instance_id'] = self.instance_ids
      predictions.to_csv(
          self.output_dir / "permutation" / f"seed_{seed}" / "predictions.csv",
          index=False
      )

    # Store the BinaryClassifierTrainer get_seed_roc_curve_and_auc method
    get_seed_aggregated_roc_curve_and_auc = self.binary_trainer.get_seed_roc_curve_and_auc

    # Change the BinaryClassifierTrainer get_seed_roc_curve_and_auc method
    # to get the aggregated roc curve and auc
    self.binary_trainer.get_seed_roc_curve_and_auc = self.get_seed_aggregated_roc_curve_and_auc

    for filter_prob_neg, filter_prob_pos in zip(
        self.filter_prob_neg_lst, self.filter_prob_pos_lst
    ):
      self.filter_prob_neg = filter_prob_neg
      self.filter_prob_pos = filter_prob_pos

      # Plot the aggregated mean ROC curve
      self.binary_trainer.plot_mean_roc_curve(
          file_suffix=f"aggregated_{self.filter_prob_neg}_{self.filter_prob_pos}"
      )

      # Get the best hyperparameters from the median auc seed based on the
      # non aggregated predictions
      _, _, median_seed_auc = self.binary_trainer.get_params_from_median_auc(
          self.output_dir
      )

      # Plot aggregated permutation distribution
      self.binary_trainer.plot_permutation_distribution(
          median_seed_auc,
          file_suffix=f"aggregated_{self.filter_prob_neg}_{self.filter_prob_pos}"
      )

    # Change the BinaryClassifierTrainer get_seed_roc_curve_and_auc method
    # back to the original method
    self.binary_trainer.get_seed_roc_curve_and_auc = get_seed_aggregated_roc_curve_and_auc

    return self
