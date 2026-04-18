import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from huggingface_hub import HfApi, login
from datasets import load_dataset
import mlflow
import mlflow.sklearn
import optuna
import joblib

warnings.filterwarnings('ignore')

# --- Configuration ---
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "Bharath1434")
DATASET_REPO = os.environ.get("DATASET_REPO", f"{HF_USERNAME}/tourism-package-prediction-dataset")
MODEL_REPO = os.environ.get("MODEL_REPO", f"{HF_USERNAME}/tourism-package-prediction-model")

MASTER_DIR = "visit_with_us_mlops"
SCRIPTS_DIR = os.path.join(MASTER_DIR, "scripts")

# Authenticate with Hugging Face
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)

# --- Load Processed Data ---
print(f"Loading processed datasets from Hugging Face dataset: {DATASET_REPO}/")

X_train_smote = load_dataset('csv', data_files=f'hf://datasets/{DATASET_REPO}/data/processed/X_train_smote.csv', split='train').to_pandas()
y_train_smote = load_dataset('csv', data_files=f'hf://datasets/{DATASET_REPO}/data/processed/y_train_smote.csv', split='train').to_pandas().squeeze()
X_test = load_dataset('csv', data_files=f'hf://datasets/{DATASET_REPO}/data/processed/X_test.csv', split='train').to_pandas()
y_test = load_dataset('csv', data_files=f'hf://datasets/{DATASET_REPO}/data/processed/y_test.csv', split='train').to_pandas().squeeze()

print(f"✅ Processed train and test datasets loaded successfully!")
print(f"X_train_smote shape: {X_train_smote.shape}, y_train_smote shape: {y_train_smote.shape}")

# --- MLflow Setup ---
mlflow.set_experiment("tourism-Package-Prediction-Tuning")

# --- Optuna Objective Function ---
def xgb_objective(trial):
    with mlflow.start_run(run_name=f"XGBoost_Trial_{trial.number}", nested=True):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'eval_metric': 'logloss', 'random_state': 42, 'verbosity': 0
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train_smote, y_train_smote, cv=cv, scoring='roc_auc', n_jobs=-1)
        roc_auc = scores.mean()

        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", roc_auc)

        return roc_auc

# --- Run Optuna Optimization ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize', study_name='XGB_Tuning')
study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
print(f"\n🏆 Best CV AUC-ROC: {study.best_value:.4f}")
print(f"📋 Best Parameters:")
for k, v in best_params.items():
    print(f"   {k:25s}: {v}")

# --- Log the Best Model to MLflow and save locally ---
with mlflow.start_run(run_name="Best_Tuned_XGBoost_Model", nested=True):
    mlflow.log_params(best_params)

    best_xgb_model = XGBClassifier(**best_params, eval_metric='logloss', random_state=42, verbosity=0)
    best_xgb_model.fit(X_train_smote, y_train_smote)

    y_pred = best_xgb_model.predict(X_test)
    y_prob = best_xgb_model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    mlflow.log_metrics({"accuracy": acc, "precision": prec,
                        "recall": rec, "f1_score": f1, "roc_auc": auc})

    # Save model locally
    MODEL_PATH = "best_xgb_model.pkl"
    joblib.dump(best_xgb_model, MODEL_PATH)
    print(f"✅ Best model saved locally as {MODEL_PATH}")

    # Log model to MLflow with trusted types
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`artifact_path` is deprecated. Please use `name` instead.",
            category=UserWarning
        )
        mlflow.sklearn.log_model(best_xgb_model, "tuned_xgboost_model", serialization_format="skops",
                                 skops_trusted_types=[
                                     'sklearn._loss.link.Interval',
                                     'sklearn._loss.link.LogitLink',
                                     'sklearn._loss.loss.HalfBinomialLoss',
                                     'xgboost.sklearn.XGBClassifier',
                                     'xgboost.core.Booster',
                                     'xgboost.callback.EarlyStopping',
                                     'numpy.ndarray'
                                 ])
    print(f"✅ Tuned XGBoost Model logged to MLflow with AUC-ROC: {auc:.4f}")

    # Create HF Model Repo and upload model
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="best_xgb_model.pkl",
        repo_id=MODEL_REPO, repo_type="model",
        token=HF_TOKEN
    )
    print(f"\n✅ Best model registered on Hugging Face Model Hub!")
    print(f"🔗 https://huggingface.co/{MODEL_REPO}")

    # Save feature names as a JSON file and upload it to the model repo
    feature_cols = X_train_smote.columns.tolist()
    features_path = os.path.join(os.getcwd(), "feature_names.json") # Save in current working directory for artifact upload
    with open(features_path, "w") as f:
        import json
        json.dump(feature_cols, f)

    api.upload_file(
        path_or_fileobj=features_path,
        path_in_repo="feature_names.json",
        repo_id=MODEL_REPO,
        repo_type="model",
        token=HF_TOKEN
    )
    print("✅ feature_names.json uploaded to model repo!")

print("--- Model training script finished ---")
