
import os
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from huggingface_hub import hf_hub_download, login
from datasets import load_dataset

# Configuration from environment variables (for GitHub Actions)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("HF_TOKEN environment variable not set. Please set it.")
    exit(1)

login(token=HF_TOKEN, add_to_git_credential=False)

HF_USERNAME = os.environ.get("HF_USERNAME", "Bharath1434")
DATASET_REPO = os.environ.get("DATASET_REPO", f"{HF_USERNAME}/tourism-package-prediction-dataset")
MODEL_REPO = os.environ.get("MODEL_REPO", f"{HF_USERNAME}/tourism-package-prediction-model")

print("--- Running evaluation script ---")

# 1. Download and load the best model
try:
    model_path_local = hf_hub_download(repo_id=MODEL_REPO, filename="best_xgb_model.pkl")
    model = joblib.load(model_path_local)
    print("✅ Model loaded successfully from Hugging Face Model Hub.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# 2. Download and load test data
try:
    # Load X_test
    raw_X_test = load_dataset('csv', data_files=f'hf://datasets/{DATASET_REPO}/data/processed/X_test.csv', split='train')
    X_test = raw_X_test.to_pandas()

    # Load y_test
    raw_y_test = load_dataset('csv', data_files=f'hf://datasets/{DATASET_REPO}/data/processed/y_test.csv', split='train')
    y_test = raw_y_test.to_pandas().squeeze()
    print("✅ Test data loaded successfully from Hugging Face Dataset Hub.")
except Exception as e:
    print(f"❌ Error loading test data: {e}")
    exit(1)


# 3. Evaluate the model
try:
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"Computed ROC AUC: {roc_auc:.4f}")
except Exception as e:
    print(f"❌ Error during model evaluation: {e}")
    # Set a default low AUC to fail the quality gate
    roc_auc = 0.0

# 4. Implement Quality Gate
QUALITY_GATE_THRESHOLD = 0.85 # As suggested in mlops_pipeline.yaml
auc_passed = "false"
if roc_auc >= QUALITY_GATE_THRESHOLD:
    auc_passed = "true"
    print(f"✅ Quality Gate Passed: AUC-ROC ({roc_auc:.4f}) >= {QUALITY_GATE_THRESHOLD}")
else:
    print(f"❌ Quality Gate Failed: AUC-ROC ({roc_auc:.4f}) < {QUALITY_GATE_THRESHOLD}")

# Set GitHub Actions output
# GITHUB_OUTPUT is available in GitHub Actions environment
if os.environ.get("GITHUB_OUTPUT"):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        print(f"auc_passed={auc_passed}", file=f)
    print("✅ GitHub Actions output 'auc_passed' set.")
else:
    print(f"ℹ️ Not running in GitHub Actions, GITHUB_OUTPUT not set. Result: auc_passed={auc_passed}")

print("--- Evaluation script finished ---")
