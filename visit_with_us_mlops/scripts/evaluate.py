import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from huggingface_hub import hf_hub_download, login
from datasets import load_dataset

# --- Configuration ---
HF_TOKEN = os.environ.get("HF_TOKEN")

# Validate HF_TOKEN early
if not HF_TOKEN or not HF_TOKEN.strip():
    print("Error: HF_TOKEN environment variable is missing or empty. Please set it in GitHub Secrets.")
    exit(1) # Exit the script
HF_TOKEN = HF_TOKEN.strip() # Clean any whitespace

HF_USERNAME = os.environ.get("HF_USERNAME", "Bharath1434")
DATASET_REPO = os.environ.get("DATASET_REPO", f"{HF_USERNAME}/tourism-package-prediction-dataset")
MODEL_REPO = os.environ.get("MODEL_REPO", f"{HF_USERNAME}/tourism-package-prediction-model")

# Authenticate with Hugging Face
login(token=HF_TOKEN, add_to_git_credential=False)

# --- Load Data and Model ---
print(f"Loading processed test datasets from Hugging Face dataset: {DATASET_REPO}/")
X_test = load_dataset('csv', data_files=f'hf://datasets/{DATASET_REPO}/data/processed/X_test.csv', split='train', token=HF_TOKEN).to_pandas()
y_test = load_dataset('csv', data_files=f'hf://datasets/{DATASET_REPO}/data/processed/y_test.csv', split='train', token=HF_TOKEN).to_pandas().squeeze()
print(f"✅ Test data loaded. X_test shape: {X_test.shape}")

print(f"Downloading best model from Hugging Face model hub: {MODEL_REPO}/")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_xgb_model.pkl", token=HF_TOKEN)
best_model = joblib.load(model_path)
print(f"✅ Best model loaded.")

print(f"Downloading feature names from Hugging Face model hub: {MODEL_REPO}/")
features_path = hf_hub_download(repo_id=MODEL_REPO, filename="feature_names.json", token=HF_TOKEN)
with open(features_path, "r") as f:
    expected_features = json.load(f)
print(f"✅ Feature names loaded.")

# Ensure X_test columns match the trained model's expected features
missing_cols = set(expected_features) - set(X_test.columns)
for c in missing_cols:
    X_test[c] = 0
# Reorder columns to ensure consistency
X_test = X_test[expected_features]

# --- Evaluate Model ---
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred_best)
prec = precision_score(y_test, y_pred_best)
rec  = recall_score(y_test, y_pred_best)
f1   = f1_score(y_test, y_pred_best)
auc  = roc_auc_score(y_test, y_prob_best)

print("\n🏆 FINAL MODEL PERFORMANCE:")
print("="*45)
print(f"  Accuracy       : {acc:.4f}")
print(f"  Precision      : {prec:.4f}")
print(f"  Recall         : {rec:.4f}")
print(f"  F1_score       : {f1:.4f}")
print(f"  ROC_AUC        : {auc:.4f}")

# --- Generate Plots and Save ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('white')

# Confusion Matrix
cm_arr = confusion_matrix(y_test, y_pred_best)
axes[0].set_facecolor('white')
sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted No', 'Predicted Yes'],
            yticklabels=['Actual No', 'Actual Yes'], ax=axes[0])
axes[0].set_title('Confusion Matrix', fontsize=13)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
axes[1].set_facecolor('white')
axes[1].plot(fpr, tpr, color='#38bdf8', lw=2, label=f'XGBoost (AUC = {auc:.3f})')
axes[1].plot([0,1], [0,1], '--', color='#475569', label='Random Classifier')
axes[1].set_title('ROC Curve', fontsize=13)
axes[1].legend()

# Feature Importance (Top 15)
feat_imp = pd.Series(best_model.feature_importances_, index=expected_features)
feat_imp_sorted = feat_imp.nlargest(15)
axes[2].set_facecolor('white')
axes[2].barh(feat_imp_sorted.index, feat_imp_sorted.values, color='#818cf8')
axes[2].set_title('Top 15 Feature Importances', fontsize=13)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=120, bbox_inches='tight')
print("✅ Model evaluation plots saved as 'model_evaluation.png'")

# Save feature importance to CSV
feat_imp.nlargest(15).to_csv("feature_importance.csv")
print("✅ Feature importance saved as 'feature_importance.csv'")

# --- Quality Gate Check ---
AUC_THRESHOLD = 0.85 # Define your quality gate threshold
if auc > AUC_THRESHOLD:
    print(f"✅ Model AUC-ROC ({auc:.4f}) is above the threshold ({AUC_THRESHOLD:.2f}). Quality gate passed!")
    # This output will be used by GitHub Actions to decide if deployment proceeds
    print(f"::set-output name=auc_passed::true")
else:
    print(f"❌ Model AUC-ROC ({auc:.4f}) is below the threshold ({AUC_THRESHOLD:.2f}). Quality gate failed!")
    print(f"::set-output name=auc_passed::false")

print("--- Model evaluation script finished ---")
