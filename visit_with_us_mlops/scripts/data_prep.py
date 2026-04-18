import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from huggingface_hub import HfApi, hf_hub_download, login
from datasets import load_dataset

warnings.filterwarnings('ignore')

# --- Configuration (using environment variables for GitHub Actions) ---
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "Bharath1434")
DATASET_REPO = os.environ.get("DATASET_REPO", f"{HF_USERNAME}/tourism-package-prediction-dataset")

MASTER_DIR = "visit_with_us_mlops"
DATA_DIR = os.path.join(MASTER_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Authenticate with Hugging Face (optional for upload if repo is public, but good practice)
# The raw dataset is now expected to be part of the Git repository, not downloaded from HF.

print(f"--- Starting data preparation for {DATASET_REPO} ---")

# 1. Load raw dataset from local path (checked out from Git)
csv_filename = "tourism.csv"
local_raw_data_path = os.path.join(DATA_DIR, csv_filename)

print(f"Attempting to load {csv_filename} from {local_raw_data_path} (assuming it's checked out from Git repo)...")
if os.path.exists(local_raw_data_path):
    data = pd.read_csv(local_raw_data_path)
    print(f"✅ Raw dataset '{csv_filename}' loaded successfully from local path.")
else:
    print(f"❌ Error: Raw dataset '{csv_filename}' not found at {local_raw_data_path}. Ensure it's committed to the Git repository.")
    exit(1)

print(f"Loaded data with shape: {data.shape}")

# 2. Data Cleaning
# Drop 'Unnamed: 0' column if it exists
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])
    print("✅ Dropped 'Unnamed: 0' column.")

# Correct 'Fe Male' to 'Female' in 'Gender' column
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].replace({'Fe Male': 'Female'})
    print("✅ Corrected 'Fe Male' to 'Female' in 'Gender' column.")

# Club 'Unmarried' with 'Single' in 'MaritalStatus' column
if 'MaritalStatus' in data.columns:
    data['MaritalStatus'] = data['MaritalStatus'].replace({'Unmarried': 'Single'})
    print("✅ Clubbed 'Unmarried' with 'Single' in 'MaritalStatus' column.")

# Drop 'CustomerID' column
if 'CustomerID' in data.columns:
    data = data.drop(columns=['CustomerID'])
    print("✅ Dropped 'CustomerID' column.")

# Check for missing values (as per EDA, none were found, but good to verify)
if data.isnull().sum().sum() == 0:
    print("✅ Verified no missing values detected in the dataset.")
else:
    print(f"❌ Missing values detected. {data.isnull().sum().sum()} total. Implement imputation if necessary.")

# IQR-based outlier capping for numerical columns
outlier_cols = ['Age', 'MonthlyIncome', 'DurationOfPitch', 'NumberOfTrips']
print("✅ Applying IQR-based outlier capping for numerical columns.")
for col in outlier_cols:
    if col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Ensure lower bound is not less than zero for applicable columns
        if col in ['Age', 'DurationOfPitch', 'NumberOfTrips']:
            lower = max(0, lower)

        data[col] = data[col].clip(lower, upper)
        print(f"  → {col}: capped outliers using bounds [{lower:.1f}, {upper:.1f}]")
    else:
        print(f"  → WARNING: Column '{col}' not found for outlier capping.")

print(f"Cleaned dataset shape: {data.shape}")

# 3. Split the cleaned dataset into training and testing sets, Apply SMOTE
X = data.drop('ProdTaken', axis=1)
y = data['ProdTaken']

# Identify categorical columns for encoding
categorical_cols = X.select_dtypes(include='object').columns

# Apply one-hot encoding to categorical features
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(f"Features shape after one-hot encoding: {X_encoded.shape}")

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train/Test split: X_train={X_train.shape}, X_test={X_test.shape}")

# Apply SMOTE only to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE applied: X_train_smote={X_train_smote.shape}, y_train_smote={y_train_smote.shape}")

# Save the processed datasets locally
X_train_smote.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_smote.csv'), index=False)
y_train_smote.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train_smote.csv'), index=False)
X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)
print(f"✅ Processed datasets saved locally to {PROCESSED_DATA_DIR}/")

# 4. Upload processed datasets to Hugging Face
# Authenticate with Hugging Face for uploading
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
else:
    print("WARNING: HF_TOKEN not found in environment. Cannot upload processed data to Hugging Face.")

api = HfApi(token=HF_TOKEN)

files_to_upload_hf = {
    'X_train_smote.csv': os.path.join(PROCESSED_DATA_DIR, 'X_train_smote.csv'),
    'y_train_smote.csv': os.path.join(PROCESSED_DATA_DIR, 'y_train_smote.csv'),
    'X_test.csv': os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'),
    'y_test.csv': os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')
}

print(f"Uploading processed datasets to Hugging Face dataset: {DATASET_REPO}")
for filename, local_path in files_to_upload_hf.items():
    remote_path = f"data/processed/{filename}"
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"✅ Successfully uploaded {filename} to {DATASET_REPO}/{remote_path}")
    except Exception as e:
        print(f"❌ Error uploading {filename}: {e}")

print("--- Data preparation script finished ---")
