
import os
import sys
from huggingface_hub import HfApi, login, create_repo

# ── Authenticate with Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

login(token=HF_TOKEN, add_to_git_credential=False)

HF_USERNAME = "Bharath1434"
SPACE_REPO  = "{}/tourism-package-predictor".format(HF_USERNAME)

MASTER_DIR           = "visit_with_us_mlops"
APP_DIR_LOCAL        = os.path.join(MASTER_DIR, "app")
DEPLOYMENT_DIR_LOCAL = os.path.join(MASTER_DIR, "deployment")

# ── Create or verify the Hugging Face Space
print("Creating/verifying Space: {}".format(SPACE_REPO))
try:
    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker",      # ← FIXED: streamlit is no longer a valid SDK option
        exist_ok=True,
        private=False
    )
    print("Space ready: https://huggingface.co/spaces/{}".format(SPACE_REPO))
except Exception as e:
    print("Error creating space: {}".format(e))
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# ── Map local file paths to their remote names in the Space repo
files_to_upload = {
    os.path.join(APP_DIR_LOCAL, "app.py"):            "app.py",
    os.path.join(APP_DIR_LOCAL, "requirements.txt"):  "requirements.txt",
    os.path.join(DEPLOYMENT_DIR_LOCAL, "Dockerfile"): "Dockerfile",
}

# ── Upload each file
for local_path, remote_name in files_to_upload.items():
    if not os.path.exists(local_path):
        print("MISSING FILE - skipping: {}".format(local_path))
        continue
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_name,
            repo_id=SPACE_REPO,
            repo_type="space",
        )
        print("Uploaded: {} -> {}".format(os.path.basename(local_path), remote_name))
    except Exception as e:
        print("Upload failed for {}: {}".format(local_path, e))

print("\nAll deployment files pushed!")
print("App URL: https://huggingface.co/spaces/{}".format(SPACE_REPO))
