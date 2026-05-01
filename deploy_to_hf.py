"""
deploy_to_hf.py — Push the Predictive Maintenance app to Hugging Face Spaces
==============================================================================

Usage:
    python deploy_to_hf.py --token YOUR_HF_TOKEN --space YOUR_USERNAME/predictive-maintenance

Get your token from: https://huggingface.co/settings/tokens
  - Make sure it has "Write" permissions
"""

import argparse
import os
from huggingface_hub import HfApi, login

# Files to upload (relative to project root)
FILES_TO_UPLOAD = [
    "app.py",
    "pipeline.py",
    "router.py",
    "requirements.txt",
    "README.md",
    "DETAILS.md",
    "sample_batch.csv",
    "Predictive_M.csv",
    "Trained_models/standard_scaler.joblib",
    "Trained_models/feature_columns.json",
    "Trained_models/binary_decision_tree_baseline_smote_8features_threshold_0p50.joblib",
    "Trained_models/binary_decision_tree_cost_sensitive_smote_8features_threshold_0p50.joblib",
    "Trained_models/multiclass_decision_tree_priority_encoded_scaled_original_features.joblib",
    "Trained_models/multilabel_decision_tree_multioutput_scaled_original_features.joblib",
]


def main():
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Spaces")
    parser.add_argument("--token", required=True, help="HuggingFace write token")
    parser.add_argument("--space", required=True,
                        help="Space ID, e.g. 'username/predictive-maintenance'")
    args = parser.parse_args()

    # Authenticate
    login(token=args.token)
    api = HfApi()
    user_info = api.whoami()
    print(f"Authenticated as: {user_info['name']}")

    # Create the Space if it doesn't exist
    repo_id = args.space
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False,
        )
        print(f"Space '{repo_id}' ready.")
    except Exception as e:
        print(f"Note: {e}")

    # Upload each file
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"\nUploading {len(FILES_TO_UPLOAD)} files to {repo_id}...")

    for rel_path in FILES_TO_UPLOAD:
        local_path = os.path.join(project_root, rel_path)
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {rel_path}")
            continue

        print(f"  Uploading: {rel_path} ({os.path.getsize(local_path):,} bytes)")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=rel_path,
            repo_id=repo_id,
            repo_type="space",
        )

    print(f"\n{'='*60}")
    print(f"  DEPLOYMENT COMPLETE!")
    print(f"  View your Space: https://huggingface.co/spaces/{repo_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
