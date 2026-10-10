"""Publish only the frontend and trusted trained model; never upload data/secrets."""
import os
import tempfile
from pathlib import Path
import shutil
from huggingface_hub import HfApi


def sync():
    token = os.environ.get("HF_TOKEN")
    repo = os.environ.get("HF_SPACE_ID")
    if not token or not repo or repo.count("/") != 1:
        raise RuntimeError("Configure HF_TOKEN secret and HF_SPACE_ID=owner/space")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    info = api.space_info(repo)
    if info.private or info.sdk != "docker":
        raise RuntimeError("Configured Space must be public and use the Docker SDK; correct its settings before publishing")
    with tempfile.TemporaryDirectory() as temp:
        dest = Path(temp)
        for file in ["app.py", "requirements.txt", "Dockerfile"]:
            shutil.copy2(file, dest / file)
        for folder in ["src", "models"]:
            shutil.copytree(folder, dest / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (dest / "README.md").write_text("---\ntitle: SuperKart\nemoji: 🛒\ncolorFrom: blue\ncolorTo: green\nsdk: docker\napp_port: 7860\n---\n\nProduct/store revenue regression; no temporal forecast claim.\n")
        api.upload_folder(repo_id=repo, repo_type="space", folder_path=dest,
                          commit_message="Publish verified SuperKart app and model")
    print(f"Published source: https://huggingface.co/spaces/{repo}; verify build and sample prediction in browser.")


if __name__ == "__main__":
    sync()
