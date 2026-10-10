"""Create or safely update sharadphatak/superkat_mlOps using a Colab secret.

Managed files are recorded in a manifest; unrelated files are preserved. Ref
updates are non-forced, so concurrent changes or branch policies stop publishing.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import requests

OWNER = "sharadphatak"
NAME = "superkat_mlOps"
URL = f"https://github.com/{OWNER}/{NAME}"
MANIFEST = "superkart-publish-manifest.json"
EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", "submission", "processed", ".ipynb_checkpoints"}


def collect_files(root):
    root = Path(root)
    required = ["app.py", "src/train.py", "requirements.txt", ".github/workflows/pipeline.yml"]
    if any(not (root / name).is_file() for name in required):
        raise ValueError("Run publisher from the generated SuperKart project root")
    files = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or path.is_symlink() or any(part in EXCLUDED or part.startswith(".venv") for part in rel.parts):
            continue
        if path.name.startswith(".env") or path.name in {"secrets.toml", MANIFEST, "github_publish_receipt.json"} or path.suffix in {".log", ".pyc", ".zip"}:
            continue
        if path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError(f"File exceeds 50 MiB upload limit: {rel}")
        files[rel.as_posix()] = path.read_bytes()
    manifest = {"project": "SuperKart", "repository": f"{OWNER}/{NAME}", "managed_files": sorted(files)}
    files[MANIFEST] = (json.dumps(manifest, indent=2) + "\n").encode()
    return files


def blob_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def publish(root=None, session=None):
    # Validate local files before creating any remote objects.
    files = collect_files(root or Path.cwd())
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("Add GH_TOKEN to Colab Secrets and enable notebook access; never paste it into code")
    session = session or requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                            "X-GitHub-Api-Version": "2022-11-28"})
    prefix = f"/repos/{OWNER}/{NAME}"

    def call(method, route, payload=None, allowed=()):
        r = session.request(method, "https://api.github.com" + route, json=payload, timeout=60)
        if r.status_code in allowed:
            return None
        if r.status_code >= 400:
            raise RuntimeError(f"GitHub {method} {route}: HTTP {r.status_code}; check token contents/workflows permissions, branch rules or concurrent updates")
        return r.json() if r.content else {}

    login = call("GET", "/user")["login"]
    if login.lower() != OWNER.lower():
        raise RuntimeError(f"Use a token for {OWNER}; authenticated account is {login}")
    repo = call("GET", prefix, allowed=(404,))
    if repo is None:
        # Visibility applies only on creation, never changes an existing repo.
        repo = call("POST", "/user/repos", {"name": NAME, "private": os.environ.get("GH_PRIVATE", "true").lower() == "true",
                    "auto_init": True, "description": "SuperKart academic MLOps pipeline"})
    if repo.get("archived"):
        raise RuntimeError("Repository is archived; publishing is unavailable")
    branch = "main"
    ref = call("GET", prefix + "/git/ref/heads/main", allowed=(404, 409))
    create_main = False
    if ref is None:
        default = repo.get("default_branch", "main")
        ref = call("GET", prefix + f"/git/ref/heads/{default}", allowed=(404, 409))
        if ref is None:
            # Git Data APIs cannot bootstrap an entirely empty repository.
            call("PUT", prefix + "/contents/README.md", {"message": "Initialize SuperKart repository",
                 "content": base64.b64encode(files["README.md"]).decode(), "branch": "main"})
            ref = call("GET", prefix + "/git/ref/heads/main")
        else:
            create_main = default != "main"
    parent = ref["object"]["sha"]
    parent_commit = call("GET", prefix + f"/git/commits/{parent}")
    remote = call("GET", prefix + f"/git/trees/{parent_commit['tree']['sha']}?recursive=1")
    if remote.get("truncated"):
        raise RuntimeError("Remote tree is truncated; cannot safely inspect existing files")
    remote_blobs = {x["path"]: x for x in remote["tree"] if x["type"] == "blob"}
    managed = set()
    if MANIFEST in remote_blobs:
        stored = call("GET", prefix + f"/git/blobs/{remote_blobs[MANIFEST]['sha']}")
        old = json.loads(base64.b64decode(stored["content"]))
        if old.get("project") != "SuperKart" or old.get("repository") != f"{OWNER}/{NAME}":
            raise RuntimeError("Remote publish manifest belongs to another project")
        managed = set(old["managed_files"]) | {MANIFEST}
    for path, data in files.items():
        existing = remote_blobs.get(path)
        if existing and existing["sha"] != blob_sha(data) and path not in managed and path != "README.md":
            raise RuntimeError(f"Unmanaged remote file conflicts: {path}. Review it before first publication")
    updates = []
    for path, data in files.items():
        if path in remote_blobs and remote_blobs[path]["sha"] == blob_sha(data):
            continue
        blob = call("POST", prefix + "/git/blobs", {"content": base64.b64encode(data).decode(), "encoding": "base64"})
        updates.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    commit_sha = parent
    if updates:
        tree = call("POST", prefix + "/git/trees", {"base_tree": parent_commit["tree"]["sha"], "tree": updates})
        commit = call("POST", prefix + "/git/commits", {"message": "Update SuperKart MLOps codebase", "tree": tree["sha"], "parents": [parent]})
        commit_sha = commit["sha"]
    if create_main:
        call("POST", prefix + "/git/refs", {"ref": "refs/heads/main", "sha": commit_sha})
    elif updates:
        call("PATCH", prefix + "/git/refs/heads/main", {"sha": commit_sha, "force": False})
    result = {"repository_url": URL, "branch": branch, "commit_sha": commit_sha, "files_updated": len(updates)}
    # Actual publish receipt, useful for notebook evidence. No secret values.
    Path(root or Path.cwd()).joinpath("reports").mkdir(exist_ok=True)
    Path(root or Path.cwd()).joinpath("reports/github_publish_receipt.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Verify workflow: {URL}/actions")
    return result


if __name__ == "__main__":
    publish()
