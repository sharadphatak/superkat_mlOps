"""Fail early when serialized-model dependencies differ from the shared pins."""
import importlib.metadata
import platform
import sys
from pathlib import Path


def check():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Use Python 3.12; found {platform.python_version()}")
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    for line in requirements.read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            package, expected = line.split("==")
            actual = importlib.metadata.version(package)
            assert actual == expected, f"{package}: expected {expected}, got {actual}"
            print(f"{package}=={actual}: OK")


if __name__ == "__main__":
    check()
