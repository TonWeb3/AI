"""Kronos futures bot.

Importing this package pins the Hugging Face cache INSIDE the repo, at
`model/weights/`, so the downloaded Kronos weights sit next to the code that
uses them instead of in a per-user cache under the home directory. This runs
before any submodule (and therefore before `huggingface_hub`) is imported.

An HF_HOME already present in the environment still wins, so a deployment can
point the cache at a mounted volume without editing code.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(ROOT, "model", "weights")

os.environ.setdefault("HF_HOME", WEIGHTS_DIR)
# Windows without Developer Mode cannot create the symlinks the HF cache
# normally uses; it falls back to real copies and warns loudly. Silence it.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
