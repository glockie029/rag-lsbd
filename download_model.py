"""Download only BGE data files, pin the commit, and verify published LFS hashes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPOSITORY = "BAAI/bge-small-zh-v1.5"
INITIAL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
MODEL_DIR = ROOT / "models" / "bge-small-zh-v1.5"
LOCK_PATH = ROOT / "embedding.lock.json"
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from huggingface_hub import HfApi, snapshot_download

    existing = json.loads(LOCK_PATH.read_text(encoding="utf-8")) if LOCK_PATH.exists() else {}
    if existing and existing.get("repository") != REPOSITORY:
        raise RuntimeError("The model lock names a different repository")
    revision = existing.get("revision", INITIAL_REVISION)
    info = HfApi(token=False).model_info(REPOSITORY, revision=revision, files_metadata=True)
    if not info.sha:
        raise RuntimeError("The server did not return a model commit")
    print(f"Downloading {REPOSITORY} at commit {info.sha}", flush=True)
    patterns = [
        "config.json", "config_sentence_transformers.json", "modules.json",
        "sentence_bert_config.json", "special_tokens_map.json", "tokenizer_config.json",
        "tokenizer.json", "vocab.txt", "added_tokens.json", "model.safetensors",
        "1_Pooling/config.json", "README.md", "LICENSE", "LICENSE.*",
    ]
    snapshot_download(
        REPOSITORY, revision=info.sha, local_dir=str(MODEL_DIR),
        allow_patterns=patterns, max_workers=2, token=False,
    )
    weights = MODEL_DIR / "model.safetensors"
    if not weights.is_file():
        raise RuntimeError("Expected safetensors weights were not downloaded")
    published = {item.rfilename: item for item in info.siblings or []}
    files = []
    for path in sorted(MODEL_DIR.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(MODEL_DIR).parts:
            continue
        name = path.relative_to(MODEL_DIR).as_posix()
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        remote = published.get(name)
        if remote is None:
            raise RuntimeError(f"Unexpected file in model directory: {name}")
        if remote.size is not None and remote.size != path.stat().st_size:
            raise RuntimeError(f"File size mismatch: {name}")
        lfs = remote.lfs
        expected = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if expected and digest != expected:
            raise RuntimeError(f"Published SHA-256 mismatch: {name}")
        files.append({"path": name, "size_bytes": path.stat().st_size,
                      "sha256": digest, "published_lfs_sha256_verified": bool(expected)})
    lock = {
        "repository": REPOSITORY, "revision": info.sha,
        "model_directory": "models/bge-small-zh-v1.5", "device": "cpu",
        "embedding_dimension": 512, "normalize_embeddings": True,
        "query_instruction": "为这个句子生成表示以用于检索相关文章：",
        "license": "MIT", "files": files,
    }
    if existing:
        expected_files = {f["path"]: f["sha256"] for f in existing.get("files", [])}
        if expected_files and expected_files != {f["path"]: f["sha256"] for f in files}:
            raise RuntimeError("Downloaded files differ from the existing model lock")
    LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"revision": info.sha, "files": len(files),
                      "total_bytes": sum(f["size_bytes"] for f in files),
                      "model_directory": str(MODEL_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
