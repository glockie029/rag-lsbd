"""Offline CPU embedding, semantic retrieval, and cross-process persistence check."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DB_PATH = ROOT / "data" / "dependency-smoke"
COLLECTION = "bge_dependency_smoke"


def make_client():
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(DB_PATH), settings=Settings(anonymized_telemetry=False))


def check_persisted() -> None:
    collection = make_client().get_collection(COLLECTION, embedding_function=None)
    assert collection.count() == 3, "Unexpected persisted record count"
    loaded = collection.get(ids=["rag"], include=["documents", "metadatas", "embeddings"])
    assert loaded["metadatas"][0]["source"] == "dependency-smoke-synthetic"
    assert "检索" in loaded["documents"][0]
    assert len(loaded["embeddings"][0]) == 512
    print(json.dumps({"cross_process_persistence": True, "record_count": 3}))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--read-persisted", action="store_true")
    args = parser.parse_args()
    if args.read_persisted:
        check_persisted()
        return

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert torch.version.cuda is None, "Expected a CPU-only PyTorch build"
    lock = json.loads((ROOT / "embedding.lock.json").read_text(encoding="utf-8"))
    model_dir = ROOT / lock["model_directory"]
    for item in lock["files"]:
        path = model_dir / item["path"]
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == item["sha256"], item["path"]
    model = SentenceTransformer(str(model_dir), device="cpu", local_files_only=True, trust_remote_code=False)
    documents = [
        "RAG 是检索增强生成：先从知识库检索与问题相关的文档，再把资料交给语言模型生成有依据的回答。",
        "番茄炒蛋的做法是先炒鸡蛋，再加入番茄翻炒，最后调味装盘。",
        "自行车轮胎漏气时，需要检查内胎，找到破损位置并修补或更换。",
    ]
    ids = ["rag", "cooking", "bicycle"]
    query = "怎样让大模型先查找本地资料，再根据资料回答问题？"
    vectors = model.encode(documents, batch_size=2, normalize_embeddings=True, show_progress_bar=False)
    query_vector = model.encode([lock["query_instruction"] + query], batch_size=1,
                                normalize_embeddings=True, show_progress_bar=False)
    assert vectors.shape == (3, 512)
    assert query_vector.shape == (1, 512)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    collection = make_client().get_or_create_collection(
        COLLECTION, embedding_function=None, metadata={"hnsw:space": "cosine"})
    collection.upsert(ids=ids, documents=documents, embeddings=vectors.tolist(),
                      metadatas=[{"source": "dependency-smoke-synthetic"} for _ in ids])
    result = collection.query(query_embeddings=query_vector.tolist(), n_results=3,
                              include=["documents", "metadatas", "distances"])
    assert result["ids"][0][0] == "rag", "Semantic retrieval did not select the expected document"
    reopen = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--read-persisted"],
                            capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    if reopen.returncode:
        raise RuntimeError("Cross-process persistence check failed: " + reopen.stderr[-4000:])
    persisted = json.loads(reopen.stdout.strip().splitlines()[-1])
    output = {
        "checked_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "python": sys.version, "python_executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "sentence-transformers", "transformers", "chromadb", "httpx", "numpy")},
        "torch_cuda_version": torch.version.cuda, "embedding_device": str(model.device),
        "embedding_model": lock["repository"], "embedding_revision": lock["revision"],
        "embedding_shape": list(vectors.shape), "normalized": True,
        "offline_model_load": True, "query": query,
        "retrieved_ids": result["ids"][0], "cosine_distances": result["distances"][0],
        "top_document": result["documents"][0][0], **persisted,
        "database_path": str(DB_PATH),
        "scope": "Dependency smoke test only; no generation-model request or RAG application built.",
    }
    evidence = ROOT / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "environment-verification.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
