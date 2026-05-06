"""Compute episodic-memory embeddings on Modal and upload retrieval artifacts.

Usage:
    modal run modal_embed_episodic_memory.py
"""

from __future__ import annotations

import modal


HF_SECRET = "my-huggingface-secret"
DEFAULT_DATASET_REPO = "bdanko/umsb-episodic-memory"
DEFAULT_MODEL_REPO = "bdanko/umsb-mpnet-episodic-memory"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.20",
        "huggingface_hub>=0.24",
        "numpy>=1.26",
        "sentence-transformers>=3.0",
        "torch>=2.2",
        "transformers>=4.44",
        "tqdm",
    )
)

app = modal.App("umsb-episodic-memory-embed", image=image)


@app.function(gpu="A10G", secrets=[modal.Secret.from_name(HF_SECRET)], timeout=3600, memory=32768)
def embed_memory(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    model_repo: str = DEFAULT_MODEL_REPO,
    split: str = "train",
    batch_size: int = 128,
):
    import json
    import os
    from pathlib import Path

    import numpy as np
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    token = os.environ["HF_TOKEN"]
    ds = load_dataset(dataset_repo, split=split, token=token)
    rows = [dict(row) for row in ds if row.get("text")]
    texts = [row["text"] for row in rows]
    print(f"Embedding {len(texts)} memory chunks from {dataset_repo}:{split}")

    model = SentenceTransformer(model_repo, device="cuda", token=token)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    out_dir = Path("/tmp/umsb_memory_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "memory_chunks.jsonl"
    emb_path = out_dir / "memory_embeddings.npy"
    meta_path = out_dir / "memory_embedding_metadata.json"

    with chunks_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    np.save(emb_path, embeddings)
    metadata = {
        "dataset_repo": dataset_repo,
        "dataset_split": split,
        "model_repo": model_repo,
        "num_chunks": len(rows),
        "embedding_dim": int(embeddings.shape[1]) if len(rows) else 0,
        "normalized": True,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    api = HfApi(token=token)
    for path in [chunks_path, emb_path, meta_path]:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=dataset_repo,
            repo_type="dataset",
        )
        print(f"Uploaded {path.name}")
    return metadata


@app.local_entrypoint()
def main(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    model_repo: str = DEFAULT_MODEL_REPO,
    split: str = "train",
    batch_size: int = 128,
):
    embed_memory.remote(dataset_repo, model_repo, split, batch_size)
