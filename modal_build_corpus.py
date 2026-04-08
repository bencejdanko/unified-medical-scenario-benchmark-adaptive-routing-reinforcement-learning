"""
Modal: Build Medical RAG Corpus
================================
Downloads pre-chunked MedRAG corpora from HuggingFace, computes
dense embeddings with PubMedBERT, and uploads to HuggingFace.

Corpora (from https://arxiv.org/abs/2402.13178):
  1. MedRAG/textbooks - 18 medical textbooks, pre-chunked (~125k snippets)
  2. MedRAG/pubmed   - PubMed abstracts, pre-chunked

Embedding model: NeuML/pubmedbert-base-embeddings (768-dim, PubMed-tuned)

Usage:
    modal run modal_build_corpus.py
    modal run modal_build_corpus.py --hf-repo bencejdanko/medical-rag-corpus
    modal run modal_build_corpus.py --pubmed-files 20
"""

import modal

# ---------------------------------------------------------------------------
# Modal configuration
# ---------------------------------------------------------------------------

VOLUME_NAME = "medqa-data-volume"
HF_SECRET = "my-huggingface-secret"
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
DEFAULT_HF_REPO = "bencejdanko/medical-rag-corpus"
DEFAULT_PUBMED_FILES = 20  # ~300k PubMed chunks (each file ~15k lines)

volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub>=0.24",
        "sentence-transformers>=3.0",
        "transformers>=4.40",
        "torch>=2.2",
        "numpy>=1.26",
        "tqdm",
    )
)

app = modal.App("medical-rag-corpus", image=image)


# ---------------------------------------------------------------------------
# Corpus downloading helpers (run inside Modal)
# ---------------------------------------------------------------------------

def download_textbooks(out_dir: str) -> list[dict]:
    """Download pre-chunked medical textbooks from MedRAG/textbooks on HuggingFace."""
    import json
    import os
    from pathlib import Path
    from huggingface_hub import HfApi, hf_hub_download

    cache = Path(out_dir) / "textbooks_chunks.jsonl"
    if cache.exists():
        print(f"  [textbooks] Loading from cache ({cache})")
        docs = []
        with open(cache) as f:
            for line in f:
                docs.append(json.loads(line))
        return docs

    print("  [textbooks] Downloading pre-chunked data from MedRAG/textbooks...")
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    repo_files = api.list_repo_files("MedRAG/textbooks", repo_type="dataset")
    chunk_files = sorted(f for f in repo_files if f.startswith("chunk/") and f.endswith(".jsonl"))
    print(f"  [textbooks] Found {len(chunk_files)} chunk files")

    docs = []
    for cf in chunk_files:
        local_path = hf_hub_download("MedRAG/textbooks", cf, repo_type="dataset", token=token)
        with open(local_path) as f:
            for line in f:
                row = json.loads(line)
                text = row.get("content", "").strip()
                if len(text) < 30:
                    continue
                docs.append({
                    "id": row.get("id", ""),
                    "source": "textbooks",
                    "title": row.get("title", ""),
                    "text": text,
                })
        print(f"    {cf}: {len(docs)} total chunks so far")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    print(f"  [textbooks] Total: {len(docs)} chunks")
    return docs


def download_pubmed(out_dir: str, num_files: int = DEFAULT_PUBMED_FILES) -> list[dict]:
    """Download pre-chunked PubMed abstracts from MedRAG/pubmed (subset)."""
    import json
    import os
    from pathlib import Path
    from huggingface_hub import HfApi, hf_hub_download

    cache = Path(out_dir) / f"pubmed_chunks_n{num_files}.jsonl"
    if cache.exists():
        print(f"  [pubmed] Loading from cache ({cache})")
        docs = []
        with open(cache) as f:
            for line in f:
                docs.append(json.loads(line))
        return docs

    print(f"  [pubmed] Downloading first {num_files} chunk files from MedRAG/pubmed...")
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    repo_files = api.list_repo_files("MedRAG/pubmed", repo_type="dataset")
    chunk_files = sorted(f for f in repo_files if f.startswith("chunk/") and f.endswith(".jsonl"))
    chunk_files = chunk_files[:num_files]
    print(f"  [pubmed] Downloading {len(chunk_files)} of {len([f for f in repo_files if f.startswith('chunk/')])} files")

    docs = []
    for cf in chunk_files:
        local_path = hf_hub_download("MedRAG/pubmed", cf, repo_type="dataset", token=token)
        with open(local_path) as f:
            for line in f:
                row = json.loads(line)
                text = row.get("content", "").strip()
                if len(text) < 30:
                    continue
                docs.append({
                    "id": row.get("id", ""),
                    "source": "pubmed",
                    "title": row.get("title", ""),
                    "text": text,
                })
        print(f"    {cf}: {len(docs)} total chunks so far")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    print(f"  [pubmed] Total: {len(docs)} chunks")
    return docs


# ---------------------------------------------------------------------------
# Main Modal function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
    timeout=7200,
    memory=32768,
)
def build_corpus(hf_repo: str = DEFAULT_HF_REPO, pubmed_files: int = DEFAULT_PUBMED_FILES):
    """Download MedRAG corpora, embed with PubMedBERT, and upload to HuggingFace."""
    import json
    import os
    from pathlib import Path

    import numpy as np
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    work_dir = "/data/rag_corpus"
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Download corpora ----
    print("=" * 60)
    print("Step 1: Downloading MedRAG corpora")
    print("=" * 60)

    all_chunks = []
    all_chunks.extend(download_textbooks(work_dir))
    all_chunks.extend(download_pubmed(work_dir, num_files=pubmed_files))

    print(f"\nTotal chunks: {len(all_chunks)}")
    for src in ["textbooks", "pubmed"]:
        count = sum(1 for d in all_chunks if d["source"] == src)
        print(f"  {src}: {count}")

    # Save unified chunks file
    chunks_path = Path(work_dir) / "chunks.jsonl"
    with open(chunks_path, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    # ---- Step 2: Compute embeddings ----
    print("\n" + "=" * 60)
    print("Step 2: Computing embeddings with PubMedBERT")
    print("=" * 60)

    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    texts = [c["text"] for c in all_chunks]

    batch_size = 256
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    emb_path = Path(work_dir) / "embeddings.npy"
    np.save(emb_path, embeddings)

    # ---- Step 3: Upload to HuggingFace ----
    print("\n" + "=" * 60)
    print("Step 3: Uploading to HuggingFace")
    print("=" * 60)

    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)

    api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True, private=False)

    api.upload_file(
        path_or_fileobj=str(chunks_path),
        path_in_repo="chunks.jsonl",
        repo_id=hf_repo,
        repo_type="dataset",
    )
    print(f"  Uploaded chunks.jsonl ({chunks_path.stat().st_size / 1e6:.1f} MB)")

    api.upload_file(
        path_or_fileobj=str(emb_path),
        path_in_repo="embeddings.npy",
        repo_id=hf_repo,
        repo_type="dataset",
    )
    print(f"  Uploaded embeddings.npy ({emb_path.stat().st_size / 1e6:.1f} MB)")

    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": int(embeddings.shape[1]),
        "num_chunks": len(all_chunks),
        "pubmed_files_used": pubmed_files,
        "sources": {
            src: sum(1 for c in all_chunks if c["source"] == src)
            for src in set(c["source"] for c in all_chunks)
        },
    }
    meta_path = Path(work_dir) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    api.upload_file(
        path_or_fileobj=str(meta_path),
        path_in_repo="metadata.json",
        repo_id=hf_repo,
        repo_type="dataset",
    )

    print(f"\n  Dataset published to: https://huggingface.co/datasets/{hf_repo}")
    print(f"  Metadata: {json.dumps(metadata, indent=2)}")

    volume.commit()

    return metadata


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(hf_repo: str = DEFAULT_HF_REPO, pubmed_files: int = DEFAULT_PUBMED_FILES):
    print(f"Building medical RAG corpus -> {hf_repo}")
    print(f"  PubMed files: {pubmed_files}")
    metadata = build_corpus.remote(hf_repo=hf_repo, pubmed_files=pubmed_files)
    print(f"\nDone! {metadata['num_chunks']} chunks across {len(metadata['sources'])} sources")
    print(f"Dataset: https://huggingface.co/datasets/{hf_repo}")
