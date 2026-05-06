"""Fine-tune all-mpnet-base-v2 on UMSB episodic memory chunks and publish it.

Usage:
    modal run modal_train_episodic_memory.py
"""

from __future__ import annotations

import modal


HF_SECRET = "my-huggingface-secret"
DEFAULT_DATASET_REPO = "bdanko/umsb-episodic-memory"
DEFAULT_MODEL_REPO = "bdanko/umsb-mpnet-episodic-memory"
BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.20",
        "huggingface_hub>=0.24",
        "sentence-transformers>=3.0",
        "torch>=2.2",
        "transformers>=4.44",
        "accelerate>=0.33",
    )
)

app = modal.App("umsb-episodic-memory-train", image=image)


@app.function(gpu="A10G", secrets=[modal.Secret.from_name(HF_SECRET)], timeout=7200, memory=32768)
def train_memory_encoder(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    model_repo: str = DEFAULT_MODEL_REPO,
    epochs: int = 2,
    batch_size: int = 32,
):
    import os

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    token = os.environ["HF_TOKEN"]
    ds = load_dataset(dataset_repo, token=token)
    train_ds = ds["train"].filter(lambda row: bool(row["text"]) and len(row["text"]) > 30)

    # Positive pairs are two independently dropout-augmented views of the same memory.
    pair_ds = train_ds.map(lambda row: {"anchor": row["text"], "positive": row["text"]})
    model = SentenceTransformer(BASE_MODEL)
    loss = MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir="/tmp/umsb_memory_encoder",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        fp16=True,
        save_strategy="epoch",
        logging_steps=25,
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=pair_ds.select_columns(["anchor", "positive"]),
        loss=loss,
    )
    trainer.train()
    model.save_pretrained("/tmp/umsb_memory_encoder/final")
    api = HfApi(token=token)
    api.create_repo(model_repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path="/tmp/umsb_memory_encoder/final", repo_id=model_repo, repo_type="model")
    print(f"Uploaded memory encoder to https://huggingface.co/{model_repo}")


@app.local_entrypoint()
def main(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    model_repo: str = DEFAULT_MODEL_REPO,
    epochs: int = 2,
    batch_size: int = 32,
):
    train_memory_encoder.remote(dataset_repo, model_repo, epochs, batch_size)
