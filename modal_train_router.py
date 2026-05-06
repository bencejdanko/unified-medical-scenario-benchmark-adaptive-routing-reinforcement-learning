"""Train and publish a DistilBERT multi-head router on Modal.

Usage:
    modal run modal_train_router.py
    modal run modal_train_router.py --dataset-repo bdanko/umsb-routing-classification --model-repo bdanko/umsb-distilbert-router
"""

from __future__ import annotations

from dataclasses import dataclass

import modal


HF_SECRET = "my-huggingface-secret"
DEFAULT_DATASET_REPO = "bdanko/umsb-routing-classification"
DEFAULT_MODEL_REPO = "bdanko/umsb-distilbert-router"
BASE_MODEL = "distilbert/distilbert-base-uncased"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.20",
        "evaluate>=0.4",
        "huggingface_hub>=0.24",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "torch>=2.2",
        "transformers>=4.44",
        "accelerate>=0.33",
    )
)

app = modal.App("umsb-distilbert-router-train", image=image)


@app.function(gpu="A10G", secrets=[modal.Secret.from_name(HF_SECRET)], timeout=7200, memory=32768)
def train_router(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    model_repo: str = DEFAULT_MODEL_REPO,
    epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
):
    import json
    import os
    from pathlib import Path

    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from sklearn.metrics import accuracy_score, f1_score
    from torch import nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer, Trainer, TrainingArguments
    from transformers.modeling_outputs import ModelOutput

    token = os.environ["HF_TOKEN"]
    ds = load_dataset(dataset_repo, token=token)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=384)

    tokenized = ds.map(tokenize, batched=True)
    keep = [
        "input_ids",
        "attention_mask",
        "model_label_id",
        "tool_label_id",
        "prompt_label_id",
    ]
    tokenized = tokenized.remove_columns([c for c in tokenized["train"].column_names if c not in keep])
    tokenized.set_format("torch")

    @dataclass
    class MultiHeadOutput(ModelOutput):
        loss: torch.Tensor | None = None
        logits: dict[str, torch.Tensor] | None = None

    class DistilBertRouter(nn.Module):
        def __init__(self):
            super().__init__()
            config = AutoConfig.from_pretrained(BASE_MODEL)
            self.encoder = AutoModel.from_pretrained(BASE_MODEL, config=config)
            hidden = config.dim
            self.dropout = nn.Dropout(config.seq_classif_dropout)
            self.model_head = nn.Linear(hidden, 3)
            self.tool_head = nn.Linear(hidden, 2)
            self.prompt_head = nn.Linear(hidden, 3)
            self.config = config

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            model_label_id=None,
            tool_label_id=None,
            prompt_label_id=None,
        ):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.dropout(outputs.last_hidden_state[:, 0])
            logits = {
                "model": self.model_head(pooled),
                "tool": self.tool_head(pooled),
                "prompt": self.prompt_head(pooled),
            }
            loss = None
            if model_label_id is not None:
                ce = nn.CrossEntropyLoss()
                loss = (
                    ce(logits["model"], model_label_id)
                    + ce(logits["tool"], tool_label_id)
                    + ce(logits["prompt"], prompt_label_id)
                ) / 3.0
            return MultiHeadOutput(loss=loss, logits=logits)

        def save_pretrained(self, save_directory: str):
            path = Path(save_directory)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), path / "pytorch_model.bin")
            self.config.save_pretrained(path)
            labels = {
                "base_model": BASE_MODEL,
                "model_labels": [
                    "google/gemini-3-flash-preview",
                    "openai/gpt-oss-120b:nitro",
                    "google/gemma-4-31b-it",
                ],
                "tool_labels": ["none", "fhir"],
                "prompt_labels": ["mmlu_medical_json", "healthbench_default", "medagentbench_fhir"],
            }
            (path / "router_labels.json").write_text(json.dumps(labels, indent=2))

    class RouterTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            with torch.no_grad():
                outputs = model(**inputs)
            labels = torch.stack(
                [inputs["model_label_id"], inputs["tool_label_id"], inputs["prompt_label_id"]],
                dim=1,
            )
            logits = torch.stack(
                [
                    outputs.logits["model"].argmax(dim=-1),
                    outputs.logits["tool"].argmax(dim=-1),
                    outputs.logits["prompt"].argmax(dim=-1),
                ],
                dim=1,
            )
            return outputs.loss, logits, labels

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        metrics = {}
        for idx, name in enumerate(["model", "tool", "prompt"]):
            metrics[f"{name}_accuracy"] = accuracy_score(labels[:, idx], preds[:, idx])
            metrics[f"{name}_f1_macro"] = f1_score(labels[:, idx], preds[:, idx], average="macro")
        metrics["joint_accuracy"] = float(np.all(preds == labels, axis=1).mean())
        return metrics

    out_dir = "/tmp/umsb_router"
    final_dir = "/tmp/umsb_router_final"
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_joint_accuracy",
        greater_is_better=True,
        logging_steps=25,
        report_to=[],
    )
    model = DistilBertRouter()
    trainer = RouterTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate(tokenized["test"])
    print(json.dumps(metrics, indent=2))

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    api = HfApi(token=token)
    api.create_repo(model_repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=final_dir, repo_id=model_repo, repo_type="model")
    print(f"Uploaded router model to https://huggingface.co/{model_repo}")
    return metrics


@app.local_entrypoint()
def main(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    model_repo: str = DEFAULT_MODEL_REPO,
    epochs: int = 4,
    batch_size: int = 32,
):
    train_router.remote(dataset_repo, model_repo, epochs, batch_size)
