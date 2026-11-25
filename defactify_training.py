"""Training, epoch logs, checkpoints, and portable detector export."""

import json
import math
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def binary_metrics(labels, logits):
    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits, dtype=float)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    balanced_accuracy = float(balanced_accuracy_score(labels, predictions))
    return {
        "balanced_accuracy": balanced_accuracy,
        "balanced_error_rate": 1 - balanced_accuracy,
        "precision_ai": float(precision),
        "recall_ai": float(recall),
        "f1_ai": float(f1),
        "auroc": (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) == 2
            else None
        ),
        "confusion_matrix_real_ai": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def run_epoch(
    model, loader, device, optimizer=None, pos_weight=None, auxiliary_weight=0.0
):
    training = optimizer is not None
    model.train(training)
    labels_seen, logits_seen, total_loss = [], [], 0.0
    with torch.set_grad_enabled(training):
        for images, labels, sources in loader:
            images, labels, sources = (
                images.to(device),
                labels.to(device),
                sources.to(device),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            binary_logits, source_logits = model(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                binary_logits, labels, pos_weight=pos_weight
            )
            if source_logits is not None and auxiliary_weight:
                loss += auxiliary_weight * torch.nn.functional.cross_entropy(
                    source_logits, sources
                )
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            labels_seen.extend(labels.cpu().tolist())
            logits_seen.extend(binary_logits.detach().cpu().tolist())
    if not labels_seen:
        raise ValueError("Cannot evaluate an empty data loader")
    return {
        "loss": total_loss / len(labels_seen),
        **binary_metrics(labels_seen, logits_seen),
    }


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_csv_save(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _epoch_row(epoch, train_result, validation_result, elapsed_seconds, is_best):
    row = {
        "epoch": epoch,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "is_best": is_best,
        "balanced_accuracy_gap": (
            train_result["balanced_accuracy"] - validation_result["balanced_accuracy"]
        ),
    }
    for prefix, result in (("train", train_result), ("validation", validation_result)):
        for name, value in result.items():
            row[f"{prefix}_{name}"] = (
                json.dumps(value) if isinstance(value, list) else value
            )
    return row


def train_experiment(
    model,
    model_name,
    model_config,
    train_loader,
    validation_loader,
    device,
    epochs,
    learning_rate,
    auxiliary_weight,
    metadata,
    checkpoints_dir,
    outputs_dir,
    resume=True,
):
    """Train or resume one named experiment, saving best/last and a CSV each epoch."""
    model = model.to(device)
    train_labels = np.asarray(train_loader.dataset.rows["Label_A"], dtype=int)
    real_count, ai_count = np.bincount(train_labels, minlength=2)
    if not real_count or not ai_count:
        raise ValueError("Training split must contain both classes")
    pos_weight = torch.tensor(
        [real_count / ai_count], device=device, dtype=torch.float32
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    checkpoint_dir = Path(checkpoints_dir) / model_name
    last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    csv_path = Path(outputs_dir) / f"epochs_{model_name}.csv"
    run_config = {
        "model_name": model_name,
        "model_config": model_config,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "auxiliary_weight": auxiliary_weight,
        "batch_size": train_loader.batch_size,
        "training_metadata": metadata,
    }
    history, start_epoch, best_epoch, best_score = [], 1, 0, -1.0
    run_id = uuid4().hex
    if last_path.exists():
        if not resume:
            raise FileExistsError(
                f"{last_path} exists; use resume or a new experiment name"
            )
        previous = torch.load(last_path, map_location=device, weights_only=True)
        if previous["run_config"] != run_config:
            raise ValueError(
                f"Checkpoint configuration differs for {model_name}; use a new name"
            )
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint is missing for {model_name}")
        previous_best = torch.load(best_path, map_location="cpu", weights_only=True)
        if (
            previous_best["run_config"] != run_config
            or previous_best["run_id"] != previous["run_id"]
            or previous_best["epoch"] != previous["best_epoch"]
        ):
            raise ValueError(f"Best/last checkpoint mismatch for {model_name}")
        run_id = previous["run_id"]
        history = previous["history"]
        if len(history) != previous["epoch"]:
            raise ValueError(f"History/checkpoint epoch mismatch for {model_name}")
        if not csv_path.exists() or len(pd.read_csv(csv_path)) != len(history):
            _atomic_csv_save(history, csv_path)
        model.load_state_dict(previous["model_state_dict"])
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        start_epoch = previous["epoch"] + 1
        best_epoch = previous["best_epoch"]
        best_score = previous["best_validation_balanced_accuracy"]
        print(
            f"[{model_name}] resuming after epoch {previous['epoch']}; best validation "
            f"balanced accuracy {best_score:.4f} at epoch {best_epoch}"
        )
    print(
        f"[{model_name}] device={device}, train={len(train_loader.dataset):,}, "
        f"validation={len(validation_loader.dataset):,}, epochs={epochs}, "
        f"batch={train_loader.batch_size}, lr={learning_rate:g}, pos_weight={pos_weight.item():.3f}"
    )
    for epoch in range(start_epoch, epochs + 1):
        started = perf_counter()
        train_result = run_epoch(
            model, train_loader, device, optimizer, pos_weight, auxiliary_weight
        )
        validation_result = run_epoch(
            model,
            validation_loader,
            device,
            pos_weight=pos_weight,
            auxiliary_weight=auxiliary_weight,
        )
        elapsed = perf_counter() - started
        score = validation_result["balanced_accuracy"]
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        history.append(
            _epoch_row(epoch, train_result, validation_result, elapsed, improved)
        )
        payload = {
            "run_id": run_id,
            "run_config": run_config,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_balanced_accuracy": best_score,
            "model_state_dict": _to_cpu(model.state_dict()),
            "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
            "history": history,
        }
        if improved:
            _atomic_torch_save(payload, best_path)
        _atomic_torch_save(payload, last_path)
        _atomic_csv_save(history, csv_path)
        print(
            f"[{model_name}] {epoch}/{epochs} ({elapsed:.1f}s) "
            f"train loss={train_result['loss']:.4f}, BA={train_result['balanced_accuracy']:.4f}; "
            f"val loss={validation_result['loss']:.4f}, BA={score:.4f}, "
            f"F1={validation_result['f1_ai']:.4f}, AUROC={validation_result['auroc']:.4f}, "
            f"BA gap={train_result['balanced_accuracy'] - score:+.4f}"
            f"{'  [new best]' if improved else ''}"
        )
    if not best_path.exists():
        raise FileNotFoundError(f"No best checkpoint for {model_name}")
    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    print(
        f"[{model_name}] best epoch={best_epoch}, validation BA={best_score:.4f}; "
        f"CSV={csv_path}; checkpoints={checkpoint_dir}"
    )
    return model, history, best_score


def export_final_model(checkpoints_dir, model_names, epochs, outputs_dir, source_names):
    """Choose the best completed validation run and save an inference bundle."""
    checkpoints_dir, outputs_dir = Path(checkpoints_dir), Path(outputs_dir)
    candidates = []
    expected_metadata = None
    for name in model_names:
        last_path = checkpoints_dir / name / "last.pt"
        best_path = checkpoints_dir / name / "best.pt"
        if not last_path.exists() or not best_path.exists():
            raise FileNotFoundError(
                f"Complete checkpoints required before final export: {name}"
            )
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        run_config = last["run_config"]
        if run_config["model_name"] != name or run_config["epochs"] != epochs:
            raise ValueError(f"Checkpoint name/epoch configuration mismatch for {name}")
        training_metadata = run_config["training_metadata"]
        split_report = training_metadata.get("split_report") or {}
        audit = training_metadata.get("post_repair_audit") or {}
        if not split_report.get("split_fingerprint_sha256") or any(
            audit.get(key) != 0
            for key in (
                "caption_overlap_groups",
                "byte_identical_overlap_groups",
                "invalid_row_count",
            )
        ):
            raise ValueError(
                f"Missing or failed leakage-safe split evidence for {name}"
            )
        if expected_metadata is None:
            expected_metadata = training_metadata
        elif training_metadata != expected_metadata:
            raise ValueError(
                "Experiment checkpoints refer to different datasets or splits"
            )
        if last["epoch"] != epochs:
            raise ValueError(
                f"{name} has completed only {last['epoch']} of {epochs} epochs"
            )
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        if (
            best["run_config"] != run_config
            or best["run_id"] != last["run_id"]
            or best["epoch"] != last["best_epoch"]
            or not math.isclose(
                best["best_validation_balanced_accuracy"],
                last["best_validation_balanced_accuracy"],
            )
        ):
            raise ValueError(f"Stale or mismatched best checkpoint for {name}")
        recorded_score = last["history"][best["epoch"] - 1][
            "validation_balanced_accuracy"
        ]
        if not math.isclose(recorded_score, best["best_validation_balanced_accuracy"]):
            raise ValueError(f"Best score/history mismatch for {name}")
        candidates.append((best["best_validation_balanced_accuracy"], name, best))
    score, chosen_name, chosen = max(candidates, key=lambda item: item[0])
    config = chosen["run_config"]
    bundle = {
        "schema_version": 1,
        "model_name": chosen_name,
        "model_config": config["model_config"],
        "model_state_dict": chosen["model_state_dict"],
        "image_size": config["training_metadata"]["image_size"],
        "threshold": 0.5,
        "source_names": source_names,
    }
    outputs_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = outputs_dir / "final_model.pt"
    weights_path = outputs_dir / "final_weights.pt"
    metadata_path = outputs_dir / "final_model.json"
    _atomic_torch_save(bundle, bundle_path)
    _atomic_torch_save(bundle["model_state_dict"], weights_path)
    metadata = {
        "model_name": chosen_name,
        "model_config": bundle["model_config"],
        "best_epoch": chosen["best_epoch"],
        "validation_balanced_accuracy": score,
        "image_size": bundle["image_size"],
        "threshold": bundle["threshold"],
        "source_names": source_names,
        "training_metadata": config["training_metadata"],
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary, metadata_path)
    print(f"Final model selected using validation only: {chosen_name}, BA={score:.4f}")
    print(
        f"Frontend bundle: {bundle_path}; raw weights: {weights_path}; metadata: {metadata_path}"
    )
    return chosen_name, bundle_path, metadata
