#!/usr/bin/env python3
# ============================================
# Fine-tune SER WavLM → DAiSEE (4 états) ou ancre IEMOCAP ER (neu/hap/ang/sad)
# ============================================
"""
Usage — modèle principal DAiSEE **complet (retenu)** :
  python scripts/finetune_ser_wavlm.py --max-samples 1200 --epochs 5 --unfreeze-all --augment --out-dir models/ser_wavlm_daisee_full

Usage — essai tête seule (repli si le full est absent) :
  python scripts/finetune_ser_wavlm.py --max-samples 800 --epochs 5 --augment

Usage — ancre hybride IEMOCAP (neu/hap/ang/sad → mapping DAiSEE au runtime) :
  python scripts/finetune_ser_wavlm.py --target iemocap_er --max-samples 1200 --epochs 5 --augment

Produit :
  models/ser_wavlm_daisee_full/     (--unfreeze-all, prioritaire au runtime)
  models/ser_wavlm_daisee/          (tête seule, repli)
  models/ser_wavlm_iemocap_er/      (--target iemocap_er)

Runtime : backend/ser_module.py charge d'abord ser_wavlm_daisee_full, sinon ser_wavlm_daisee.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from iemocap_audio import charger_segment_wav
from iemocap_parser import mapping_iemocap_daisee as mapping_ser_daisee

RESULTS_DIR = os.path.join(ROOT, "results")
BASE_MODEL = "microsoft/wavlm-base"

ETATS_DAISEE = ["Boredom", "Confusion", "Engagement", "Frustration"]
EMOTIONS_IEMOCAP_ER = ["neu", "hap", "ang", "sad"]

OUT_DIRS = {
    "daisee": os.path.join(ROOT, "models", "ser_wavlm_daisee"),
    "iemocap_er": os.path.join(ROOT, "models", "ser_wavlm_iemocap_er"),
}
METRICS_FILES = {
    "daisee": os.path.join(RESULTS_DIR, "ser_wavlm_daisee_metrics.json"),
    "iemocap_er": os.path.join(RESULTS_DIR, "ser_wavlm_iemocap_er_metrics.json"),
}


def _backbone_module(model):
    for name in ("wavlm", "wav2vec2", "hubert"):
        if hasattr(model, name):
            return getattr(model, name)
    return None


def charger_corpus_daisee(max_samples: int, seed: int, min_duree: float):
    label2id = {e: i for i, e in enumerate(ETATS_DAISEE)}
    csv_path = os.path.join(ROOT, "data", "iemocap_parsed.csv")
    df = pd.read_csv(csv_path)
    df = df[~df["emotion"].isin(["xxx", "oth"])].copy()
    df = df[df["duree"] >= min_duree]
    if "wav_exists" in df.columns:
        df = df[df["wav_exists"]]

    def _etat(row):
        if pd.notna(row.get("etat_daisee")) and row["etat_daisee"] in label2id:
            return row["etat_daisee"]
        return mapping_ser_daisee.get(row["emotion"], {}).get("etat")

    df["label"] = df.apply(_etat, axis=1)
    df = df[df["label"].isin(ETATS_DAISEE)]
    return _sample_df(df, "label", ETATS_DAISEE, max_samples, seed)


def charger_corpus_iemocap_er(max_samples: int, seed: int, min_duree: float):
    csv_path = os.path.join(ROOT, "data", "iemocap_parsed.csv")
    df = pd.read_csv(csv_path)
    df = df[df["emotion"].isin(EMOTIONS_IEMOCAP_ER)].copy()
    df = df[df["duree"] >= min_duree]
    if "wav_exists" in df.columns:
        df = df[df["wav_exists"]]
    df["label"] = df["emotion"].astype(str)
    return _sample_df(df, "label", EMOTIONS_IEMOCAP_ER, max_samples, seed)


def _sample_df(df, label_col, labels, max_samples, seed):
    parts = []
    per_class = max(20, max_samples // len(labels))
    for lab in labels:
        sub = df[df[label_col] == lab]
        if len(sub) == 0:
            continue
        take = min(per_class, len(sub))
        parts.append(sub.sample(n=take, random_state=seed))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=seed)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print("Distribution labels:", dict(Counter(sampled[label_col])))
    return sampled


def _augmenter_audio(y: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).copy()
    if rng.rand() < 0.7:
        y *= float(rng.uniform(0.75, 1.25))
    if rng.rand() < 0.5:
        y = y + rng.normal(0.0, 0.008, size=y.shape).astype(np.float32)
    if rng.rand() < 0.4 and len(y) > 1600:
        y = np.roll(y, int(rng.randint(-1600, 1601)))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:
        y = y / peak
    return y


class SerSegmentDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processor,
        label2id: dict,
        max_seconds: float = 4.0,
        augment: bool = False,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.label2id = label2id
        self.max_len = int(16000 * max_seconds)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, _sr = charger_segment_wav(
            row["utterance_id"],
            float(row["start_time"]),
            float(row["end_time"]),
            wav_path=row.get("wav_path"),
        )
        if y is None or len(y) < 1600:
            y = np.zeros(16000, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if self.augment:
            y = _augmenter_audio(y, self.rng)
        if len(y) > self.max_len:
            start = (len(y) - self.max_len) // 2
            y = y[start : start + self.max_len]

        inputs = self.processor(
            y,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
        )
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(self.label2id[row["label"]], dtype=torch.long)
        return item


def collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def freeze_backbone(model, unfreeze_last: bool = False, unfreeze_all: bool = False):
    if unfreeze_all:
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if hasattr(model, "projector"):
        for p in model.projector.parameters():
            p.requires_grad = True
    backbone = _backbone_module(model)
    if unfreeze_last and backbone is not None:
        try:
            for p in backbone.encoder.layers[-1].parameters():
                p.requires_grad = True
        except Exception:
            pass


def evaluate(model, loader, device, id2label):
    model.eval()
    ys, ps = [], []
    loss_sum, n = 0.0, 0
    labels = list(id2label.values())
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss_sum += float(out.loss) * len(batch["labels"])
            n += len(batch["labels"])
            pred = out.logits.argmax(dim=-1)
            ys.extend(batch["labels"].cpu().tolist())
            ps.extend(pred.cpu().tolist())
    y_true = [id2label[i] for i in ys]
    y_pred = [id2label[i] for i in ps]
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    return {
        "loss": round(loss_sum / max(n, 1), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            4,
        ),
        "report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SER WavLM → DAiSEE ou ancre IEMOCAP ER")
    parser.add_argument(
        "--target",
        choices=("daisee", "iemocap_er"),
        default="daisee",
        help="daisee=4 états DAiSEE ; iemocap_er=neu/hap/ang/sad pour hybride",
    )
    parser.add_argument("--max-samples", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duree", type=float, default=1.0)
    parser.add_argument("--unfreeze-last", action="store_true")
    parser.add_argument("--unfreeze-all", action="store_true")
    parser.add_argument("--no-class-weighted", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--out-metrics", default=None)
    args = parser.parse_args()

    if args.target == "iemocap_er":
        labels = EMOTIONS_IEMOCAP_ER
        charger = charger_corpus_iemocap_er
    else:
        labels = ETATS_DAISEE
        charger = charger_corpus_daisee

    label2id = {e: i for i, e in enumerate(labels)}
    id2label = {i: e for e, i in label2id.items()}
    out_dir = args.out_dir or OUT_DIRS[args.target]
    if args.out_metrics:
        out_metrics = args.out_metrics
    elif out_dir.rstrip("/\\").endswith("_full"):
        out_metrics = os.path.join(
            RESULTS_DIR,
            os.path.basename(out_dir.rstrip("/\\")) + "_metrics.json",
        )
    else:
        out_metrics = METRICS_FILES[args.target]

    if args.lr is None:
        args.lr = 1e-5 if args.unfreeze_all else 2e-4

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    print(f"Chargement corpus IEMOCAP ({args.target})…", flush=True)
    df = charger(args.max_samples, args.seed, args.min_duree)
    if len(df) < 40:
        raise SystemExit("Corpus trop petit — vérifiez data/IEMOCAP_full_release et iemocap_parsed.csv")

    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=df["label"],
    )

    print(f"Train={len(train_df)} Val={len(val_df)} | base={BASE_MODEL}")
    processor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    freeze_backbone(model, args.unfreeze_last, args.unfreeze_all)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    mode_ft = "FULL" if args.unfreeze_all else ("last_layer" if args.unfreeze_last else "head_only")
    print(f"Device: {device} | params entraînables: {n_tr:,}/{n_all:,} ({mode_ft}) | lr={args.lr}")

    train_loader = DataLoader(
        SerSegmentDataset(train_df, processor, label2id, augment=args.augment, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        SerSegmentDataset(val_df, processor, label2id, augment=False, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1
    )

    class_weights = None
    if not args.no_class_weighted:
        counts = Counter(train_df["label"].tolist())
        weights = [len(train_df) / (len(labels) * max(counts.get(l, 1), 1)) for l in labels]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"Class weights: {dict(zip(labels, [round(w, 3) for w in weights]))}")

    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"--- Epoch {epoch}/{args.epochs} (train batches={len(train_loader)}) ---", flush=True)
        model.train()
        running, n = 0.0, 0
        for bi, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(**batch)
            loss = torch.nn.functional.cross_entropy(
                out.logits,
                batch["labels"],
                weight=class_weights,
                label_smoothing=max(0.0, float(args.label_smoothing)),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += float(loss.detach()) * len(batch["labels"])
            n += len(batch["labels"])
        scheduler.step()
        train_loss = running / max(n, 1)
        val_metrics = evaluate(model, val_loader, device, id2label)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), **val_metrics})
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_f1={val_metrics['f1_macro']:.3f}",
            flush=True,
        )
        if val_metrics["f1_macro"] >= best_f1:
            best_f1 = val_metrics["f1_macro"]
            model.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
            with open(os.path.join(out_dir, "emolearn_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "base_model": BASE_MODEL,
                        "target": args.target,
                        "labels": labels,
                        "label2id": label2id,
                        "inclut_confusion": args.target == "daisee",
                        "unfreeze_all": args.unfreeze_all,
                        "unfreeze_last": args.unfreeze_last,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    mode_name = f"ser_wavlm_{args.target}_finetune"
    if args.unfreeze_all:
        mode_name += "_full"
    elif args.unfreeze_last:
        mode_name += "_last"

    report = {
        "mode": mode_name,
        "target": args.target,
        "unfreeze_all": args.unfreeze_all,
        "lr": args.lr,
        "base_model": BASE_MODEL,
        "out_dir": out_dir,
        "max_samples": args.max_samples,
        "epochs": args.epochs,
        "unfreeze_last": args.unfreeze_last,
        "class_weighted": not args.no_class_weighted,
        "augment": args.augment,
        "label_smoothing": args.label_smoothing,
        "device": str(device),
        "distribution": dict(Counter(df["label"].tolist())),
        "history": history,
        "best_val_f1_macro": best_f1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({"best_val_f1_macro": best_f1, "out_dir": out_dir, "metrics": out_metrics}, indent=2))


if __name__ == "__main__":
    main()
