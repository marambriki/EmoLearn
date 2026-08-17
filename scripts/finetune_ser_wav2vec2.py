#!/usr/bin/env python3
# ============================================
# Fine-tune tête SER wav2vec2 → 4 états DAiSEE
# (inclut surprise/fear → Confusion)
# ============================================
"""
Usage (CPU, défaut recommandé) :
  python scripts/finetune_ser_wav2vec2.py --max-samples 800 --epochs 5 --augment

Usage plus complet :
  python scripts/finetune_ser_wav2vec2.py --max-samples 2000 --epochs 8 --unfreeze-last --augment

Produit :
  models/ser_wav2vec2_daisee/
  results/ser_finetune_metrics.json

Le backbone reste largement gelé ; seule la tête (et optionnellement les
dernières couches) est entraînée. Confusion devient une vraie classe cible.

Runtime produit : SER hybride (FT DAiSEE + SUPERB ER) dans backend/ser_module.py.
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
from ser_module import mapping_ser_daisee

RESULTS_DIR = os.path.join(ROOT, "results")
OUT_DIR_DEFAULT = os.path.join(ROOT, "models", "ser_wav2vec2_daisee")
BASE_MODEL = "superb/wav2vec2-base-superb-er"
ETATS = ["Boredom", "Confusion", "Engagement", "Frustration"]
LABEL2ID = {e: i for i, e in enumerate(ETATS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}


def charger_corpus(max_samples: int, seed: int, min_duree: float):
    csv_path = os.path.join(ROOT, "data", "iemocap_parsed.csv")
    df = pd.read_csv(csv_path)
    # Exclure labels non informatifs
    df = df[~df["emotion"].isin(["xxx", "oth"])].copy()
    df = df[df["duree"] >= min_duree]
    if "wav_exists" in df.columns:
        df = df[df["wav_exists"]]

    # Label cible = état DAiSEE (sur/fea → Confusion via mapping)
    def _etat(row):
        if pd.notna(row.get("etat_daisee")) and row["etat_daisee"] in LABEL2ID:
            return row["etat_daisee"]
        return mapping_ser_daisee.get(row["emotion"], {}).get("etat")

    df["label_daisee"] = df.apply(_etat, axis=1)
    df = df[df["label_daisee"].isin(ETATS)]

    # Oversample léger des classes rares (Confusion) dans le tirage
    rng = np.random.RandomState(seed)
    parts = []
    per_class = max(20, max_samples // 4)
    for etat in ETATS:
        sub = df[df["label_daisee"] == etat]
        if len(sub) == 0:
            continue
        take = min(per_class, len(sub))
        parts.append(sub.sample(n=take, random_state=seed))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=seed)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print("Distribution labels:", dict(Counter(sampled["label_daisee"])))
    print("dont emotions Confusion:", dict(Counter(
        sampled.loc[sampled["label_daisee"] == "Confusion", "emotion"]
    )))
    return sampled


def _augmenter_audio(y: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Augmentations légères (gain, bruit, décalage temporel) — train only."""
    y = np.asarray(y, dtype=np.float32).copy()
    if rng.rand() < 0.7:
        gain = float(rng.uniform(0.75, 1.25))
        y *= gain
    if rng.rand() < 0.5:
        noise = rng.normal(0.0, 0.008, size=y.shape).astype(np.float32)
        y = y + noise
    if rng.rand() < 0.4 and len(y) > 1600:
        shift = int(rng.randint(-1600, 1601))
        y = np.roll(y, shift)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:
        y = y / peak
    return y


class SerSegmentDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processor,
        max_seconds: float = 4.0,
        augment: bool = False,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_len = int(16000 * max_seconds)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, sr = charger_segment_wav(
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
        item["labels"] = torch.tensor(LABEL2ID[row["label_daisee"]], dtype=torch.long)
        return item


def collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    return out


def freeze_backbone(model, unfreeze_last: bool = False, unfreeze_all: bool = False):
    """
    - unfreeze_all : tout wav2vec2 + classifier
    - unfreeze_last : classifier (+ projector) + dernière couche encodeur
    - sinon : classifier (+ projector) seul
    """
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
    if unfreeze_last and hasattr(model, "wav2vec2"):
        try:
            for p in model.wav2vec2.encoder.layers[-1].parameters():
                p.requires_grad = True
        except Exception:
            pass


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss_sum += float(out.loss) * len(batch["labels"])
            n += len(batch["labels"])
            pred = out.logits.argmax(dim=-1)
            ys.extend(batch["labels"].cpu().tolist())
            ps.extend(pred.cpu().tolist())
    y_true = [ID2LABEL[i] for i in ys]
    y_pred = [ID2LABEL[i] for i in ps]
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    return {
        "loss": round(loss_sum / max(n, 1), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(
            float(f1_score(y_true, y_pred, labels=ETATS, average="macro", zero_division=0)),
            4,
        ),
        "report": classification_report(
            y_true, y_pred, labels=ETATS, zero_division=0, output_dict=True
        ),
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune tête SER wav2vec2 → DAiSEE")
    parser.add_argument("--max-samples", type=int, default=800)
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Défaut 5 (mieux que 2 pour la tête DAiSEE)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (défaut 2e-4 tête, 1e-5 si --unfreeze-all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duree", type=float, default=1.0)
    parser.add_argument("--unfreeze-last", action="store_true")
    parser.add_argument(
        "--unfreeze-all",
        action="store_true",
        help="Fine-tune COMPLET wav2vec2 (backbone entier + tête)",
    )
    parser.add_argument(
        "--no-class-weighted",
        action="store_true",
        help="Désactive la loss CrossEntropy pondérée par classe (activée par défaut)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augmentation audio train (gain / bruit / shift)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing CrossEntropy (0 pour désactiver)",
    )
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    parser.add_argument(
        "--out-metrics",
        default=os.path.join(RESULTS_DIR, "ser_finetune_metrics.json"),
    )
    args = parser.parse_args()
    if args.lr is None:
        args.lr = 1e-5 if args.unfreeze_all else 2e-4

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    print("Chargement corpus IEMOCAP…")
    df = charger_corpus(args.max_samples, args.seed, args.min_duree)
    if len(df) < 40:
        raise SystemExit("Corpus trop petit — vérifiez data/IEMOCAP_full_release et iemocap_parsed.csv")

    # Split 80/20 stratifié
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=df["label_daisee"],
    )

    print(f"Train={len(train_df)} Val={len(val_df)} | base={BASE_MODEL}")
    processor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(ETATS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )
    freeze_backbone(
        model,
        unfreeze_last=args.unfreeze_last,
        unfreeze_all=args.unfreeze_all,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    mode_ft = "FULL" if args.unfreeze_all else (
        "last_layer" if args.unfreeze_last else "head_only"
    )
    print(f"Device: {device} | params entraînables: {n_tr:,}/{n_all:,} ({mode_ft}) | lr={args.lr}")

    train_loader = DataLoader(
        SerSegmentDataset(
            train_df, processor, augment=args.augment, seed=args.seed
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        SerSegmentDataset(val_df, processor, augment=False, seed=args.seed),
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
        counts = Counter(train_df["label_daisee"].tolist())
        weights = []
        for etat in ETATS:
            n_c = max(counts.get(etat, 1), 1)
            weights.append(len(train_df) / (len(ETATS) * n_c))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"Class weights: {dict(zip(ETATS, [round(w, 3) for w in weights]))}")

    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
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
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), **val_metrics})
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_f1={val_metrics['f1_macro']:.3f}"
        )
        if val_metrics["f1_macro"] >= best_f1:
            best_f1 = val_metrics["f1_macro"]
            model.save_pretrained(args.out_dir)
            processor.save_pretrained(args.out_dir)
            with open(os.path.join(args.out_dir, "emolearn_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "base_model": BASE_MODEL,
                        "labels": ETATS,
                        "label2id": LABEL2ID,
                        "inclut_confusion": True,
                        "source_labels": "IEMOCAP emotion → etat_daisee (sur/fea → Confusion)",
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    report = {
        "mode": "ser_wav2vec2_finetune_full" if args.unfreeze_all else "ser_wav2vec2_finetune_head",
        "unfreeze_all": args.unfreeze_all,
        "lr": args.lr,
        "base_model": BASE_MODEL,
        "out_dir": args.out_dir,
        "max_samples": args.max_samples,
        "epochs": args.epochs,
        "unfreeze_last": args.unfreeze_last,
        "class_weighted": not args.no_class_weighted,
        "augment": args.augment,
        "label_smoothing": args.label_smoothing,
        "device": str(device),
        "distribution": dict(Counter(df["label_daisee"].tolist())),
        "history": history,
        "best_val_f1_macro": best_f1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Tête de classification fine-tunée vers 4 états DAiSEE. "
            "Confusion couverte via surprise/fear (et mapping IEMOCAP→DAiSEE). "
            "Loss pondérée + label smoothing + cosine LR ; "
            "runtime hybride SUPERB+FT dans ser_module."
        ),
    }
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "best_val_f1_macro": best_f1,
        "out_dir": args.out_dir,
        "metrics": args.out_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
