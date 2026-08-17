#!/usr/bin/env python3
# ============================================
# Fine-tune ViT FER → 4 états DAiSEE
# (tête neuve + dernières couches backbone)
# ============================================
"""
Usage (CPU, recommandé) :
  python scripts/finetune_fer_vit_daisee.py --sample 800 --epochs 3 --unfreeze-layers 2

Usage plus complet :
  python scripts/finetune_fer_vit_daisee.py --sample 1200 --epochs 5 --unfreeze-layers 4 --batch-size 8

Fine-tune COMPLET (tout le backbone) :
  python scripts/finetune_fer_vit_daisee.py --from-faces-cache --epochs 5 --unfreeze-all --batch-size 4

Produit :
  models/fer_vit_daisee/
  results/fer_finetune_metrics.json
  results/fer_faces_cache/  (visages JPEG pour rejouer sans ré-extraire)

Par défaut : échantillonnage stratifié + class weights.
Le backbone ViT (dima806) est partiellement dégelé (--unfreeze-layers).
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from daisee_utils import charger_labels_daisee, etat_depuis_ligne, normaliser_clip_id
from daisee_video import extract_face_frames, resolve_daisee_video_path

RESULTS_DIR = os.path.join(ROOT, "results")
FACES_CACHE = os.path.join(RESULTS_DIR, "fer_faces_cache")
OUT_DIR_DEFAULT = os.path.join(ROOT, "models", "fer_vit_daisee")
BASE_MODEL = "dima806/facial_emotions_image_detection"
ETATS = ["Boredom", "Confusion", "Engagement", "Frustration"]
LABEL2ID = {e: i for i, e in enumerate(ETATS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}


def echantillonner_stratifie(df: pd.DataFrame, sample: int, seed: int) -> pd.DataFrame:
    etats = []
    for _, row in df.iterrows():
        etat, _, _ = etat_depuis_ligne(row)
        etats.append(etat)
    labeled = df.copy()
    labeled["etat_gt"] = etats
    per_class = max(1, sample // len(ETATS))
    rng = np.random.RandomState(seed)
    parts = []
    for etat in ETATS:
        pool = labeled[labeled["etat_gt"] == etat]
        n = min(per_class, len(pool))
        if n == 0:
            continue
        idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
        parts.append(pool.loc[idx])
        print(f"  stratifié {etat}: {n}/{len(pool)}")
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def preparer_faces(df: pd.DataFrame, cache_dir: str, force: bool = False) -> pd.DataFrame:
    """Extrait 1 visage / clip → JPEG + manifeste CSV."""
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "manifest.csv")
    if os.path.isfile(manifest_path) and not force:
        man = pd.read_csv(manifest_path)
        if len(man) >= max(20, int(0.5 * len(df))):
            print(f"Cache visages réutilisé: {manifest_path} ({len(man)} images)")
            return man

    rows = []
    skipped = 0
    for i, row in df.iterrows():
        clip_id = normaliser_clip_id(row["ClipID"])
        etat = row["etat_gt"]
        video = resolve_daisee_video_path(clip_id)
        if not video:
            skipped += 1
            continue
        images, _meta = extract_face_frames(clip_id, n_frames=1, avi_path=video)
        if not images:
            skipped += 1
            continue
        out_path = os.path.join(cache_dir, f"{clip_id}.jpg")
        images[0].convert("RGB").save(out_path, quality=90)
        rows.append({"clip_id": clip_id, "etat_gt": etat, "image_path": out_path})
        if (len(rows) % 25) == 0:
            print(f"  faces: {len(rows)} ({skipped} skip)…")

    man = pd.DataFrame(rows)
    man.to_csv(manifest_path, index=False)
    print(f"Faces OK: {len(man)} | skip: {skipped} | dist={dict(Counter(man['etat_gt']))}")
    return man


class FaceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor):
        from PIL import Image

        self.df = df.reset_index(drop=True)
        self.processor = processor
        self._Image = Image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._Image.open(row["image_path"]).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label = LABEL2ID[row["etat_gt"]]
        return {"pixel_values": pixel_values, "labels": label}


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


def freeze_backbone(model, unfreeze_layers: int, unfreeze_all: bool = False):
    """
    - unfreeze_all=True : tout le ViT + classifier entraînable
    - unfreeze_layers=N  : classifier + N dernières couches (+ layernorm)
    - unfreeze_layers=0  : classifier seul
    """
    if unfreeze_all:
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if unfreeze_layers <= 0:
        return
    layers = model.vit.encoder.layer
    n = len(layers)
    for i in range(max(0, n - unfreeze_layers), n):
        for p in layers[i].parameters():
            p.requires_grad = True
    for p in model.vit.layernorm.parameters():
        p.requires_grad = True


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            out = model(pixel_values=pixel_values, labels=labels)
            total_loss += float(out.loss.item()) * len(labels)
            n += len(labels)
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.cpu().numpy().tolist())
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    labels_idx = list(range(len(ETATS)))
    return {
        "loss": round(total_loss / max(n, 1), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(
            float(f1_score(y_true, y_pred, labels=labels_idx, average="macro", zero_division=0)),
            4,
        ),
        "report": classification_report(
            y_true,
            y_pred,
            labels=labels_idx,
            target_names=ETATS,
            zero_division=0,
            output_dict=True,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT FER → DAiSEE")
    parser.add_argument("--sample", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (défaut 2e-5 partiel, 1e-5 si --unfreeze-all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=2,
        help="Nb de dernières couches ViT à entraîner (0 = tête seule ; ignoré si --unfreeze-all)",
    )
    parser.add_argument(
        "--unfreeze-all",
        action="store_true",
        help="Fine-tune COMPLET du ViT (backbone entier + tête). LR plus bas recommandé.",
    )
    parser.add_argument("--labels-split", default="all", choices=["train", "validation", "test", "all"])
    parser.add_argument("--faces-cache", default=FACES_CACHE)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--from-faces-cache", action="store_true", help="Ignore --sample, utilise le manifest")
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    parser.add_argument("--out-metrics", default=os.path.join(RESULTS_DIR, "fer_finetune_metrics.json"))
    args = parser.parse_args()
    if args.lr is None:
        args.lr = 1e-5 if args.unfreeze_all else 2e-5

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | unfreeze_all={args.unfreeze_all} | lr={args.lr}")

    if args.from_faces_cache and os.path.isfile(os.path.join(args.faces_cache, "manifest.csv")):
        man = pd.read_csv(os.path.join(args.faces_cache, "manifest.csv"))
        print(f"Manifest chargé: {len(man)} | dist={dict(Counter(man['etat_gt']))}")
    else:
        print(f"Échantillonnage stratifié (sample={args.sample})…")
        df = charger_labels_daisee(split=args.labels_split)
        df = echantillonner_stratifie(df, sample=args.sample, seed=args.seed)
        print("Extraction visages…")
        man = preparer_faces(df, args.faces_cache, force=args.force_extract)

    if len(man) < 40:
        raise SystemExit(f"Trop peu d'images ({len(man)}). Vérifiez DAiSEE.")

    from sklearn.model_selection import train_test_split
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    train_df, test_df = train_test_split(
        man,
        test_size=0.25,
        random_state=args.seed,
        stratify=man["etat_gt"],
    )
    print(f"Split: train={len(train_df)} test={len(test_df)}")

    from transformers import AutoConfig

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    config = AutoConfig.from_pretrained(BASE_MODEL)
    config.num_labels = len(ETATS)
    config.id2label = {str(i): e for i, e in ID2LABEL.items()}
    config.label2id = LABEL2ID
    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL,
        config=config,
        ignore_mismatched_sizes=True,
    )
    freeze_backbone(model, args.unfreeze_layers, unfreeze_all=args.unfreeze_all)
    model.to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    mode_ft = "FULL" if args.unfreeze_all else f"partial(layers={args.unfreeze_layers})"
    print(f"Params entraînables: {n_train:,} / {n_all:,} ({mode_ft})")

    train_ds = FaceDataset(train_df, processor)
    test_ds = FaceDataset(test_df, processor)

    # Sampler pondéré (rééquilibrage fort)
    counts = Counter(train_df["etat_gt"].tolist())
    weights = [1.0 / counts[e] for e in train_df["etat_gt"]]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # class weights dans la loss
    class_w = torch.tensor(
        [1.0 / counts.get(e, 1) for e in ETATS], dtype=torch.float32, device=device
    )
    class_w = class_w / class_w.sum() * len(ETATS)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    best_f1 = -1.0
    history = []
    # Toujours écrire dans un dossier dédié (évite verrou si l'API charge out_dir)
    ckpt_dir = args.out_dir.rstrip("/\\") + "_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            out = model(pixel_values=pixel_values)
            loss = nn.functional.cross_entropy(out.logits, labels, weight=class_w)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(labels)
            n_seen += len(labels)
        train_loss = running / max(n_seen, 1)
        metrics = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), **{
            k: metrics[k] for k in ("loss", "accuracy", "f1_macro")
        }})
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_acc={metrics['accuracy']:.3f} "
            f"val_f1={metrics['f1_macro']:.3f}",
            flush=True,
        )
        if metrics["f1_macro"] >= best_f1:
            best_f1 = metrics["f1_macro"]
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            print(f"  ✓ checkpoint → {ckpt_dir}", flush=True)

    # Recharge best + métriques finales
    model = AutoModelForImageClassification.from_pretrained(ckpt_dir)
    model.to(device)
    final = evaluate(model, test_loader, device)

    # Promotion vers out_dir si possible (API arrêtée) ; sinon garder _ckpt
    promoted = ckpt_dir
    try:
        import shutil

        if os.path.isdir(args.out_dir):
            shutil.rmtree(args.out_dir)
        shutil.copytree(ckpt_dir, args.out_dir)
        promoted = args.out_dir
        print(f"Promu → {args.out_dir}", flush=True)
    except OSError as e:
        print(f"Promotion out_dir impossible ({e}) — utiliser {ckpt_dir}", flush=True)

    report = {
        "mode": "fer_vit_finetune_full" if args.unfreeze_all else "fer_vit_finetune",
        "base_model": BASE_MODEL,
        "unfreeze_all": args.unfreeze_all,
        "unfreeze_layers": args.unfreeze_layers,
        "lr": args.lr,
        "epochs": args.epochs,
        "sample": args.sample,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "distribution_train": dict(Counter(train_df["etat_gt"].tolist())),
        "distribution_test": dict(Counter(test_df["etat_gt"].tolist())),
        "trainable_params": int(n_train),
        "total_params": int(n_all),
        "history": history,
        "best_val_f1_macro": best_f1,
        "final_test": final,
        "model_path": promoted,
        "checkpoint_dir": ckpt_dir,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Fine-tune ViT : nouvelle tête 4 classes DAiSEE + "
            f"{'FULL backbone' if args.unfreeze_all else str(args.unfreeze_layers) + ' couches'}. "
            "WeightedRandomSampler + class weights."
        ),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    summary = {
        "val_acc": final["accuracy"],
        "val_f1_macro": final["f1_macro"],
        "f1_par_classe": {
            e: round(float(final["report"][e]["f1-score"]), 3) for e in ETATS
        },
        "recall_par_classe": {
            e: round(float(final["report"][e]["recall"]), 3) for e in ETATS
        },
        "model": promoted,
        "metrics": args.out_metrics,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
