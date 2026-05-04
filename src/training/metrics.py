import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from src.models.losses import noisy_or_prob


@torch.no_grad()
def predict_bad_probs(model, loader, device):
    model.eval()

    y_true = []
    p_bad_all = []

    for batch in loader:
        audio = batch["audio"].to(device)
        sounds_mask = batch["sounds_mask"].to(device)
        y_bad = batch["y_bad"].to(device)

        attn = (audio != 0).long()
        p_defect, _ = model(audio, attention_mask=attn)

        p_bad = noisy_or_prob(p_defect, sounds_mask)

        y_true.append(y_bad.detach().cpu().numpy())
        p_bad_all.append(p_bad.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    p_bad_all = np.concatenate(p_bad_all)

    return y_true, p_bad_all


def compute_metrics(y_true, p, thr):
    y_pred = (p >= thr).astype(int)

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, p))
        ap = float(average_precision_score(y_true, p))
    else:
        auc = None
        ap = None

    return {
        "thr": float(thr),
        "auc": auc,
        "ap": ap,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", pos_label=1)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
    }


def best_f1_threshold(y_true, p, thr_grid=None):
    if thr_grid is None:
        thr_grid = np.linspace(0.05, 0.95, 19)

    best_thr = 0.5
    best_f1 = -1.0

    for thr in thr_grid:
        y_pred = (p >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, best_f1