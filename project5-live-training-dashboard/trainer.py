"""
trainer.py - the training engine for Project 5 (City Hospital).

This file knows NOTHING about Streamlit. It trains a model and, after every
epoch, calls back whatever function you give it (on_epoch). The Streamlit page
passes a function that redraws the chart; a notebook can pass nothing at all.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Default settings = the "256 neurons, no fixes" run from the slides
DEFAULTS = dict(
    hidden=256,        # neurons in each of the 2 hidden layers
    dropout=0.0,       # 0.0 = dropout off
    weight_decay=0.0,  # 0.0 = weight decay off
    lr=0.001,          # learning rate for AdamW
    epochs=300,        # maximum number of epochs
    batch_size=341,   # 341 = all training rows = one update per epoch (full batch)
    patience=0,        # 0 = early stopping off
    scale=True,        # StandardScaler on/off
    seed=0,
)


# ---------------------------------------------------------------- data
def load_data(scale=True, seed=0):
    """569 patients, 30 measurements. Split 60/20/20 -> 341/114/114, stratified."""
    ds = load_breast_cancer()
    X = ds.data.astype(np.float32)
    # sklearn uses 0 = malignant, 1 = benign. The project wants malignant = 1,
    # so we flip the labels. Forgetting this makes "cancers found" mean "healthy found".
    y = (1 - ds.target).astype(np.int64)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    raw_va = X_va.copy()  # unscaled copy, so the page can show real measurements
    if scale:
        # fit on TRAINING rows only, then apply the same numbers everywhere
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_va, X_te = (scaler.transform(a).astype(np.float32) for a in (X_tr, X_va, X_te))

    t = torch.tensor
    return dict(X_tr=t(X_tr), y_tr=t(y_tr), X_va=t(X_va), y_va=t(y_va),
                X_te=t(X_te), y_te=t(y_te), raw_va=raw_va,
                feature_names=list(ds.feature_names))


# ---------------------------------------------------------------- model
def build_model(hidden, dropout, n_in=30):
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 2),          # 2 outputs: benign (0), malignant (1)
    )


# ---------------------------------------------------------------- training
def train(cfg, data, on_epoch=None):
    """Train with the settings in cfg. Returns (model, hist, info)."""
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(cfg["seed"])

    model = build_model(cfg["hidden"], cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(data["X_tr"], data["y_tr"]),
                        batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg["seed"]))

    hist = {"train": [], "val": [], "train_acc": [], "val_acc": []}
    best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()                                   # dropout ON
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()                                    # dropout OFF for measuring
        with torch.no_grad():
            out_tr, out_va = model(data["X_tr"]), model(data["X_va"])
            hist["train"].append(loss_fn(out_tr, data["y_tr"]).item())
            hist["val"].append(loss_fn(out_va, data["y_va"]).item())
            hist["train_acc"].append((out_tr.argmax(1) == data["y_tr"]).float().mean().item())
            hist["val_acc"].append((out_va.argmax(1) == data["y_va"]).float().mean().item())

        if hist["val"][-1] < best_val:                  # remember the best weights
            best_val, best_epoch, wait = hist["val"][-1], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        if on_epoch:                                    # tell the page (or anyone) about it
            on_epoch(epoch, hist)

        if cfg["patience"] and wait >= cfg["patience"]:  # early stopping
            break

    # With early stopping we ship the best checkpoint, not the last epoch.
    # (best_state stays None only if validation loss was never a real number.)
    if cfg["patience"] and best_state is not None:
        model.load_state_dict(best_state)

    info = dict(epochs_run=epoch, best_val=best_val, best_epoch=best_epoch)
    return model, hist, info


# ---------------------------------------------------------------- diagnosis
def diagnose(hist):
    """Rule from Part 2, plus one extra check for jumpy curves."""
    tr, va = hist["train"], hist["val"]
    # A loss of NaN/inf means training blew up. Every comparison below is False for
    # NaN, so without this line a blown-up run would be called "healthy".
    if not np.all(np.isfinite(tr + va)):
        return "unstable"
    # Extra check: with one update per epoch, a healthy TRAINING loss only goes down.
    # If it climbs back more than 0.5 above the lowest value it already reached,
    # the steps are too big: learning rate too high, or features not scaled.
    lowest_so_far = np.minimum.accumulate(tr)
    if max(np.array(tr) - lowest_so_far) > 0.5:
        return "unstable"
    if tr[-1] > 0.3:
        return "underfitting"
    if va[-1] > 1.3 * min(va):
        return "overfitting"
    return "healthy"


# ---------------------------------------------------------------- evaluation
def predict_proba(model, X):
    """Probability of malignant (class 1) for each row."""
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(X), dim=1)[:, 1].numpy()


def evaluate(model, X, y, threshold=0.5):
    """Counts the doctors care about. A patient is flagged malignant if prob >= threshold."""
    probs = predict_proba(model, X)
    y = y.numpy()
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=float((pred == y).mean()),
                cancers_found=int(tp), cancers_total=int(tp + fn), missed=int(fn),
                false_alarms=int(fp), probs=probs, pred=pred, y=y,
                cm=np.array([[tn, fp], [fn, tp]]))
