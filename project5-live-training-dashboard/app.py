"""
app.py - City Hospital live training dashboard (Project 5).
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score

import trainer

st.set_page_config(page_title="City Hospital · training dashboard", layout="wide")
COLORS = ["#E8833A", "#3A7DE8"]          # orange = training, blue = validation
VERDICT = {
    "overfitting":  ("error",   "Overfitting: training loss keeps falling while validation loss climbs back up. Try early stopping, a smaller model, or more regularisation."),
    "underfitting": ("warning", "Underfitting: training loss is still above 0.3. The model is too weak or trained too little."),
    "unstable":     ("warning", "Unstable: the loss jumps back up. The steps are too big. Lower the learning rate or switch scaling on."),
    "healthy":      ("success", "Healthy: both losses fell together and validation stayed close to its best."),
}

# session_state survives between button clicks (Streamlit re-runs the whole script on every click)
if "runs" not in st.session_state:
    st.session_state.runs = []       # the experiments table
if "final" not in st.session_state:
    st.session_state.final = None    # the one-time final test result


@st.cache_resource
def get_data(scale):
    # cached: the data is loaded once per scaling choice, not on every click
    return trainer.load_data(scale=scale)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    hidden = st.select_slider("Neurons per hidden layer", [4, 8, 16, 32, 64, 128, 256], value=256)
    dropout = st.slider("Dropout", 0.0, 0.6, 0.0, 0.05)
    weight_decay = st.select_slider("Weight decay", [0.0, 0.001, 0.01, 0.05, 0.1, 0.5], value=0.0)
    lr = st.select_slider("Learning rate", [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], value=0.001)
    epochs = st.slider("Max epochs", 50, 500, 300, 50)
    use_es = st.checkbox("Early stopping", value=False)
    patience = st.slider("Patience (epochs)", 5, 50, 20, disabled=not use_es)
    scale = st.checkbox("Scale features (StandardScaler)", value=True)
    st.divider()
    threshold = st.slider("Decision threshold: call it malignant if probability ≥", 0.05, 0.95, 0.50, 0.05,
                          help="Doesn't need retraining. Lower = find more cancers, but more false alarms.")
    go = st.button("Train", type="primary", width="stretch")

cfg = dict(hidden=hidden, dropout=dropout, weight_decay=weight_decay, lr=lr,
           epochs=epochs, patience=patience if use_es else 0, scale=scale)

st.title("City Hospital · live training dashboard")
st.caption("Wisconsin breast cancer data: 569 patients, 30 measurements each. "
           "Split 341 train / 114 validation / 114 test. 1 = malignant.")

chart_slot = st.empty()              # the live chart is redrawn in place here


def loss_frame(hist):
    return pd.DataFrame({"training loss": hist["train"], "validation loss": hist["val"]},
                        index=pd.RangeIndex(1, len(hist["train"]) + 1, name="epoch"))


# ---------------------------------------------------------------- training
if go:
    data = get_data(scale)
    bar = st.progress(0.0, text="Training…")

    def redraw(epoch, hist):
        bar.progress(epoch / cfg["epochs"], text=f"Epoch {epoch} / {cfg['epochs']}")
        if epoch % 5 == 0 or epoch == 1:
            chart_slot.line_chart(loss_frame(hist), color=COLORS, height=380)

    model, hist, info = trainer.train(cfg, data, on_epoch=redraw)
    bar.empty()
    verdict = trainer.diagnose(hist)
    ev = trainer.evaluate(model, data["X_va"], data["y_va"], threshold)

    st.session_state.last = dict(model=model, hist=hist, info=info, cfg=cfg, verdict=verdict)
    st.session_state.runs.append({
        "run": len(st.session_state.runs) + 1,
        "neurons": hidden, "dropout": dropout, "weight decay": weight_decay, "lr": lr,
        "early stopping": str(cfg["patience"]) if cfg["patience"] else "off", "scaled": "yes" if scale else "no",
        "diagnosis": verdict, "epochs run": info["epochs_run"],
        "best val loss": round(info["best_val"], 3), "best epoch": info["best_epoch"],
        "threshold": threshold,
        "cancers found": f"{ev['cancers_found']} of {ev['cancers_total']}",
        "false alarms": ev["false_alarms"],
    })


# ---------------------------------------------------------------- results
if "last" not in st.session_state:
    st.info("Pick settings in the sidebar and press **Train**. Tip: start with the defaults (256 neurons, no fixes).")
    st.stop()

last = st.session_state.last
model, hist, info, verdict = last["model"], last["hist"], last["info"], last["verdict"]
data = get_data(last["cfg"]["scale"])
ev = trainer.evaluate(model, data["X_va"], data["y_va"], threshold)   # uses the CURRENT threshold

chart_slot.line_chart(loss_frame(hist), color=COLORS, height=380)
kind, text = VERDICT[verdict]
getattr(st, kind)(f"**Diagnosis: {verdict}.** {text}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Epochs run", info["epochs_run"])
c2.metric("Best validation loss", f"{info['best_val']:.3f}", f"at epoch {info['best_epoch']}", delta_color="off")
c3.metric("Cancers found (validation)", f"{ev['cancers_found']} of {ev['cancers_total']}",
          f"{ev['missed']} missed", delta_color="inverse" if ev["missed"] else "off")
c4.metric("False alarms", ev["false_alarms"])

# ---------------------------------------------------------------- extra graphs
st.subheader("More graphs for this run")
tabs = st.tabs(["Accuracy curves", "Generalisation gap", "Confusion matrix",
                "Probability spread", "ROC curve", "Threshold trade-off"])

with tabs[0]:
    st.caption("Accuracy (share of patients classified correctly) per epoch.")
    acc = pd.DataFrame({"training accuracy": hist["train_acc"], "validation accuracy": hist["val_acc"]},
                       index=pd.RangeIndex(1, len(hist["train_acc"]) + 1, name="epoch"))
    st.line_chart(acc, color=COLORS, height=320)

with tabs[1]:
    st.caption("Validation loss minus training loss. A gap that keeps growing means the model is memorising.")
    gap = pd.DataFrame({"gap (val − train)": np.array(hist["val"]) - np.array(hist["train"])},
                       index=pd.RangeIndex(1, len(hist["val"]) + 1, name="epoch"))
    st.area_chart(gap, color="#D62728", height=320)

with tabs[2]:
    st.caption("Validation set, current threshold. Bottom-left = cancers missed: the most dangerous cell.")
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(ev["cm"], cmap="Blues")
    labels = [["healthy, said healthy", "healthy, said CANCER\n(false alarm)"],
              ["cancer, said healthy\n(MISSED)", "cancer, said cancer"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{ev['cm'][i, j]}\n{labels[i][j]}", ha="center", va="center", fontsize=8,
                    color="white" if ev["cm"][i, j] > ev["cm"].max() / 2 else "black")
    ax.set_xticks([0, 1], ["predicted benign", "predicted malignant"])
    ax.set_yticks([0, 1], ["truly benign", "truly malignant"])
    plt.tight_layout(); st.pyplot(fig, width="content"); plt.close(fig)

with tabs[3]:
    st.caption("How sure the model is. Good models push benign patients to the left and malignant ones to the right. "
               "The dashed line is your threshold.")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.linspace(0, 1, 21)
    ax.hist(ev["probs"][ev["y"] == 0], bins=bins, alpha=0.7, label="truly benign", color="#2CA02C")
    ax.hist(ev["probs"][ev["y"] == 1], bins=bins, alpha=0.7, label="truly malignant", color="#D62728")
    ax.axvline(threshold, color="black", ls="--", label=f"threshold {threshold:.2f}")
    ax.set_xlabel("predicted probability of malignant"); ax.set_ylabel("patients"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

with tabs[4]:
    fpr, tpr, _ = roc_curve(ev["y"], ev["probs"])
    auc = roc_auc_score(ev["y"], ev["probs"])
    st.caption(f"Every possible threshold at once. Area under the curve (AUC) = **{auc:.3f}** "
               "(1.0 = perfect, 0.5 = coin flip).")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color=COLORS[1], lw=2, label=f"model (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls=":", label="coin flip")
    ax.set_xlabel("false alarm rate"); ax.set_ylabel("cancers found rate (recall)"); ax.legend()
    plt.tight_layout(); st.pyplot(fig, width="content"); plt.close(fig)

with tabs[5]:
    st.caption("What changing the threshold would do on the validation set. "
               "A missed cancer costs far more than a false alarm, so doctors usually lower it.")
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        e = trainer.evaluate(model, data["X_va"], data["y_va"], t)
        rows.append({"threshold": t, "cancers missed": e["missed"], "false alarms": e["false_alarms"]})
    tdf = pd.DataFrame(rows).set_index("threshold")
    st.line_chart(tdf, color=["#D62728", "#7F7F7F"], height=300)

# ---------------------------------------------------------------- experiments table
st.subheader("Experiments table")
runs_df = pd.DataFrame(st.session_state.runs)
st.dataframe(runs_df, hide_index=True, width="stretch")
col_a, col_b = st.columns([1, 4])
col_a.download_button("Download CSV", runs_df.to_csv(index=False), "experiments.csv", "text/csv")
if col_b.button("Clear table"):
    st.session_state.runs = []
    st.rerun()

# ---------------------------------------------------------------- check a patient
st.subheader("Check a patient")
st.caption("Patients from the validation set: the model never trained on them.")
idx = st.number_input("Validation patient number (0 to 113)", 0, len(data["y_va"]) - 1, 0)
prob = float(trainer.predict_proba(model, data["X_va"][idx:idx + 1])[0])
truth = "malignant" if int(data["y_va"][idx]) == 1 else "benign"
said = "malignant" if prob >= threshold else "benign"
p1, p2, p3 = st.columns(3)
p1.metric("Probability of malignant", f"{prob:.1%}")
p2.metric("Model says", said)
p3.metric("Real diagnosis", truth, "correct" if said == truth else "WRONG",
          delta_color="normal" if said == truth else "inverse")
with st.expander("Show this patient's 30 measurements"):
    st.dataframe(pd.DataFrame({"measurement": data["feature_names"], "value": data["raw_va"][idx]}),
                 hide_index=True, height=300)

# ---------------------------------------------------------------- final test, once
st.subheader("Final test (one time only)")
st.caption("Tick this ONCE, after you've chosen your settings. The test set is never used to choose anything.")
done = st.session_state.final is not None
if st.checkbox("Run the final test on the current model", value=done, disabled=done) and not done:
    te = trainer.evaluate(model, data["X_te"], data["y_te"], threshold)
    st.session_state.final = dict(settings=dict(last["cfg"], threshold=threshold), accuracy=te["accuracy"],
                                  found=te["cancers_found"], total=te["cancers_total"],
                                  false_alarms=te["false_alarms"])
    st.rerun()
if st.session_state.final:
    f = st.session_state.final
    st.success(f"**Final test:** accuracy {f['accuracy']:.3f} · {f['found']} of {f['total']} cancers found · "
               f"{f['false_alarms']} false alarms")
    st.json(f["settings"], expanded=False)
