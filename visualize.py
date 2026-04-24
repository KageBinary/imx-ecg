"""ECG model training visualizer — Streamlit dashboard."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

CHECKPOINT_DIR = Path("checkpoints")
CLASS_NAMES = ["Normal", "AF", "Other", "Noisy"]

RUNS = {
    "deploy_tk25  (best, F1=0.704)": "deploy_tk25",
    "deploy_v2    (F1=0.607)":       "deploy_v2",
    "deploy_v3    (F1=0.605)":       "deploy_v3",
    "deploy_wide  (F1=0.604)":       "deploy_wide",
}

RUN_COLORS = {
    "deploy_tk25": "#00c8ff",
    "deploy_v2":   "#ff6b6b",
    "deploy_v3":   "#69db7c",
    "deploy_wide": "#ffd43b",
}


@st.cache_data
def load_metrics(run_key: str) -> list[dict]:
    path = CHECKPOINT_DIR / f"{run_key}.metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def epochs(records):
    return [r["epoch"] for r in records]


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Deploy Model — Training Dashboard",
    page_icon="💓",
    layout="wide",
)

st.title("💓 ECGDeployNet — Training Dashboard")
st.caption(
    "NXP i.MX 8M Plus NPU deployment target · PhysioNet 2017 · "
    "4-class arrhythmia classification (Normal / AF / Other / Noisy)"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Display options")
selected_labels = st.sidebar.multiselect(
    "Runs to show",
    options=list(RUNS.keys()),
    default=list(RUNS.keys()),
)
selected_runs = [RUNS[l] for l in selected_labels]

smooth_window = st.sidebar.slider("Smoothing window (epochs)", 1, 15, 1)


def smooth(vals, w):
    if w <= 1:
        return vals
    kernel = np.ones(w) / w
    return np.convolve(vals, kernel, mode="same").tolist()


# ── Load all selected runs ────────────────────────────────────────────────────
all_data = {run: load_metrics(run) for run in selected_runs}

# ── Section 1: Macro-F1 overview ─────────────────────────────────────────────
st.subheader("Macro-F1 over epochs")

fig_f1 = go.Figure()
for run, records in all_data.items():
    if not records:
        continue
    best_rec = max(records, key=lambda r: r["val_macro_f1"])
    fig_f1.add_trace(go.Scatter(
        x=epochs(records),
        y=smooth([r["val_macro_f1"] for r in records], smooth_window),
        mode="lines",
        name=run,
        line=dict(color=RUN_COLORS[run], width=2),
    ))
    fig_f1.add_trace(go.Scatter(
        x=[best_rec["epoch"]],
        y=[best_rec["val_macro_f1"]],
        mode="markers+text",
        marker=dict(color=RUN_COLORS[run], size=10, symbol="star"),
        text=[f"  {best_rec['val_macro_f1']:.3f} @ ep{best_rec['epoch']}"],
        textposition="middle right",
        showlegend=False,
        hovertemplate=f"{run}<br>Epoch %{{x}}<br>F1 %{{y:.3f}}<extra></extra>",
    ))

fig_f1.update_layout(
    xaxis_title="Epoch",
    yaxis_title="Val Macro-F1",
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="#fafafa",
    legend=dict(bgcolor="#0e1117"),
    height=380,
)
st.plotly_chart(fig_f1, use_container_width=True)

# ── Section 2: Loss curves ────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Train loss")
    fig_tl = go.Figure()
    for run, records in all_data.items():
        if not records:
            continue
        fig_tl.add_trace(go.Scatter(
            x=epochs(records),
            y=smooth([r["train_loss"] for r in records], smooth_window),
            mode="lines", name=run,
            line=dict(color=RUN_COLORS[run], width=2),
        ))
    fig_tl.update_layout(
        xaxis_title="Epoch", yaxis_title="Loss",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#fafafa", showlegend=False, height=300,
    )
    st.plotly_chart(fig_tl, use_container_width=True)

with col_b:
    st.subheader("Val loss")
    fig_vl = go.Figure()
    for run, records in all_data.items():
        if not records:
            continue
        fig_vl.add_trace(go.Scatter(
            x=epochs(records),
            y=smooth([r["val_loss"] for r in records], smooth_window),
            mode="lines", name=run,
            line=dict(color=RUN_COLORS[run], width=2),
        ))
    fig_vl.update_layout(
        xaxis_title="Epoch", yaxis_title="Loss",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#fafafa", showlegend=False, height=300,
    )
    st.plotly_chart(fig_vl, use_container_width=True)

# ── Section 3: Per-class recall (one run at a time) ───────────────────────────
st.subheader("Per-class recall over epochs")

recall_run_label = st.selectbox(
    "Run", options=[l for l in selected_labels], key="recall_run"
)
recall_run = RUNS[recall_run_label] if recall_run_label else None

if recall_run and all_data.get(recall_run):
    records = all_data[recall_run]
    fig_rc = go.Figure()
    class_colors = {"Normal": "#00c8ff", "AF": "#ff6b6b", "Other": "#69db7c", "Noisy": "#ffd43b"}
    for cls in CLASS_NAMES:
        vals = [r["val_recalls"][cls] for r in records]
        fig_rc.add_trace(go.Scatter(
            x=epochs(records),
            y=smooth(vals, smooth_window),
            mode="lines", name=cls,
            line=dict(color=class_colors[cls], width=2),
        ))
    fig_rc.update_layout(
        xaxis_title="Epoch", yaxis_title="Recall",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#fafafa", legend=dict(bgcolor="#0e1117"),
        height=320,
    )
    st.plotly_chart(fig_rc, use_container_width=True)

# ── Section 4: Confusion matrix at any epoch ─────────────────────────────────
st.subheader("Confusion matrix — pick a run and epoch")

cm_col1, cm_col2 = st.columns([1, 2])

with cm_col1:
    cm_run_label = st.selectbox(
        "Run", options=[l for l in selected_labels], key="cm_run"
    )
    cm_run = RUNS[cm_run_label] if cm_run_label else None
    cm_records = all_data.get(cm_run, [])

    if cm_records:
        ep_min = cm_records[0]["epoch"]
        ep_max = cm_records[-1]["epoch"]
        best_ep = max(cm_records, key=lambda r: r["val_macro_f1"])["epoch"]

        epoch_choice = st.slider(
            "Epoch", min_value=ep_min, max_value=ep_max, value=best_ep
        )

        # Find closest epoch record
        rec = min(cm_records, key=lambda r: abs(r["epoch"] - epoch_choice))

        st.metric("Val Macro-F1", f"{rec['val_macro_f1']:.3f}")
        st.metric("Val Accuracy", f"{rec['val_acc']:.1%}")
        st.metric("Epoch", rec["epoch"])
        if rec.get("checkpoint_saved"):
            st.success("Best checkpoint saved at this epoch")

with cm_col2:
    if cm_records:
        cm = np.array(rec["confusion_matrix"])
        # Normalize rows to recall %
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_pct = cm / row_sums

        fig_cm = px.imshow(
            cm_pct,
            x=CLASS_NAMES, y=CLASS_NAMES,
            color_continuous_scale="Blues",
            zmin=0, zmax=1,
            labels=dict(x="Predicted", y="True", color="Recall"),
            text_auto=".1%",
        )
        fig_cm.update_traces(textfont_size=14)
        fig_cm.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#fafafa",
            coloraxis_colorbar=dict(tickformat=".0%"),
            height=380,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# ── Section 5: Run comparison table ──────────────────────────────────────────
st.subheader("Run summary")

rows = []
for label, run in RUNS.items():
    records = load_metrics(run)
    if not records:
        continue
    best = max(records, key=lambda r: r["val_macro_f1"])
    final = records[-1]
    rows.append({
        "Run": run,
        "Epochs": final["epoch"],
        "Best macro-F1": f"{best['val_macro_f1']:.3f}",
        "Best epoch": best["epoch"],
        "Final val loss": f"{final['val_loss']:.4f}",
        "Normal recall": f"{best['val_recalls']['Normal']:.1%}",
        "AF recall":     f"{best['val_recalls']['AF']:.1%}",
        "Other recall":  f"{best['val_recalls']['Other']:.1%}",
        "Noisy recall":  f"{best['val_recalls']['Noisy']:.1%}",
    })

st.dataframe(rows, use_container_width=True, hide_index=True)
