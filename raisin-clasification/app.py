import os
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

# =========================
# 1) Robust path resolver
# =========================
def pick_path(*candidates: str) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"File not found. Tried: {candidates}")

MODEL_PATH = pick_path("models/knn_model.joblib", "knn_model.joblib")
SCALER_PATH = pick_path("models/scaler.joblib", "scaler.joblib")
META_PATH = pick_path("models/metadata.json", "metadata.json")

# =========================
# 2) Load artifacts
# =========================
knn = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(META_PATH, "r") as f:
    metadata = json.load(f)

FEATURES = list(metadata["features"])  # MUST be training order
CLASSES = list(metadata.get("classes", []))

# =========================
# 3) Optional dataset stats for defaults/clamp
#    (In HF Spaces there is no df, so clamp will be OFF.)
# =========================
HAS_DATA_STATS = False
defaults = {c: 0.0 for c in FEATURES}
minmax = {}  # empty = no clamp

def _clamp(x, lo, hi):
    x = float(x)
    return max(lo, min(hi, x))

# If you ever define df in notebook, this will activate automatically
try:
    df  # noqa: F821
    defaults = {c: float(df[c].median()) for c in FEATURES}
    minmax = {c: (float(df[c].min()), float(df[c].max())) for c in FEATURES}
    HAS_DATA_STATS = True
except Exception:
    HAS_DATA_STATS = False
    minmax = {}

# =========================
# 4) Core prediction (dict-based, order-safe)
# =========================
def predict_and_explain(vals: dict):
    # Build feature vector strictly by FEATURES order
    x_list = []
    for feat in FEATURES:
        v = float(vals.get(feat, 0.0))

        # Clamp only if real stats exist
        if HAS_DATA_STATS and feat in minmax:
            lo, hi = minmax[feat]
            v = _clamp(v, lo, hi)

        x_list.append(v)

    x = np.array(x_list, dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)

    pred = knn.predict(x_scaled)[0]

    # Probabilities
    if hasattr(knn, "predict_proba") and CLASSES:
        proba = knn.predict_proba(x_scaled)[0]
        proba_map = {CLASSES[i]: float(proba[i]) for i in range(len(CLASSES))}
    elif hasattr(knn, "predict_proba"):
        # fallback: use estimator's classes_
        est_classes = list(getattr(knn, "classes_", []))
        proba = knn.predict_proba(x_scaled)[0]
        proba_map = {est_classes[i]: float(proba[i]) for i in range(len(est_classes))}
    else:
        proba_map = {}

    # Confidence badge
    if proba_map:
        best_class = max(proba_map, key=proba_map.get)
        best_conf = proba_map[best_class]
    else:
        best_class = str(pred)
        best_conf = 0.0

    if best_conf >= 0.85:
        conf_badge = "High ✅"
    elif best_conf >= 0.65:
        conf_badge = "Medium ⚠️"
    else:
        conf_badge = "Low ❗"

    # Probability table
    df_proba = pd.DataFrame(
        [{"Class": k, "Probability": v} for k, v in sorted(proba_map.items(), key=lambda x: -x[1])]
    )

    # Plot
    fig = plt.figure()
    if len(df_proba) > 0:
        plt.bar(df_proba["Class"], df_proba["Probability"])
        plt.ylim(0, 1)
        plt.title("Prediction Probability")
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.grid(True, axis="y", alpha=0.3)
    else:
        plt.title("Prediction Probability (not available)")
    plt.tight_layout()

    details = (
        f"**Predicted:** `{pred}`\n\n"
        f"**Top Confidence:** `{best_class}` = `{best_conf:.2%}`\n\n"
        f"**Confidence Level:** **{conf_badge}**\n\n"
        f"**Model:** KNN (metric=`{metadata.get('metric','-')}`, K=`{metadata.get('best_k','-')}`)\n"
    )

    return pred, proba_map, df_proba, fig, details

# =========================
# 5) UI (safe mapping!)
# =========================
theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate")

with gr.Blocks(theme=theme, css="""
#title {text-align:center; margin-bottom: 8px;}
.small {opacity: 0.85; font-size: 0.95rem;}
""") as demo:

    gr.Markdown(
        """
        <h1 id="title">🍇 Raisin Classification (Kecimen vs Besni)</h1>
        <p class="small" style="text-align:center;">
        Masukkan fitur raisin, lalu klik <b>Predict</b> untuk melihat hasil, confidence, dan probabilitas.
        </p>
        """
    )

    # Create inputs by FEATURES order, but display in 2 columns
    inputs = {}
    with gr.Row():
        with gr.Column(scale=6):
            gr.Markdown("### 🧾 Input Features")
            for feat in FEATURES[: (len(FEATURES)+1)//2]:
                inputs[feat] = gr.Number(label=feat, value=defaults.get(feat, 0.0))
        with gr.Column(scale=6):
            gr.Markdown("### 🧾 Input Features (cont.)")
            for feat in FEATURES[(len(FEATURES)+1)//2 :]:
                inputs[feat] = gr.Number(label=feat, value=defaults.get(feat, 0.0))

    with gr.Row():
        btn_pred = gr.Button("🔮 Predict", variant="primary")
        btn_clear = gr.Button("🧹 Clear", variant="secondary")

    with gr.Row():
        with gr.Column(scale=6):
            gr.Markdown("### ✅ Prediction Result")
            pred_out = gr.Textbox(label="Predicted Class", interactive=False)
            conf_out = gr.Label(label="Confidence", num_top_classes=2)

        with gr.Column(scale=6):
            gr.Markdown("### 📊 Probability Details")
            proba_table = gr.Dataframe(
                headers=["Class", "Probability"],
                datatype=["str", "number"],
                interactive=False,
                wrap=True
            )
            proba_plot = gr.Plot()

    details_md = gr.Markdown()

    # Wrapper: build vals dict from UI in correct order
    def predict_from_ui(*vals_in):
        vals_dict = {FEATURES[i]: vals_in[i] for i in range(len(FEATURES))}
        return predict_and_explain(vals_dict)

    btn_pred.click(
        fn=predict_from_ui,
        inputs=[inputs[f] for f in FEATURES],
        outputs=[pred_out, conf_out, proba_table, proba_plot, details_md]
    )

    def clear_all():
        return [defaults.get(f, 0.0) for f in FEATURES] + ["", {}, pd.DataFrame(columns=["Class","Probability"]), None, ""]

    btn_clear.click(
        fn=clear_all,
        inputs=[],
        outputs=[inputs[f] for f in FEATURES] + [pred_out, conf_out, proba_table, proba_plot, details_md]
    )

demo.launch()