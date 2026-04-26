"""
app.py — Predictive Maintenance Gradio Dashboard
==================================================
Entry point for Hugging Face Spaces (and local dev).
Imports preprocessing from pipeline.py and prediction logic from router.py.
Contains zero ML logic — only UI wiring.

Compatible with Gradio 6.x
"""

import os
import tempfile
import gradio as gr
import pandas as pd
from pipeline import preprocess, build_single_input
from router import predict

# ── Shared UI options ────────────────────────────────────────────────
PRIORITY_OPTIONS = [
    "Minimize missed failures",
    "Minimize unnecessary maintenance",
]
DETAIL_OPTIONS = [
    "Primary cause only",
    "All contributing causes",
]

# ── Custom CSS for a polished look ───────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.header-bar {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 16px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.header-bar h1 {
    color: #fff;
    font-size: 2em;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.header-bar p {
    color: #b8b8d0;
    margin: 0;
    font-size: 0.95em;
}

.result-card {
    padding: 24px;
    border-radius: 14px;
    margin-top: 12px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
}
.result-ok {
    background: linear-gradient(135deg, #0d3320, #1a5c38);
    border-left: 5px solid #2ecc71;
}
.result-fail {
    background: linear-gradient(135deg, #3b0f0f, #5c1a1a);
    border-left: 5px solid #e74c3c;
}
.result-card .status {
    font-size: 1.35em;
    font-weight: 700;
    margin-bottom: 8px;
}
.result-ok .status  { color: #2ecc71; }
.result-fail .status { color: #e74c3c; }
.result-card .reason {
    font-size: 1.05em;
    color: #ddd;
    margin-bottom: 6px;
}
.result-card .meta {
    font-size: 0.82em;
    color: #999;
}
"""


# ── Tab 1: Manual single prediction ─────────────────────────────────
def predict_single(
    air_temp, process_temp, rpm, torque, tool_wear,
    machine_type, business_priority, diagnostic_detail,
):
    try:
        df = build_single_input(
            air_temp, process_temp, rpm, torque, tool_wear, machine_type
        )
        X = preprocess(df)
        res = predict(X, business_priority, diagnostic_detail)[0]
    except Exception as exc:
        return f"<div style='color:#e74c3c;font-weight:600'>Error: {exc}</div>"

    if res["failure_predicted"]:
        return f"""
        <div class='result-card result-fail'>
            <div class='status'>WARNING: FAILURE DETECTED</div>
            <div class='reason'>Predicted cause: <strong>{res['failure_reason']}</strong></div>
            <div class='meta'>
                Priority: {business_priority} &nbsp;|&nbsp;
                Detail: {diagnostic_detail}
            </div>
        </div>"""
    else:
        return f"""
        <div class='result-card result-ok'>
            <div class='status'>SYSTEM NORMAL</div>
            <div class='reason'>No failure predicted - machine operating within safe parameters.</div>
            <div class='meta'>
                Priority: {business_priority} &nbsp;|&nbsp;
                Detail: {diagnostic_detail}
            </div>
        </div>"""


# ── Tab 2: Batch CSV prediction ──────────────────────────────────────
def predict_batch(csv_file, business_priority, diagnostic_detail):
    if csv_file is None:
        gr.Warning("Please upload a CSV file first.")
        return None, None

    try:
        filepath = csv_file.name if hasattr(csv_file, "name") else csv_file
        df_raw = pd.read_csv(filepath)
    except Exception as exc:
        gr.Warning(f"Failed to read CSV: {exc}")
        return None, None

    if "Type" not in df_raw.columns:
        gr.Warning(
            "CSV is missing the 'Type' column (expected values: L, M, H). "
            "Cannot proceed."
        )
        return None, None

    try:
        X = preprocess(df_raw.copy())
        results = predict(X, business_priority, diagnostic_detail)
    except Exception as exc:
        gr.Warning(f"Prediction failed: {exc}")
        return None, None

    df_out = df_raw.copy()
    df_out["Predicted_Failure"] = [int(r["failure_predicted"]) for r in results]
    df_out["Failure_Reason"]    = [r["failure_reason"] for r in results]

    output_path = os.path.join(tempfile.gettempdir(), "predictions_output.csv")
    df_out.to_csv(output_path, index=False)

    preview = df_out.head(15)
    return preview, output_path


# ── Build Gradio interface (Gradio 6.x compatible) ───────────────────
with gr.Blocks(
    title="Predictive Maintenance Dashboard",
) as demo:

    # ── Header ───────────────────────────────────────────────────────
    gr.HTML("""
    <div class='header-bar'>
        <h1>Predictive Maintenance Dashboard</h1>
        <p>Real-time failure prediction from industrial sensor data &mdash;
        powered by Decision Tree ensemble models</p>
    </div>
    """)

    # ── Tab 1 - Manual Input ─────────────────────────────────────────
    with gr.Tab("Manual Input"):
        gr.Markdown(
            "Enter sensor readings manually to get an instant prediction. "
            "Adjust the **Business Priority** and **Diagnostic Detail** to "
            "change model behaviour."
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown("### Sensor Readings")
                air_temp     = gr.Number(label="Air Temperature (K)",     value=298.1, minimum=290, maximum=310)
                process_temp = gr.Number(label="Process Temperature (K)", value=308.6, minimum=300, maximum=320)
                rpm          = gr.Number(label="Rotational Speed (RPM)",  value=1551,  minimum=0,   maximum=3000)
                torque       = gr.Number(label="Torque (Nm)",             value=42.8,  minimum=0,   maximum=100)
                tool_wear    = gr.Number(label="Tool Wear (min)",         value=0,     minimum=0,   maximum=300)
                mtype        = gr.Dropdown(
                    label="Machine Type",
                    choices=["L", "M", "H"],
                    value="M",
                    info="L = Low quality, M = Medium, H = High",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Model Configuration")
                priority = gr.Radio(
                    label="Business Priority",
                    choices=PRIORITY_OPTIONS,
                    value=PRIORITY_OPTIONS[1],
                    info="Cost-sensitive model penalises missed failures more heavily",
                )
                detail = gr.Radio(
                    label="Diagnostic Detail",
                    choices=DETAIL_OPTIONS,
                    value=DETAIL_OPTIONS[0],
                    info="Multi-class = single root cause / Multi-label = all concurrent causes",
                )
                run_btn = gr.Button("Run Prediction", variant="primary", size="lg")

                gr.Markdown("### Prediction Result")
                output = gr.HTML()

        run_btn.click(
            fn=predict_single,
            inputs=[air_temp, process_temp, rpm, torque, tool_wear, mtype, priority, detail],
            outputs=output,
        )

    # ── Tab 2 - Batch CSV Upload ─────────────────────────────────────
    with gr.Tab("Batch CSV Upload"):
        gr.Markdown(
            "Upload a CSV with the same columns as the training data. "
            "`Predicted_Failure` and `Failure_Reason` columns will be appended."
        )
        with gr.Row():
            with gr.Column(scale=2):
                csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
            with gr.Column(scale=1):
                priority_b = gr.Radio(
                    label="Business Priority",
                    choices=PRIORITY_OPTIONS,
                    value=PRIORITY_OPTIONS[0],
                )
                detail_b = gr.Radio(
                    label="Diagnostic Detail",
                    choices=DETAIL_OPTIONS,
                    value=DETAIL_OPTIONS[0],
                )

        batch_btn   = gr.Button("Run Batch Prediction", variant="primary", size="lg")
        preview_tbl = gr.Dataframe(label="Preview (first 15 rows)", wrap=True)
        download_btn = gr.File(label="Download Full Results CSV")

        batch_btn.click(
            fn=predict_batch,
            inputs=[csv_input, priority_b, detail_b],
            outputs=[preview_tbl, download_btn],
        )

    # ── Footer ───────────────────────────────────────────────────────
    gr.Markdown(
        "<center style='color:#666;font-size:0.82em;margin-top:16px;'>"
        "Predictive Maintenance Project | Decision Tree Models | "
        "Built with Gradio</center>"
    )

# ── Launch ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        ssr_mode=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    )
