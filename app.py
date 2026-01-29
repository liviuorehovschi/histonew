import os
import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Paths for sample images (work on HF and locally)
_BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMAGES = [
    os.path.join(_BASE, "test_images", "adenocarcinoma.jpg"),
    os.path.join(_BASE, "test_images", "benign_tissue.png"),
    os.path.join(_BASE, "test_images", "squamous_cell_carcinoma.png"),
]

# ============================================================
# MODEL
# ============================================================
MODEL_PATH = 'model/best_model_efficientnet.keras'
CLASS_NAMES = ['lung_aca', 'lung_n', 'lung_scc']
CLASS_LABELS = {
    'lung_aca': 'Lung Adenocarcinoma',
    'lung_n': 'Normal Lung Tissue',
    'lung_scc': 'Lung Squamous Cell Carcinoma'
}

print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
print("[SUCCESS] Model loaded!")

def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(arr)

def predict(image):
    if image is None:
        return None
    preds = model.predict(preprocess(image), verbose=0)
    return {CLASS_LABELS[c]: float(preds[0][i]) for i, c in enumerate(CLASS_NAMES)}

def make_gradcam(image):
    if image is None:
        return None
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    orig = arr.astype(np.uint8)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    preds = model.predict(arr, verbose=0)
    idx = np.argmax(preds[0])
    base = model.get_layer('efficientnetb0')
    conv = base.get_layer('top_conv')
    grad_model = tf.keras.models.Model(inputs=base.input, outputs=[conv.output, base.output])
    with tf.GradientTape() as tape:
        conv_out, pred_out = grad_model(tf.cast(arr, tf.float32))
        loss = pred_out[:, idx]
    grads = tape.gradient(loss, conv_out)[0]
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    cam = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(orig, 0.6, cam, 0.4, 0))

def make_saliency(image):
    if image is None:
        return None
    img = image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    tensor = tf.Variable(arr, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(tensor)
        preds = model(tensor, training=False)
        loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, tensor)
    sal = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    sal = (sal - tf.reduce_min(sal)) / (tf.reduce_max(sal) - tf.reduce_min(sal) + 1e-10)
    sal = np.uint8(255 * sal.numpy())
    heatmap = cv2.applyColorMap(sal, cv2.COLORMAP_HOT)
    return Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

def run_analysis(img):
    if img is None:
        return None, gr.update(visible=False)
    preds = predict(img)
    if not preds:
        return None, gr.update(visible=False)
    winner = max(preds, key=preds.get)
    conf = preds[winner]
    diagnosis_only = {winner: conf}
    return diagnosis_only, gr.update(visible=True)


def _path_to_pil_for_preview(val):
    """Ensure gr.Image receives a PIL image for display (not a string path). UI only."""
    if val is None:
        return None
    if isinstance(val, Image.Image):
        return val
    path = val
    if isinstance(val, dict):
        path = val.get("path") or val.get("url")
    if isinstance(path, str) and os.path.isfile(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return val

# ============================================================
# CSS
# ============================================================
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, .gradio-container {
    background: #000000 !important;
    color: #ffffff !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    min-height: 100vh !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
}

/* Allow scrolling */
.gradio-container {
    height: auto !important;
    overflow: visible !important;
}

.main, .wrap, .contain {
    overflow: visible !important;
    height: auto !important;
}

footer { display: none !important; }

/* Header */
#header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: rgba(0,0,0,0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    display: flex;
    align-items: center;
    padding: 0 24px;
    z-index: 9999;
}

#header a {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #fff;
    text-decoration: none;
    font-weight: 600;
    font-size: 18px;
}

#header img {
    width: 32px;
    height: 32px;
}

/* Signature watermark: persistent, bottom-right, clickable */
#watermark {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 9999;
    opacity: 0.22;
    transition: opacity 0.3s ease;
    cursor: pointer;
}

#watermark:hover {
    opacity: 0.5;
}

/* Tabs */
.tabs {
    margin-top: 60px !important;
    background: transparent !important;
    border: none !important;
}

/* Nav: clean, aligned, recruiter-ready */
.tab-nav {
    position: fixed !important;
    top: 60px !important;
    left: 0 !important;
    right: 0 !important;
    background: rgba(0,0,0,0.92) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 14px 24px !important;
    z-index: 9998 !important;
}

.tab-nav button {
    background: transparent !important;
    color: rgba(255,255,255,0.65) !important;
    border: none !important;
    padding: 10px 28px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: color 0.2s ease, background 0.2s ease !important;
}

.tab-nav button:hover {
    color: #fff !important;
    background: rgba(255,255,255,0.08) !important;
}

.tab-nav button.selected {
    color: #000 !important;
    background: #fff !important;
    font-weight: 600 !important;
}

@media (max-width: 640px) {
    .tab-nav { padding: 10px 12px !important; gap: 4px !important; }
    .tab-nav button { padding: 8px 16px !important; font-size: 14px !important; }
}

/* Tab Content */
.tabitem {
    padding: 140px 24px 100px 24px !important;
    background: transparent !important;
    border: none !important;
    min-height: calc(100vh - 60px) !important;
    overflow: visible !important;
}

/* Home Page */
.home-hero {
    max-width: 800px;
    margin: 0 auto 80px auto;
    text-align: center;
    padding-top: 40px;
}

.home-hero img {
    width: 80px;
    height: 80px;
    margin-bottom: 24px;
}

.home-hero h1 {
    font-size: 56px;
    font-weight: 700;
    letter-spacing: -2px;
    margin-bottom: 16px;
    background: linear-gradient(180deg, #fff 0%, rgba(255,255,255,0.7) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.home-hero p {
    font-size: 18px;
    color: rgba(255,255,255,0.6);
    line-height: 1.7;
    max-width: 500px;
    margin: 0 auto;
}

.features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    max-width: 900px;
    margin: 0 auto 60px auto;
}

@media (max-width: 768px) {
    .features { grid-template-columns: 1fr; }
    .home-hero h1 { font-size: 36px; }
}

.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 24px;
    transition: all 0.3s ease;
}

.feature-card:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.15);
    transform: translateY(-2px);
}

.feature-card h3 {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
}

.feature-card p {
    font-size: 14px;
    color: rgba(255,255,255,0.5);
    line-height: 1.6;
}

/* Diagnostic Page */
.diag-header {
    text-align: center;
    margin-bottom: 40px;
}

.diag-header h1 {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 8px;
}

.diag-header p {
    color: rgba(255,255,255,0.5);
}

/* Gradio overrides */
.gr-block, .gr-box, .gr-panel, .block {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

.gr-input, .gr-text-input, textarea, input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #fff !important;
    border-radius: 8px !important;
}

.gr-button, button {
    transition: all 0.2s ease !important;
}

button.primary, .gr-button-primary {
    background: #fff !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
}

button.primary:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

button.secondary, button:not(.primary) {
    background: rgba(255,255,255,0.08) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
}

button.secondary:hover, button:not(.primary):hover {
    background: rgba(255,255,255,0.12) !important;
}

/* Confidence interpretation (UX only, non-medical) */
.confidence-help {
    font-size: 13px;
    color: rgba(255,255,255,0.5);
    margin-top: 8px;
    line-height: 1.5;
}

.confidence-help-icon {
    opacity: 0.8;
    cursor: help;
}

/* Image component */
.image-container, .gr-image, [data-testid="image"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}

/* Label */
.output-label, .gr-label {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}

/* Examples */
.gr-examples, .gr-sample-textbox {
    background: transparent !important;
    border: none !important;
}

.gr-samples-table button, .gr-sample-btn {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.gr-samples-table button:hover, .gr-sample-btn:hover {
    border-color: #fff !important;
}

/* Remove blue */
:root {
    --color-accent: #ffffff !important;
    --color-accent-soft: rgba(255,255,255,0.1) !important;
}

*:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.2) !important;
}

/* About page */
.about-content {
    max-width: 680px;
    margin: 0 auto;
    padding-bottom: 80px;
}

.about-content h1 {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 32px;
}

.about-section {
    margin-bottom: 32px;
}

.about-section h2 {
    font-size: 12px;
    font-weight: 600;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

.about-section p {
    color: rgba(255,255,255,0.7);
    line-height: 1.8;
    margin-bottom: 12px;
}

.about-section ul {
    color: rgba(255,255,255,0.6);
    padding-left: 20px;
    line-height: 1.8;
}

.about-section li {
    margin-bottom: 6px;
}

.about-section strong {
    color: #fff;
}

.author-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 24px;
    margin-top: 40px;
}

.author-box h4 {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
}

.author-box a {
    color: rgba(255,255,255,0.6);
    text-decoration: none;
    margin-right: 16px;
    font-size: 14px;
    transition: color 0.2s ease;
}

.author-box a:hover {
    color: #fff;
}

/* Disclaimer */
.disclaimer {
    text-align: center;
    color: rgba(255,255,255,0.4);
    font-size: 13px;
    margin-top: 32px;
    padding-top: 24px;
    border-top: 1px solid rgba(255,255,255,0.08);
}

/* Visualization section */
.viz-section {
    margin-top: 20px;
}
"""

# ============================================================
# HTML
# ============================================================
header = '''
<div id="header">
    <a href="#" onclick="document.querySelectorAll('.tab-nav button')[0].click(); return false;">
        <img src="https://huggingface.co/spaces/liviuorehovschi/histomancer/resolve/main/histo.ico" alt="">
        <span>Histomancer</span>
    </a>
</div>
'''

watermark = '''
<a id="watermark" href="https://orehovschi.com" target="_blank">
    <svg viewBox="200 80 520 120" width="70" fill="none">
        <path d="M220.228 124.394C216.295 136.784 197.467 151.923 213.286 164.067C233.494 179.581 311.081 114.354 287.264 97.9612C267.04 84.0407 230.956 101.639 227.963 116.478M275.112 144.263C303.698 122.796 289.382 147.552 285.867 155.813C283.982 160.244 311.078 154.469 314.689 153.41M314.689 153.41C359.924 140.146 311.373 127.238 314.689 153.41ZM314.689 153.41C316.788 169.975 356.198 147.569 361.872 143.865M361.872 143.865C379.996 132.033 396.654 110.1 396.142 93.0151C395.587 74.4442 371.8 126.504 361.872 143.865ZM361.872 143.865C358.713 149.388 356.853 155.059 354.151 160.646C353.56 161.868 357.043 156.178 366.562 150.218C393.466 133.372 380.946 180.567 408.42 157.542M408.42 157.542C422.113 146.068 406.519 142.186 408.42 157.542ZM408.42 157.542C409.261 164.335 419.291 164.376 427.904 164.518C446.758 164.829 436.156 128.918 413.569 144.253M443.616 141.885C461.237 189.037 461.247 141.561 494.921 144.092M530.069 142.59C492.635 142.764 525.754 153.079 519.876 160.419C508.831 167.863 482.829 171.175 468.519 176M567.777 142.645C548.176 142.644 535.687 158.692 551.095 168.622C563.172 176.406 592.423 158.35 598.782 153.314M598.782 153.314C612.006 142.84 620.551 100.564 615.577 113.834C610.591 127.137 604.922 140.225 598.782 153.314ZM598.782 153.314C586.286 179.948 593.453 164.248 617.545 156.091C635.029 150.172 619.238 171.51 629.98 172.179C645.463 173.144 652.018 163.388 658.901 155.799M660.543 135.24C663.427 135.24 666.987 132.79 669 131.565" stroke="#ffffff" stroke-width="4" stroke-linecap="round"/>
    </svg>
</a>
'''

home_page = '''
<div class="home-hero">
    <img src="https://huggingface.co/spaces/liviuorehovschi/histomancer/resolve/main/histo.ico" alt="">
    <h1>Histomancer</h1>
    <p>Lung tissue classification from histopathology slides with explainable predictions.</p>
</div>
<div class="features">
    <div class="feature-card">
        <h3>Classification</h3>
        <p>Identifies adenocarcinoma, squamous cell carcinoma, and normal tissue with 98% accuracy.</p>
    </div>
    <div class="feature-card">
        <h3>Explainability</h3>
        <p>Grad-CAM and saliency maps show which regions influenced the prediction.</p>
    </div>
    <div class="feature-card">
        <h3>Transparency</h3>
        <p>Open source research tool with confidence scores for every prediction.</p>
    </div>
</div>
'''

about_page = '''
<div class="about-content">
    <h1>About</h1>

    <div class="about-section">
        <h2>What It Does</h2>
        <p>Histomancer classifies lung histopathology images into adenocarcinoma, squamous cell carcinoma, or normal tissue using deep learning.</p>
    </div>

    <div class="about-section">
        <h2>Technical</h2>
        <p><strong>Model:</strong> EfficientNet B0</p>
        <p><strong>Input:</strong> 224x224 histopathology images</p>
        <p><strong>Explainability:</strong> Grad-CAM and saliency mapping</p>
    </div>

    <div class="about-section">
        <h2>Data</h2>
        <p><strong>LC25000:</strong> ~25,000 augmented images</p>
        <p><strong>LungHist700:</strong> 691 real clinical samples</p>
    </div>

    <div class="about-section">
        <h2>Limitations</h2>
        <ul>
            <li>Training data includes synthetic augmentation</li>
            <li>Not validated across multiple institutions</li>
            <li>For research and education only</li>
        </ul>
    </div>

    <div class="author-box">
        <h4>Liviu Orehovschi</h4>
        <a href="https://orehovschi.com" target="_blank">Portfolio</a>
        <a href="https://github.com/liviuorehovschi" target="_blank">GitHub</a>
        <a href="https://linkedin.com/in/liviuorehovschi" target="_blank">LinkedIn</a>
    </div>
</div>
'''

# ============================================================
# APP (all visible UI is defined inside this Blocks layout)
# ============================================================
with gr.Blocks(css=css, title="Histomancer") as demo:
    gr.HTML(header)
    gr.HTML(watermark)

    # Navigation: Gradio Tabs (Home / Diagnostic / About)
    with gr.Tabs():
        with gr.Tab("Home"):
            gr.HTML(home_page)

        with gr.Tab("Diagnostic"):
            gr.Markdown("## Analyze Tissue")
            gr.Markdown("Upload an image or choose a sample below. **Preview appears here; analysis runs only when you press Analyze.**")

            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Image preview", height=300, sources=["upload"])
                    # Convert path → PIL so preview always shows an image (no broken icon)
                    img_input.change(_path_to_pil_for_preview, inputs=img_input, outputs=img_input)

                    gr.Markdown("---")
                    gr.Markdown("**Sample images (playable demos)** — click one to load it in the preview above. Then press **Analyze** to run.")
                    # Pass PIL images to Examples so gr.Image receives image objects, not paths
                    _example_pils = []
                    for _p in SAMPLE_IMAGES:
                        if os.path.exists(_p):
                            try:
                                _example_pils.append([Image.open(_p).convert("RGB")])
                            except Exception:
                                pass
                    if not _example_pils:
                        _example_pils = [[_p] for _p in SAMPLE_IMAGES]
                    gr.Examples(examples=_example_pils, inputs=img_input, label="Samples")

                    btn_analyze = gr.Button("Analyze", variant="primary")

                with gr.Column(scale=1):
                    results = gr.Label(label="Diagnosis", num_top_classes=1)
                    gr.Markdown("*Confidence is informational only. High = more certain; low = uncertainty.*")
                    # Grad-CAM and Saliency: hidden until AFTER analysis (only Analyze button runs analysis)
                    with gr.Column(visible=False) as viz_col:
                        gr.Markdown("**Explainability**")
                        with gr.Row():
                            btn_gradcam = gr.Button("Grad-CAM")
                            btn_saliency = gr.Button("Saliency")
                        with gr.Row():
                            out_gradcam = gr.Image(type="pil", label="Grad-CAM", height=200)
                            out_saliency = gr.Image(type="pil", label="Saliency", height=200)

            gr.Markdown("---")
            gr.Markdown("*For research and education only. Not for clinical use.*")

            btn_analyze.click(run_analysis, inputs=img_input, outputs=[results, viz_col])
            btn_gradcam.click(make_gradcam, inputs=img_input, outputs=out_gradcam)
            btn_saliency.click(make_saliency, inputs=img_input, outputs=out_saliency)

        with gr.Tab("About"):
            gr.Markdown("## About Histomancer")
            gr.Markdown("**What it does** — Classifies lung histopathology images into adenocarcinoma, squamous cell carcinoma, or normal tissue using deep learning.")
            gr.Markdown("**Model** — EfficientNet B0. Input: 224×224 images. Explainability: Grad-CAM and saliency mapping.")
            gr.Markdown("**Data** — LC25000 (~25k augmented images), LungHist700 (691 clinical samples).")
            gr.Markdown("**Limitations** — Training data includes synthetic augmentation; not validated across institutions; for research and education only.")
            gr.Markdown("**Author** — [Liviu Orehovschi](https://orehovschi.com) · [GitHub](https://github.com/liviuorehovschi) · [LinkedIn](https://linkedin.com/in/liviuorehovschi)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
