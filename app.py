import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

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

def preprocess_image(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(arr)

def predict(image):
    if image is None:
        return None
    preds = model.predict(preprocess_image(image), verbose=0)
    return {CLASS_LABELS[c]: float(preds[0][i]) for i, c in enumerate(CLASS_NAMES)}

def generate_gradcam(image):
    if image is None:
        return None
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    resized = arr.astype(np.uint8)
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
    return Image.fromarray(cv2.addWeighted(resized, 0.6, cam, 0.4, 0))

def generate_saliency(image):
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

# ============================================================
# HANDLERS
# ============================================================
def do_analyze(img):
    if img is None:
        return None, gr.update(visible=False), gr.update(visible=False)
    result = predict(img)
    return result, gr.update(visible=True), gr.update(visible=True)

def do_gradcam(img):
    return generate_gradcam(img) if img else None

def do_saliency(img):
    return generate_saliency(img) if img else None

# ============================================================
# CSS - MODERN DESIGN
# ============================================================
css = """
/* Reset & Base */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg: #0a0a0a;
    --bg-card: #111111;
    --bg-elevated: #1a1a1a;
    --border: #222222;
    --border-hover: #333333;
    --text: #ffffff;
    --text-secondary: #888888;
    --accent: #ffffff;
    --radius: 16px;
    --radius-sm: 12px;
    --shadow: 0 4px 24px rgba(0,0,0,0.4);
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

html, body {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
    line-height: 1.6;
    overflow-x: hidden;
}

.gradio-container {
    background: var(--bg) !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

footer, .footer { display: none !important; }

/* Fixed Header */
.site-header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 64px;
    background: rgba(10, 10, 10, 0.8);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 32px;
    z-index: 10000;
}

.site-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    text-decoration: none;
    color: var(--text);
    transition: var(--transition);
}

.site-logo:hover { opacity: 0.8; }
.site-logo img { width: 36px; height: 36px; }
.site-logo span { font-size: 20px; font-weight: 600; letter-spacing: -0.02em; }

/* Tabs Navigation - Fixed */
.tabs {
    position: fixed !important;
    top: 64px !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 9999 !important;
    background: rgba(10, 10, 10, 0.9) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: none !important;
}

.tab-nav, [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    display: flex !important;
    justify-content: center !important;
    gap: 8px !important;
    padding: 12px 0 !important;
}

.tab-nav button, [role="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    padding: 12px 28px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    border-radius: 8px !important;
    transition: var(--transition) !important;
    cursor: pointer !important;
}

.tab-nav button:hover, [role="tab"]:hover {
    color: var(--text) !important;
    background: var(--bg-elevated) !important;
}

.tab-nav button.selected, [role="tab"][aria-selected="true"] {
    color: var(--bg) !important;
    background: var(--accent) !important;
    font-weight: 600 !important;
}

/* Content spacing for fixed elements */
.tabitem, [role="tabpanel"] {
    padding-top: 140px !important;
    min-height: 100vh !important;
    background: transparent !important;
    border: none !important;
}

/* Watermark - Fixed */
.watermark {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 10000;
    opacity: 0.15;
    transition: var(--transition);
}
.watermark:hover { opacity: 0.4; transform: scale(1.05); }
.watermark svg { width: 80px; height: auto; }

/* Hero Section */
.hero-section {
    min-height: calc(100vh - 140px);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 60px 24px;
    animation: fadeInUp 0.8s ease-out;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-logo {
    width: 100px;
    height: 100px;
    margin-bottom: 32px;
    animation: float 6s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

.hero-title {
    font-size: clamp(48px, 8vw, 80px);
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #fff 0%, #888 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 20px;
    color: var(--text-secondary);
    max-width: 560px;
    margin: 0 auto 48px;
    line-height: 1.7;
}

/* Feature Cards */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 24px;
    max-width: 1000px;
    margin: 0 auto;
    padding: 0 24px 80px;
}

.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px;
    transition: var(--transition);
    animation: fadeInUp 0.6s ease-out backwards;
}

.feature-card:nth-child(1) { animation-delay: 0.1s; }
.feature-card:nth-child(2) { animation-delay: 0.2s; }
.feature-card:nth-child(3) { animation-delay: 0.3s; }

.feature-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-4px);
    box-shadow: var(--shadow);
}

.feature-icon {
    width: 48px;
    height: 48px;
    background: var(--bg-elevated);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 20px;
}

.feature-card h3 {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 12px;
}

.feature-card p {
    color: var(--text-secondary);
    font-size: 15px;
    line-height: 1.6;
}

/* Diagnostic Page */
.diagnostic-container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 24px 100px;
}

.diagnostic-header {
    text-align: center;
    margin-bottom: 48px;
    animation: fadeInUp 0.6s ease-out;
}

.diagnostic-header h1 {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 12px;
}

.diagnostic-header p {
    color: var(--text-secondary);
    font-size: 16px;
}

/* Gradio Component Overrides */
.gr-panel, .gr-box, .gr-form, .block {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    transition: var(--transition) !important;
}

.gr-panel:hover, .block:hover {
    border-color: var(--border-hover) !important;
}

/* Image Upload */
.image-container, [data-testid="image"], .gr-image {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    min-height: 300px !important;
    transition: var(--transition) !important;
    overflow: hidden !important;
}

.image-container:hover, [data-testid="image"]:hover {
    border-color: var(--text-secondary) !important;
    border-style: dashed !important;
}

.image-container img, [data-testid="image"] img {
    border-radius: var(--radius-sm) !important;
    object-fit: contain !important;
}

/* Buttons */
button, .gr-button {
    transition: var(--transition) !important;
}

.gr-button-primary, button.primary, button[variant="primary"] {
    background: var(--accent) !important;
    color: var(--bg) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 14px 32px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
}

.gr-button-primary:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(255,255,255,0.15) !important;
}

.gr-button-secondary, button:not(.primary):not([variant="primary"]) {
    background: var(--bg-elevated) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 24px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

.gr-button-secondary:hover, button:not(.primary):hover {
    background: var(--bg-card) !important;
    border-color: var(--border-hover) !important;
}

/* Labels */
.label-wrap, .gr-label, .output-label {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
}

/* Examples */
.examples-holder, .gr-examples {
    background: transparent !important;
    border: none !important;
}

.examples-holder button, .gr-sample-btn {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px !important;
    transition: var(--transition) !important;
}

.examples-holder button:hover, .gr-sample-btn:hover {
    border-color: var(--accent) !important;
    transform: scale(1.02) !important;
}

/* Remove blue accents */
:root {
    --color-accent: var(--accent) !important;
    --color-accent-soft: var(--bg-elevated) !important;
    --checkbox-background-color-selected: var(--text-secondary) !important;
    --slider-color: var(--accent) !important;
}

*:focus, *:focus-visible {
    outline: 2px solid var(--text-secondary) !important;
    outline-offset: 2px !important;
}

/* About Page */
.about-container {
    max-width: 720px;
    margin: 0 auto;
    padding: 40px 24px 120px;
    animation: fadeInUp 0.6s ease-out;
}

.about-container h1 {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 40px;
    letter-spacing: -0.02em;
}

.about-section {
    margin-bottom: 40px;
}

.about-section h2 {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}

.about-section p {
    color: var(--text-secondary);
    font-size: 16px;
    line-height: 1.8;
    margin-bottom: 16px;
}

.about-section ul {
    color: var(--text-secondary);
    padding-left: 24px;
    line-height: 1.8;
}

.about-section li { margin-bottom: 8px; }
.about-section strong { color: var(--text); }
.about-section a { color: var(--text); text-decoration: underline; }

.author-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px;
    margin-top: 48px;
    transition: var(--transition);
}

.author-card:hover {
    border-color: var(--border-hover);
}

.author-card h4 {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 16px;
}

.author-links {
    display: flex;
    gap: 20px;
}

.author-links a {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 14px;
    transition: var(--transition);
}

.author-links a:hover { color: var(--text); }

/* Viz buttons row */
.viz-btn-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
}

/* Disclaimer */
.disclaimer {
    text-align: center;
    color: var(--text-secondary);
    font-size: 13px;
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid var(--border);
}

/* Responsive */
@media (max-width: 768px) {
    .site-header { padding: 0 16px; }
    .hero-section { padding: 40px 16px; }
    .features-grid { padding: 0 16px 60px; }
    .diagnostic-container { padding: 24px 16px 80px; }
    .about-container { padding: 24px 16px 100px; }
    .tab-nav button, [role="tab"] { padding: 10px 16px !important; font-size: 13px !important; }
}
"""

# ============================================================
# HTML
# ============================================================
header_html = '''
<div class="site-header">
    <a href="#" class="site-logo" onclick="document.querySelectorAll('[role=tab]')[0].click(); return false;">
        <img src="https://huggingface.co/spaces/liviuorehovschi/histomancer/resolve/main/histo.ico" alt="">
        <span>Histomancer</span>
    </a>
</div>
'''

watermark_html = '''
<a href="https://orehovschi.com" target="_blank" class="watermark" title="orehovschi.com">
<svg viewBox="200 80 520 120" fill="none"><path d="M220.228 124.394C216.295 136.784 197.467 151.923 213.286 164.067C233.494 179.581 311.081 114.354 287.264 97.9612C267.04 84.0407 230.956 101.639 227.963 116.478M275.112 144.263C303.698 122.796 289.382 147.552 285.867 155.813C283.982 160.244 311.078 154.469 314.689 153.41M314.689 153.41C359.924 140.146 311.373 127.238 314.689 153.41ZM314.689 153.41C316.788 169.975 356.198 147.569 361.872 143.865M361.872 143.865C379.996 132.033 396.654 110.1 396.142 93.0151C395.587 74.4442 371.8 126.504 361.872 143.865ZM361.872 143.865C358.713 149.388 356.853 155.059 354.151 160.646C353.56 161.868 357.043 156.178 366.562 150.218C393.466 133.372 380.946 180.567 408.42 157.542M408.42 157.542C422.113 146.068 406.519 142.186 408.42 157.542ZM408.42 157.542C409.261 164.335 419.291 164.376 427.904 164.518C446.758 164.829 436.156 128.918 413.569 144.253M443.616 141.885C461.237 189.037 461.247 141.561 494.921 144.092M530.069 142.59C492.635 142.764 525.754 153.079 519.876 160.419C508.831 167.863 482.829 171.175 468.519 176M567.777 142.645C548.176 142.644 535.687 158.692 551.095 168.622C563.172 176.406 592.423 158.35 598.782 153.314M598.782 153.314C612.006 142.84 620.551 100.564 615.577 113.834C610.591 127.137 604.922 140.225 598.782 153.314ZM598.782 153.314C586.286 179.948 593.453 164.248 617.545 156.091C635.029 150.172 619.238 171.51 629.98 172.179C645.463 173.144 652.018 163.388 658.901 155.799M660.543 135.24C663.427 135.24 666.987 132.79 669 131.565" stroke="#fff" stroke-width="4" stroke-linecap="round"/></svg>
</a>
'''

# Lucide icons as SVG
icon_microscope = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/></svg>'
icon_eye = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>'
icon_shield = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'

home_html = f'''
<div class="hero-section">
    <img src="https://huggingface.co/spaces/liviuorehovschi/histomancer/resolve/main/histo.ico" class="hero-logo" alt="">
    <h1 class="hero-title">Histomancer</h1>
    <p class="hero-subtitle">Lung tissue classification from histopathology slides using deep learning with explainable visualizations.</p>
</div>
<div class="features-grid">
    <div class="feature-card">
        <div class="feature-icon">{icon_microscope}</div>
        <h3>Classification</h3>
        <p>Identifies adenocarcinoma, squamous cell carcinoma, and normal lung tissue using EfficientNet B0 with 98% accuracy.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">{icon_eye}</div>
        <h3>Explainability</h3>
        <p>Visualize model attention with Grad CAM heatmaps and saliency maps to understand prediction reasoning.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">{icon_shield}</div>
        <h3>Transparency</h3>
        <p>Open source research tool. All predictions include confidence scores and visual explanations.</p>
    </div>
</div>
'''

about_html = '''
<div class="about-container">
    <h1>About Histomancer</h1>

    <div class="about-section">
        <h2>Overview</h2>
        <p>Histomancer is a deep learning tool that classifies lung histopathology images into adenocarcinoma, squamous cell carcinoma, or normal tissue. Built to explore how CNNs learn histological patterns with transparent, explainable predictions.</p>
    </div>

    <div class="about-section">
        <h2>Technical Details</h2>
        <p><strong>Model:</strong> EfficientNet B0 with transfer learning from ImageNet</p>
        <p><strong>Input:</strong> 224x224 pixel histopathology images</p>
        <p><strong>Output:</strong> Classification with confidence scores</p>
        <p><strong>Explainability:</strong> Grad CAM and saliency mapping</p>
    </div>

    <div class="about-section">
        <h2>Training Data</h2>
        <p><strong>LC25000:</strong> ~25,000 augmented histopathology images providing scale and class balance.</p>
        <p><strong>LungHist700:</strong> 691 real clinical samples added for improved generalization.</p>
    </div>

    <div class="about-section">
        <h2>Limitations</h2>
        <ul>
            <li>Training data includes synthetic augmentation</li>
            <li>Confidence scores are softmax outputs, not calibrated probabilities</li>
            <li>Not validated across multiple institutions</li>
            <li>Intended for research and education only</li>
        </ul>
    </div>

    <div class="author-card">
        <h4>Liviu Orehovschi</h4>
        <div class="author-links">
            <a href="https://orehovschi.com" target="_blank">Portfolio</a>
            <a href="https://github.com/liviuorehovschi" target="_blank">GitHub</a>
            <a href="https://linkedin.com/in/liviuorehovschi" target="_blank">LinkedIn</a>
        </div>
    </div>
</div>
'''

# ============================================================
# APP
# ============================================================
with gr.Blocks(css=css, title="Histomancer") as demo:
    gr.HTML(header_html)
    gr.HTML(watermark_html)

    with gr.Tabs():
        with gr.Tab("Home"):
            gr.HTML(home_html)

        with gr.Tab("Diagnostic"):
            gr.HTML('<div class="diagnostic-header"><h1>Analyze Tissue</h1><p>Upload a histopathology image or click a sample below</p></div>')

            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Upload Image", height=320)

                    gr.Markdown("**Sample Images** — click to load")
                    gr.Examples(
                        examples=[
                            ["test_images/adenocarcinoma.jpg"],
                            ["test_images/benign_tissue.png"],
                            ["test_images/squamous_cell_carcinoma.png"]
                        ],
                        inputs=img_input,
                        label=""
                    )

                    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

                with gr.Column(scale=1):
                    result_label = gr.Label(label="Classification Results", num_top_classes=3)

                    viz_row = gr.Row(visible=False)
                    with viz_row:
                        gradcam_btn = gr.Button("Generate Grad CAM")
                        saliency_btn = gr.Button("Generate Saliency Map")

                    viz_output_row = gr.Row(visible=False)
                    with viz_output_row:
                        gradcam_img = gr.Image(type="pil", label="Grad CAM", height=240)
                        saliency_img = gr.Image(type="pil", label="Saliency Map", height=240)

            gr.HTML('<div class="disclaimer">For research and education only. Not intended for clinical diagnosis.</div>')

            analyze_btn.click(do_analyze, inputs=img_input, outputs=[result_label, viz_row, viz_output_row])
            gradcam_btn.click(do_gradcam, inputs=img_input, outputs=gradcam_img)
            saliency_btn.click(do_saliency, inputs=img_input, outputs=saliency_img)

        with gr.Tab("About"):
            gr.HTML(about_html)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
