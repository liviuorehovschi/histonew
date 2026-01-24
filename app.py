import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# ============================================================
# LOAD MODEL ONCE AT STARTUP
# ============================================================
MODEL_PATH = 'model/best_model_efficientnet.keras'
CLASS_NAMES = ['lung_aca', 'lung_n', 'lung_scc']
CLASS_LABELS = {
    'lung_aca': 'Lung Adenocarcinoma',
    'lung_n': 'Normal Lung Tissue',
    'lung_scc': 'Lung Squamous Cell Carcinoma'
}

print("[INFO] Loading model at startup v2...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
print("[SUCCESS] Model loaded once and ready!")


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


def predict(image: Image.Image) -> dict:
    if image is None:
        return {}
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)
    results = {}
    for i, class_name in enumerate(CLASS_NAMES):
        label = CLASS_LABELS[class_name]
        results[label] = float(predictions[0][i])
    return results


def generate_gradcam(image: Image.Image) -> Image.Image:
    if image is None:
        return None
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    resized = img_array.astype(np.uint8)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    preds = model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds[0])
    base_model = model.get_layer('efficientnetb0')
    conv_layer = base_model.get_layer('top_conv')
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[conv_layer.output, base_model.output]
    )
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    cam = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    combined = cv2.addWeighted(resized, 0.6, cam, 0.4, 0)
    return Image.fromarray(combined)


def generate_saliency(image: Image.Image) -> Image.Image:
    if image is None:
        return None
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    image_tensor = tf.Variable(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        preds = model(image_tensor, training=False)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, image_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-10)
    saliency = np.uint8(255 * saliency.numpy())
    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap)


def analyze_image(image: Image.Image):
    if image is None:
        return {}
    try:
        if not isinstance(image, Image.Image):
            return {}
        if image.size[0] < 50 or image.size[1] < 50:
            return {}
        return predict(image)
    except Exception as e:
        print(f"[ERROR] Analysis failed: {str(e)}")
        return {}

def get_gradcam(image: Image.Image):
    if image is None:
        return None
    try:
        return generate_gradcam(image)
    except:
        return None

def get_saliency(image: Image.Image):
    if image is None:
        return None
    try:
        return generate_saliency(image)
    except:
        return None


# ============================================================
# CUSTOM CSS - MODERN MEDICAL AI DESIGN
# ============================================================
custom_css = """
/* No boot screen - keeping it simple */

/* ===== WATERMARK ===== */
.watermark {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 100;
    opacity: 0.08;
    transition: opacity 0.3s;
}
.watermark:hover { opacity: 0.25; }
.watermark svg { width: 90px; height: auto; }

/* ===== ROOT VARIABLES ===== */
:root {
    --bg: #0a0a0a;
    --bg-subtle: #141414;
    --bg-muted: #1a1a1a;
    --border: #262626;
    --border-subtle: #1a1a1a;
    --text: #fafafa;
    --text-muted: #a3a3a3;
    --text-dim: #737373;
    --primary: #fafafa;
    --primary-hover: #e5e5e5;
    --success: #16a34a;
    --warning: #ca8a04;
    --danger: #dc2626;
    --radius: 8px;
    --radius-sm: 6px;
}

/* ===== GLOBAL ===== */
*, *::before, *::after { box-sizing: border-box; }

body, html {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

.gradio-container {
    background: var(--bg) !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.main, .contain, .wrap {
    max-width: 100% !important;
    background: transparent !important;
}

footer { display: none !important; }

/* ===== TABS NAVIGATION ===== */
.tabs { background: transparent !important; border: none !important; }

.tab-nav, div[role="tablist"] {
    background: var(--bg-subtle) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 !important;
    gap: 0 !important;
    justify-content: center !important;
    display: flex !important;
}

.tab-nav button, button[role="tab"] {
    background: transparent !important;
    color: #e5e5e5 !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 18px 40px !important;
    margin: 0 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}

.tab-nav button:hover, button[role="tab"]:hover {
    color: #ffffff !important;
    background: rgba(255,255,255,0.05) !important;
}

.tab-nav button.selected, button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom-color: #ffffff !important;
    background: rgba(255,255,255,0.08) !important;
}

.tabitem, div[role="tabpanel"] { background: transparent !important; border: none !important; padding: 0 !important; }

/* ===== HOME PAGE ===== */
.home-wrapper {
    min-height: calc(100vh - 60px);
    display: flex;
    flex-direction: column;
    background: var(--bg);
}

.hero {
    padding: 80px 24px;
    text-align: center;
    background: linear-gradient(180deg, var(--bg-subtle) 0%, var(--bg) 100%);
    border-bottom: 1px solid var(--border-subtle);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: var(--bg-muted);
    border: 1px solid var(--border);
    border-radius: 100px;
    font-size: 13px;
    color: var(--text-muted);
    margin-bottom: 32px;
}

.hero-badge span { color: var(--success); }

.hero h1 {
    font-size: clamp(40px, 6vw, 64px);
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--text);
    margin: 0 0 20px 0;
    line-height: 1.1;
}

.hero p {
    font-size: 18px;
    color: var(--text-muted);
    max-width: 560px;
    margin: 0 auto 40px;
    line-height: 1.7;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 48px;
    flex-wrap: wrap;
}

.stat {
    text-align: center;
}

.stat-value {
    font-size: 36px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
}

.stat-label {
    font-size: 13px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

.features {
    padding: 80px 24px;
    max-width: 1200px;
    margin: 0 auto;
}

.features-header {
    text-align: center;
    margin-bottom: 48px;
}

.features-header h2 {
    font-size: 28px;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 12px 0;
}

.features-header p {
    color: var(--text-muted);
    font-size: 16px;
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 24px;
}

.feature {
    background: var(--bg-subtle);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 32px;
    transition: all 0.2s;
}

.feature:hover {
    border-color: var(--border);
    transform: translateY(-2px);
}

.feature-icon {
    width: 48px;
    height: 48px;
    background: var(--bg-muted);
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    margin-bottom: 20px;
}

.feature h3 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 8px 0;
}

.feature p {
    font-size: 14px;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.6;
}

/* ===== DIAGNOSTIC PAGE ===== */
.diagnostic-wrapper {
    max-width: 1100px;
    margin: 0 auto;
    padding: 48px 24px;
}

.diagnostic-header {
    text-align: center;
    margin-bottom: 40px;
}

.diagnostic-header h1 {
    font-size: 32px;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 8px 0;
}

.diagnostic-header p {
    color: var(--text-muted);
    font-size: 16px;
    margin: 0;
}

.section-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
}

/* Gradio component overrides */
.gradio-container .gr-box,
.gradio-container .gr-panel {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
}

.gradio-container .gr-input,
.gradio-container .gr-text-input,
.gradio-container textarea {
    background: var(--bg-muted) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius-sm) !important;
}

.gradio-container button.primary {
    background: var(--primary) !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}

.gradio-container button.primary:hover {
    background: var(--primary-hover) !important;
    transform: translateY(-1px) !important;
}

.gradio-container .label-container,
.gradio-container .output-class {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
}

/* Image upload and preview */
.gradio-container .image-container,
.gradio-container [data-testid="image"],
.image-frame {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    min-height: 280px !important;
}

.gradio-container .image-container img,
.gradio-container [data-testid="image"] img,
.image-frame img {
    max-height: 300px !important;
    width: auto !important;
    object-fit: contain !important;
    display: block !important;
}

.gradio-container .upload-container,
.gradio-container [data-testid="dropzone"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-subtle) !important;
    min-height: 200px !important;
}

.gradio-container .upload-container:hover,
.gradio-container [data-testid="dropzone"]:hover {
    border-color: #ffffff !important;
    background: rgba(255,255,255,0.02) !important;
}

/* Remove any blue focus/accent colors */
*:focus, *:focus-visible {
    outline-color: #71717a !important;
    box-shadow: none !important;
}

.gradio-container button:focus,
.gradio-container input:focus {
    outline: 2px solid #71717a !important;
    outline-offset: 2px !important;
    box-shadow: none !important;
}

/* Override Gradio blue accents */
.gradio-container {
    --color-accent: #ffffff !important;
    --color-accent-soft: rgba(255,255,255,0.1) !important;
}

[class*="blue"], [class*="primary"] {
    --tw-ring-color: #71717a !important;
}

.svelte-1f354aw, .border-orange-500, [style*="border-color: rgb(249"] {
    border-color: #3f3f46 !important;
}

/* Examples styling */
.gradio-container .examples-table {
    background: transparent !important;
}

.gradio-container .examples-table td {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    cursor: pointer;
    transition: all 0.2s;
}

.gradio-container .examples-table td:hover {
    border-color: var(--primary) !important;
}

.viz-section {
    margin-top: 40px;
}

.viz-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
    text-align: center;
    margin-bottom: 24px;
}

.disclaimer {
    background: var(--bg-muted);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px;
    margin-top: 32px;
}

.disclaimer p {
    color: var(--text-muted);
    font-size: 13px;
    margin: 0;
}

/* ===== ABOUT PAGE ===== */
.about-wrapper {
    max-width: 800px;
    margin: 0 auto;
    padding: 64px 24px;
}

.about-header {
    margin-bottom: 56px;
}

.about-header h1 {
    font-size: 40px;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 16px 0;
    letter-spacing: -0.02em;
}

.about-header p {
    font-size: 18px;
    color: var(--text-muted);
    line-height: 1.7;
}

.about-section {
    margin-bottom: 48px;
}

.about-section h2 {
    font-size: 13px;
    font-weight: 600;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 16px 0;
}

.about-section h3 {
    font-size: 24px;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 16px 0;
}

.about-section p {
    color: var(--text-muted);
    font-size: 16px;
    line-height: 1.8;
    margin: 0 0 16px 0;
}

.about-section ul {
    color: var(--text-muted);
    padding-left: 20px;
    margin: 0;
}

.about-section li {
    margin-bottom: 8px;
    line-height: 1.6;
}

.about-section strong {
    color: var(--text);
}

.tech-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 16px;
}

.tech-tag {
    background: var(--bg-muted);
    border: 1px solid var(--border-subtle);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    color: var(--text-muted);
}

.author-card {
    background: var(--bg-subtle);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 32px;
    margin-top: 24px;
}

.author-card h4 {
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 8px 0;
}

.author-card p {
    color: var(--text-muted);
    margin: 0 0 16px 0;
}

.author-links {
    display: flex;
    gap: 16px;
}

.author-links a {
    color: var(--primary);
    text-decoration: none;
    font-size: 14px;
    font-weight: 500;
}

.author-links a:hover {
    text-decoration: underline;
}

/* ===== RESPONSIVE ===== */
@media (max-width: 640px) {
    .hero { padding: 48px 16px; }
    .hero h1 { font-size: 32px; }
    .hero-stats { gap: 32px; }
    .features { padding: 48px 16px; }
    .features-grid { grid-template-columns: 1fr; }
    .diagnostic-wrapper { padding: 32px 16px; }
    .about-wrapper { padding: 32px 16px; }
}

/* ===== FORCE REMOVE ALL BLUE ===== */
* {
    --primary-50: #fafafa !important;
    --primary-100: #f5f5f5 !important;
    --primary-200: #e5e5e5 !important;
    --primary-300: #d4d4d4 !important;
    --primary-400: #a3a3a3 !important;
    --primary-500: #737373 !important;
    --primary-600: #525252 !important;
    --primary-700: #404040 !important;
    --primary-800: #262626 !important;
    --primary-900: #171717 !important;
}

.gradio-container,
.gradio-container * {
    --checkbox-background-color-selected: #525252 !important;
    --button-primary-background-fill: #fafafa !important;
    --button-primary-background-fill-hover: #e5e5e5 !important;
    --button-primary-text-color: #0a0a0a !important;
    --slider-color: #fafafa !important;
    --block-label-text-color: #a3a3a3 !important;
    --body-text-color: #fafafa !important;
    --color-accent: #fafafa !important;
    --link-text-color: #fafafa !important;
    --link-text-color-hover: #d4d4d4 !important;
}

/* Examples gallery fix */
.gallery, .grid-wrap, .thumbnail-item {
    background: var(--bg-subtle) !important;
}

.gallery img, .thumbnail-item img {
    border-radius: 6px !important;
    cursor: pointer !important;
}

/* Label styling */
.label-wrap, .output-label {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

.label-wrap .text, .confidence-bar {
    background: linear-gradient(90deg, #404040 0%, #262626 100%) !important;
}

/* Accordion styling */
.accordion {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

.accordion-header {
    background: transparent !important;
    color: var(--text) !important;
}

/* Secondary buttons */
.gradio-container button.secondary,
.gradio-container button:not(.primary) {
    background: var(--bg-muted) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

.gradio-container button.secondary:hover,
.gradio-container button:not(.primary):hover {
    background: var(--bg-subtle) !important;
    border-color: var(--text-muted) !important;
}
"""

# ============================================================
# HTML CONTENT
# ============================================================
watermark_html = """
<a href="https://orehovschi.com" target="_blank" class="watermark" title="orehovschi.com">
    <svg viewBox="200 80 520 120" fill="none"><path d="M220.228 124.394C216.295 136.784 197.467 151.923 213.286 164.067C233.494 179.581 311.081 114.354 287.264 97.9612C267.04 84.0407 230.956 101.639 227.963 116.478M275.112 144.263C303.698 122.796 289.382 147.552 285.867 155.813C283.982 160.244 311.078 154.469 314.689 153.41M314.689 153.41C359.924 140.146 311.373 127.238 314.689 153.41ZM314.689 153.41C316.788 169.975 356.198 147.569 361.872 143.865M361.872 143.865C379.996 132.033 396.654 110.1 396.142 93.0151C395.587 74.4442 371.8 126.504 361.872 143.865ZM361.872 143.865C358.713 149.388 356.853 155.059 354.151 160.646C353.56 161.868 357.043 156.178 366.562 150.218C393.466 133.372 380.946 180.567 408.42 157.542M408.42 157.542C422.113 146.068 406.519 142.186 408.42 157.542ZM408.42 157.542C409.261 164.335 419.291 164.376 427.904 164.518C446.758 164.829 436.156 128.918 413.569 144.253M443.616 141.885C461.237 189.037 461.247 141.561 494.921 144.092M530.069 142.59C492.635 142.764 525.754 153.079 519.876 160.419C508.831 167.863 482.829 171.175 468.519 176M567.777 142.645C548.176 142.644 535.687 158.692 551.095 168.622C563.172 176.406 592.423 158.35 598.782 153.314M598.782 153.314C612.006 142.84 620.551 100.564 615.577 113.834C610.591 127.137 604.922 140.225 598.782 153.314ZM598.782 153.314C586.286 179.948 593.453 164.248 617.545 156.091C635.029 150.172 619.238 171.51 629.98 172.179C645.463 173.144 652.018 163.388 658.901 155.799M660.543 135.24C663.427 135.24 666.987 132.79 669 131.565" stroke="#fafafa" stroke-width="4" stroke-linecap="round" fill="none"/></svg>
</a>
"""

home_html = """
<div class="home-wrapper">
    <div class="hero">
        <img src="https://huggingface.co/spaces/liviuorehovschi/histomancer/resolve/main/histo.ico" alt="Histomancer" style="width: 80px; height: 80px; margin-bottom: 24px;">
        <h1>Histomancer</h1>
        <p>A tool for classifying lung histopathology images. It analyzes H&E stained tissue slides and predicts whether a sample corresponds to lung adenocarcinoma, lung squamous cell carcinoma, or normal (benign) lung tissue.</p>
        <p style="font-size: 15px; margin-top: 16px;">The goal is not to automate diagnosis or replace clinical judgment. It was built to explore whether modern convolutional neural networks can learn histologically meaningful patterns and present those predictions in a way that is inspectable, explainable, and usable in real time.</p>
    </div>
    <div class="features">
        <div class="features-header">
            <h2>What the Model Does</h2>
        </div>
        <div class="features-grid">
            <div class="feature">
                <h3>Classification</h3>
                <p>Classifies lung histopathology images into: Adenocarcinoma, Squamous cell carcinoma, Normal lung tissue. Produces a single predicted class with a confidence score.</p>
            </div>
            <div class="feature">
                <h3>Explainability</h3>
                <p>Generates Grad-CAM (region level attention) and saliency maps (pixel level sensitivity). These visualizations often align with known histopathological markers.</p>
            </div>
            <div class="feature">
                <h3>Transparency</h3>
                <p>Most academic models remain locked in papers or notebooks, and many commercial tools are proprietary and opaque. Histomancer exists in between: public, interactive, and transparent.</p>
            </div>
        </div>
    </div>
</div>
"""

about_html = """
<div class="about-wrapper">
    <div class="about-header">
        <h1>About Histomancer</h1>
    </div>

    <div class="about-section">
        <h2>What Histomancer Is</h2>
        <p>Histomancer is a tool for classifying lung histopathology images. It analyzes H&E stained tissue slides and predicts whether a sample corresponds to lung adenocarcinoma, lung squamous cell carcinoma, or normal (benign) lung tissue.</p>
        <p>The goal of Histomancer is not to automate diagnosis or replace clinical judgment. It was built to explore whether modern convolutional neural networks can learn histologically meaningful patterns and present those predictions in a way that is inspectable, explainable, and usable in real time.</p>
        <p>Most academic models remain locked in papers or notebooks, and many commercial tools are proprietary and opaque. Histomancer exists in between: public, interactive, and transparent.</p>
    </div>

    <div class="about-section">
        <h2>Why Histomancer Was Made</h2>
        <p>Lung cancer diagnosis ultimately depends on histopathological examination, a process that is time intensive and subject to variability, especially as biopsy volume increases. At the same time, deep learning models have shown strong performance on histology images but often without interpretability or public accessibility.</p>
        <p>Histomancer was created to answer three questions:</p>
        <ul>
            <li>Can a modern CNN reliably distinguish major lung cancer subtypes from histology images?</li>
            <li>Does the model's attention align with known pathological features?</li>
            <li>Can this be explored through a simple, public interface rather than a closed research pipeline?</li>
        </ul>
        <p>The project prioritizes clarity and explainability over raw automation.</p>
    </div>

    <div class="about-section">
        <h2>About Me and My Role</h2>
        <p>I built Histomancer end to end: model development, training, evaluation, deployment, and frontend integration.</p>
        <p>My background spans machine learning, natural sciences, and software engineering, and this project reflects my interest in how these systems behave when exposed to real biological structure rather than abstract benchmarks.</p>
        <p>Rather than optimizing only for accuracy, I focused on: dataset composition and generalizability, interpretability through visual explanations, real world deployment constraints, and responsible framing in medical contexts.</p>
    </div>

    <div class="about-section">
        <h2>What the Model Does (Capabilities)</h2>
        <p>Classifies lung histopathology images into: Adenocarcinoma, Squamous cell carcinoma, Normal lung tissue.</p>
        <p>Produces a single predicted class with a confidence score.</p>
        <p>Generates explainability visualizations: Grad-CAM (region level attention), Saliency maps (pixel level sensitivity).</p>
        <p>In many cases, these visualizations align with known histopathological markers: glandular structures in adenocarcinoma, keratinization and keratin pearls in squamous carcinoma, diffuse or low intensity attention in benign tissue.</p>
    </div>

    <div class="about-section">
        <h2>Technical Overview</h2>
        <p>Model: EfficientNet B0 (convolutional neural network)</p>
        <p>Framework: TensorFlow / Keras</p>
        <p>Input: Histopathology images (resized and normalized to match training conditions)</p>
        <p>Output: Class prediction + confidence score</p>
        <p>Explainability: Grad-CAM and saliency mapping computed on demand</p>
        <p>The model was trained using transfer learning and fine tuned for lung histology classification.</p>
    </div>

    <div class="about-section">
        <h2>Datasets Used (With Important Caveats)</h2>
        <p>Histomancer was trained using a combined dataset approach, which is essential to understand its strengths and limitations.</p>
        <p><strong>LC25000:</strong> Large histopathology dataset (~25,000 images). Derived from a much smaller number of real samples. Expanded via heavy synthetic augmentation (rotations, flips, contrast changes, zooms, noise). Balanced across classes. This dataset provides scale and stability, but its synthetic nature makes generalization tricky. <a href="https://doi.org/10.57702/cfrwm2gi" target="_blank">Dataset link</a></p>
        <p><strong>LungHist700:</strong> Smaller dataset (691 images). Real clinical histopathology samples. Higher resolution and natural variability. Includes adenocarcinoma, squamous carcinoma, and normal lung tissue. This dataset was added specifically to counterbalance the synthetic nature of LC25000 and improve real world relevance. <a href="https://doi.org/10.1038/s41597-024-03944-3" target="_blank">Dataset link</a></p>
    </div>

    <div class="about-section">
        <h2>Important Limitations and Shortcomings</h2>
        <p>Some limitations are fundamental and intentionally disclosed:</p>
        <ul>
            <li>A significant portion of training data is synthetically augmented</li>
            <li>Augmentation cannot fully replicate: inter patient variability, differences in staining protocols, differences in slide preparation or imaging equipment</li>
            <li>Confidence scores are derived from softmax probabilities and are not calibrated uncertainty estimates</li>
            <li>The model has not been externally validated on datasets from multiple hospitals</li>
        </ul>
        <p>Because of this, Histomancer should be viewed as: a research and demonstration tool, a learning aid, an exploration of explainable approaches in pathology. Not as a clinical diagnostic system.</p>
    </div>

    <div class="about-section">
        <h2>Responsible Use and Framing</h2>
        <p>Histomancer is designed to support exploration and understanding, not to replace expert interpretation.</p>
        <p>Predictions should always be interpreted cautiously, especially in borderline cases where histological features overlap. Any real clinical application would require additional validation, uncertainty estimation, and regulatory review.</p>
    </div>

    <div class="about-section">
        <div class="author-card">
            <h4>Liviu Orehovschi</h4>
            <div class="author-links">
                <a href="https://orehovschi.com" target="_blank">Portfolio</a>
                <a href="https://github.com/liviuorehovschi" target="_blank">GitHub</a>
                <a href="https://linkedin.com/in/liviuorehovschi" target="_blank">LinkedIn</a>
            </div>
        </div>
    </div>
</div>
"""

# ============================================================
# BUILD GRADIO APP
# ============================================================
with gr.Blocks(
    title="Histomancer",
    theme=gr.themes.Base(),
    css=custom_css,
    head='<link rel="icon" href="https://huggingface.co/spaces/liviuorehovschi/histomancer/resolve/main/histo.ico">'
) as demo:

    gr.HTML(watermark_html)

    with gr.Tabs():

        # HOME
        with gr.Tab("Home"):
            gr.HTML(home_html)

        # DIAGNOSTIC TOOL
        with gr.Tab("Diagnostic Tool"):
            gr.HTML("""
            <div class="diagnostic-wrapper">
                <div class="diagnostic-header">
                    <h1>Analyze Tissue</h1>
                    <p>Upload a histopathology image or try one of the samples below</p>
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        type="pil",
                        label="Upload Image",
                        height=300,
                        sources=["upload", "clipboard"],
                        show_download_button=False,
                        show_share_button=False,
                        interactive=True
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

                    gr.Markdown("**Try a sample:**")
                    gr.Examples(
                        examples=[
                            ["test_images/adenocarcinoma.jpg"],
                            ["test_images/benign_tissue.png"],
                            ["test_images/squamous_cell_carcinoma.png"]
                        ],
                        inputs=input_image,
                        label=""
                    )

                with gr.Column(scale=1):
                    output_label = gr.Label(label="Classification", num_top_classes=3)

            with gr.Accordion("Explainability Visualizations", open=False):
                with gr.Row():
                    gradcam_btn = gr.Button("Generate Grad-CAM")
                    saliency_btn = gr.Button("Generate Saliency Map")
                with gr.Row():
                    gradcam_output = gr.Image(type="pil", label="Grad-CAM", height=280)
                    saliency_output = gr.Image(type="pil", label="Saliency Map", height=280)

            gr.HTML("""
            <div class="disclaimer">
                <p><strong>Disclaimer:</strong> This is for research and exploration, not clinical diagnosis. Consult medical professionals for healthcare decisions.</p>
            </div>
            """)

            analyze_btn.click(fn=analyze_image, inputs=input_image, outputs=output_label)
            gradcam_btn.click(fn=get_gradcam, inputs=input_image, outputs=gradcam_output)
            saliency_btn.click(fn=get_saliency, inputs=input_image, outputs=saliency_output)

        # ABOUT
        with gr.Tab("About"):
            gr.HTML(about_html)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
