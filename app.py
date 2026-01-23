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

print("[INFO] Loading model at startup...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
print("[SUCCESS] Model loaded once and ready!")


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess image for EfficientNet."""
    img = img.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


def predict(image: Image.Image) -> dict:
    """Classify lung histopathology image."""
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
    """Generate Grad-CAM visualization."""
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
    """Generate saliency map visualization."""
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
    """Main analysis function."""
    if image is None:
        return {}, None, None
    try:
        if not isinstance(image, Image.Image):
            return {}, None, None
        if image.size[0] < 50 or image.size[1] < 50:
            return {}, None, None
        prediction = predict(image)
        saliency = generate_saliency(image)
        gradcam = generate_gradcam(image)
        return prediction, saliency, gradcam
    except Exception as e:
        print(f"[ERROR] Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, None, None


# ============================================================
# CUSTOM CSS - COMPLETE REDESIGN
# ============================================================
custom_css = """
/* ===== BOOT SCREEN ANIMATION (Pure CSS - no JS needed) ===== */
.sig-overlay {
    position: fixed;
    inset: 0;
    z-index: 99999;
    background-color: #020617;
    display: flex;
    align-items: center;
    justify-content: center;
    /* Auto fade out after animation completes */
    animation: bootFadeOut 0.8s ease-out 2.2s forwards;
}
@keyframes bootFadeOut {
    to {
        opacity: 0;
        visibility: hidden;
        pointer-events: none;
    }
}
.sig-stroke {
    fill: none;
    stroke: white;
    stroke-width: 5;
    stroke-linecap: round;
    stroke-dasharray: 500;
    stroke-dashoffset: 500;
    animation: drawStroke 0.07s ease-out forwards;
}
.sig-stroke.s1 { animation-delay: 0.1s; }
.sig-stroke.s2 { animation-delay: 0.17s; }
.sig-stroke.s3 { animation-delay: 0.26s; }
.sig-stroke.s4 { animation-delay: 0.33s; }
.sig-stroke.s5 { animation-delay: 0.42s; }
.sig-stroke.s6 { animation-delay: 0.49s; }
.sig-stroke.s7 { animation-delay: 0.58s; }
.sig-stroke.s8 { animation-delay: 0.65s; }
.sig-stroke.s9 { animation-delay: 0.74s; }
.sig-stroke.s10 { animation-delay: 0.83s; }
.sig-stroke.s11 { animation-delay: 0.92s; }
.sig-stroke.s12 { animation-delay: 0.99s; }
.sig-stroke.s13 { animation-delay: 1.06s; }
.sig-stroke.s14 { animation-delay: 1.13s; }
@keyframes drawStroke {
    to { stroke-dashoffset: 0; }
}

/* ===== SIGNATURE WATERMARK ===== */
.signature-watermark {
    position: fixed;
    bottom: 16px;
    right: 16px;
    z-index: 1000;
    opacity: 0.12;
    transition: opacity 0.3s ease;
    pointer-events: auto;
    cursor: pointer;
}
.signature-watermark:hover {
    opacity: 0.35;
}
.signature-watermark svg {
    width: 100px;
    height: auto;
}

/* ===== GLOBAL STYLES ===== */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #1a1a24;
    --bg-card-hover: #22222e;
    --text-primary: #f0f0f5;
    --text-secondary: #a0a0b0;
    --text-muted: #6a6a7a;
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.3);
    --border: rgba(255, 255, 255, 0.08);
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
}

body, .gradio-container {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* Hide default gradio footer */
footer { display: none !important; }

/* ===== NAVIGATION TABS ===== */
.tabs {
    background: transparent !important;
    border: none !important;
}

.tab-nav {
    background: var(--bg-secondary) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 2rem !important;
    gap: 0 !important;
    justify-content: center !important;
}

.tab-nav button {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    padding: 1rem 2rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
}

.tab-nav button:hover {
    color: var(--text-primary) !important;
    background: rgba(255,255,255,0.03) !important;
}

.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}

.tabitem {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ===== HERO SECTION (HOME) ===== */
.hero-section {
    min-height: 85vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 4rem 2rem;
    background: radial-gradient(ellipse at top, rgba(99, 102, 241, 0.08) 0%, transparent 60%);
}

.hero-logo {
    width: 120px;
    height: 120px;
    margin-bottom: 2rem;
    border-radius: 24px;
    box-shadow: 0 0 60px var(--accent-glow);
}

.hero-title {
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -2px;
    margin: 0 0 1rem 0;
    background: linear-gradient(135deg, #fff 0%, #a0a0b0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.4rem;
    color: var(--text-secondary);
    margin: 0 0 3rem 0;
    max-width: 600px;
    line-height: 1.6;
}

.hero-stats {
    display: flex;
    gap: 3rem;
    margin-bottom: 3rem;
}

.stat-item {
    text-align: center;
}

.stat-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--accent);
}

.stat-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.hero-cta {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem 2.5rem;
    background: var(--accent);
    color: white;
    font-weight: 600;
    font-size: 1rem;
    border-radius: 12px;
    text-decoration: none;
    transition: all 0.3s ease;
    cursor: pointer;
    border: none;
}

.hero-cta:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 40px var(--accent-glow);
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    max-width: 1000px;
    margin: 4rem auto 0;
    padding: 0 2rem;
}

.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: left;
    transition: all 0.3s ease;
}

.feature-card:hover {
    background: var(--bg-card-hover);
    transform: translateY(-4px);
    border-color: rgba(99, 102, 241, 0.3);
}

.feature-icon {
    font-size: 2rem;
    margin-bottom: 1rem;
}

.feature-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: var(--text-primary);
}

.feature-desc {
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ===== DIAGNOSTIC TOOL PAGE ===== */
.diagnostic-container {
    padding: 3rem 2rem;
    max-width: 1200px;
    margin: 0 auto;
}

.diagnostic-header {
    text-align: center;
    margin-bottom: 3rem;
}

.diagnostic-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.diagnostic-subtitle {
    color: var(--text-secondary);
}

/* Upload area */
.upload-zone {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}

.upload-zone:hover {
    border-color: var(--accent) !important;
    background: var(--bg-card-hover) !important;
}

/* Results styling */
.results-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
}

/* Button overrides */
button.primary, button[variant="primary"] {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.875rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

button.primary:hover, button[variant="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px var(--accent-glow) !important;
}

/* Image component styling */
.image-container {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Label component */
.label-container .label {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ===== ABOUT PAGE ===== */
.about-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 4rem 2rem;
}

.about-header {
    text-align: center;
    margin-bottom: 4rem;
}

.about-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.about-intro {
    font-size: 1.2rem;
    color: var(--text-secondary);
    line-height: 1.7;
}

.about-section {
    margin-bottom: 4rem;
}

.about-section-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid var(--accent);
    display: inline-block;
}

.about-text {
    color: var(--text-secondary);
    line-height: 1.8;
    font-size: 1.05rem;
}

.about-text strong {
    color: var(--text-primary);
}

.tech-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 1.5rem;
}

.tech-badge {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 0.9rem;
    color: var(--text-secondary);
}

.creator-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    display: flex;
    gap: 2rem;
    align-items: center;
}

.creator-info h3 {
    font-size: 1.3rem;
    margin-bottom: 0.5rem;
}

.creator-info p {
    color: var(--text-secondary);
    margin-bottom: 1rem;
}

.creator-links {
    display: flex;
    gap: 1rem;
}

.creator-link {
    color: var(--accent);
    text-decoration: none;
    font-size: 0.9rem;
}

.creator-link:hover {
    text-decoration: underline;
}

/* ===== RESPONSIVE ===== */
@media (max-width: 768px) {
    .hero-title { font-size: 2.5rem; }
    .hero-subtitle { font-size: 1.1rem; }
    .features-grid { grid-template-columns: 1fr; }
    .hero-stats { flex-direction: column; gap: 1.5rem; }
    .creator-card { flex-direction: column; text-align: center; }
}

/* ===== MARKDOWN OVERRIDES ===== */
.prose h1, .prose h2, .prose h3 {
    color: var(--text-primary) !important;
}

.prose p, .prose li {
    color: var(--text-secondary) !important;
}

.prose a {
    color: var(--accent) !important;
}

/* Hide some gradio defaults */
.gradio-container > .prose:first-child {
    display: none;
}
"""

# ============================================================
# BOOT SCREEN HTML (Pure CSS animation - no JS needed)
# ============================================================
boot_screen_html = """
<div class="sig-overlay" id="bootScreen">
    <svg viewBox="200 80 520 120" style="width: 80vw; max-width: 500px; height: auto;" fill="none">
        <path class="sig-stroke s1" d="M220.228 124.394C216.295 136.784 197.467 151.923 213.286 164.067C233.494 179.581 311.081 114.354 287.264 97.9612C267.04 84.0407 230.956 101.639 227.963 116.478" />
        <path class="sig-stroke s2" d="M275.112 144.263C303.698 122.796 289.382 147.552 285.867 155.813C283.982 160.244 311.078 154.469 314.689 153.41" />
        <path class="sig-stroke s3" d="M314.689 153.41C359.924 140.146 311.373 127.238 314.689 153.41Z" />
        <path class="sig-stroke s4" d="M314.689 153.41C316.788 169.975 356.198 147.569 361.872 143.865" />
        <path class="sig-stroke s5" d="M361.872 143.865C379.996 132.033 396.654 110.1 396.142 93.0151C395.587 74.4442 371.8 126.504 361.872 143.865Z" />
        <path class="sig-stroke s6" d="M361.872 143.865C358.713 149.388 356.853 155.059 354.151 160.646C353.56 161.868 357.043 156.178 366.562 150.218C393.466 133.372 380.946 180.567 408.42 157.542" />
        <path class="sig-stroke s7" d="M408.42 157.542C422.113 146.068 406.519 142.186 408.42 157.542Z" />
        <path class="sig-stroke s8" d="M408.42 157.542C409.261 164.335 419.291 164.376 427.904 164.518C446.758 164.829 436.156 128.918 413.569 144.253" />
        <path class="sig-stroke s9" d="M443.616 141.885C461.237 189.037 461.247 141.561 494.921 144.092" />
        <path class="sig-stroke s10" d="M530.069 142.59C492.635 142.764 525.754 153.079 519.876 160.419C508.831 167.863 482.829 171.175 468.519 176" />
        <path class="sig-stroke s11" d="M567.777 142.645C548.176 142.644 535.687 158.692 551.095 168.622C563.172 176.406 592.423 158.35 598.782 153.314" />
        <path class="sig-stroke s12" d="M598.782 153.314C612.006 142.84 620.551 100.564 615.577 113.834C610.591 127.137 604.922 140.225 598.782 153.314Z" />
        <path class="sig-stroke s13" d="M598.782 153.314C586.286 179.948 593.453 164.248 617.545 156.091C635.029 150.172 619.238 171.51 629.98 172.179C645.463 173.144 652.018 163.388 658.901 155.799" />
        <path class="sig-stroke s14" d="M660.543 135.24C663.427 135.24 666.987 132.79 669 131.565" />
    </svg>
</div>
"""

# ============================================================
# SIGNATURE WATERMARK HTML
# ============================================================
watermark_html = """
<a href="https://orehovschi.com" target="_blank" class="signature-watermark" title="Made by Liviu Orehovschi">
    <svg viewBox="200 80 520 120" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M220.228 124.394C216.295 136.784 197.467 151.923 213.286 164.067C233.494 179.581 311.081 114.354 287.264 97.9612C267.04 84.0407 230.956 101.639 227.963 116.478M275.112 144.263C303.698 122.796 289.382 147.552 285.867 155.813C283.982 160.244 311.078 154.469 314.689 153.41M314.689 153.41C359.924 140.146 311.373 127.238 314.689 153.41ZM314.689 153.41C316.788 169.975 356.198 147.569 361.872 143.865M361.872 143.865C379.996 132.033 396.654 110.1 396.142 93.0151C395.587 74.4442 371.8 126.504 361.872 143.865ZM361.872 143.865C358.713 149.388 356.853 155.059 354.151 160.646C353.56 161.868 357.043 156.178 366.562 150.218C393.466 133.372 380.946 180.567 408.42 157.542M408.42 157.542C422.113 146.068 406.519 142.186 408.42 157.542ZM408.42 157.542C409.261 164.335 419.291 164.376 427.904 164.518C446.758 164.829 436.156 128.918 413.569 144.253M443.616 141.885C461.237 189.037 461.247 141.561 494.921 144.092M530.069 142.59C492.635 142.764 525.754 153.079 519.876 160.419C508.831 167.863 482.829 171.175 468.519 176M567.777 142.645C548.176 142.644 535.687 158.692 551.095 168.622C563.172 176.406 592.423 158.35 598.782 153.314M598.782 153.314C612.006 142.84 620.551 100.564 615.577 113.834C610.591 127.137 604.922 140.225 598.782 153.314ZM598.782 153.314C586.286 179.948 593.453 164.248 617.545 156.091C635.029 150.172 619.238 171.51 629.98 172.179C645.463 173.144 652.018 163.388 658.901 155.799M660.543 135.24C663.427 135.24 666.987 132.79 669 131.565" stroke="white" stroke-width="5" stroke-linecap="round" fill="none"/>
    </svg>
</a>
"""

# ============================================================
# PAGE CONTENT
# ============================================================

home_content = """
<div class="hero-section">
    <img src="https://huggingface.co/spaces/liviuorehovschi/histomancer/resolve/main/histo.ico" alt="Histomancer Logo" class="hero-logo" onerror="this.style.display='none'">
    <h1 class="hero-title">Histomancer</h1>
    <p class="hero-subtitle">AI-powered lung cancer classification from histopathology images. Built with deep learning, designed for transparency.</p>

    <div class="hero-stats">
        <div class="stat-item">
            <div class="stat-value">98.25%</div>
            <div class="stat-label">Test Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">25K+</div>
            <div class="stat-label">Training Images</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">3</div>
            <div class="stat-label">Cancer Types</div>
        </div>
    </div>

    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">Histopathology Analysis</div>
            <div class="feature-desc">Classifies lung tissue into adenocarcinoma, squamous cell carcinoma, or normal tissue from microscopy images.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Explainable AI</div>
            <div class="feature-desc">Grad-CAM and saliency maps show exactly which regions influence the AI's decision, building trust and transparency.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-Time Results</div>
            <div class="feature-desc">Get instant classification with confidence scores. Powered by EfficientNetB0 with transfer learning.</div>
        </div>
    </div>
</div>
"""

about_content = """
<div class="about-container">
    <div class="about-header">
        <h1 class="about-title">About Histomancer</h1>
        <p class="about-intro">A deep learning tool for lung cancer histopathology classification, built as a capstone research project exploring the intersection of AI and medical diagnostics.</p>
    </div>

    <div class="about-section">
        <h2 class="about-section-title">The Problem</h2>
        <p class="about-text">
            Lung cancer remains the leading cause of cancer-related mortality worldwide, with <strong>2.2 million new cases and 1.8 million deaths in 2020</strong>. While imaging can identify suspicious nodules, definitive diagnosis requires histopathological examination by pathologists who analyze tissue slides to classify cancer subtypes.
            <br><br>
            This process is <strong>time-consuming and prone to inter-observer variability</strong>. The increasing number of biopsies due to expanded screening programs has increased workload for pathologists, leading to diagnostic delays and fatigue-related errors.
        </p>
    </div>

    <div class="about-section">
        <h2 class="about-section-title">The Solution</h2>
        <p class="about-text">
            Histomancer leverages <strong>Convolutional Neural Networks (CNNs)</strong> trained on histopathological images to improve consistency and efficiency in tumor classification. The model distinguishes between:
            <br><br>
            <strong>• Lung Adenocarcinoma</strong> — Characterized by glandular structures and mucin production<br>
            <strong>• Lung Squamous Cell Carcinoma</strong> — Identified by keratinization and intercellular bridges<br>
            <strong>• Normal Lung Tissue</strong> — Organized alveoli with thin epithelial cells
            <br><br>
            To address the "black box" problem in AI, the tool employs <strong>Grad-CAM and saliency mapping</strong> to visualize which histological regions influence predictions, aligning AI insights with pathologist reasoning.
        </p>
    </div>

    <div class="about-section">
        <h2 class="about-section-title">Technical Approach</h2>
        <p class="about-text">
            The model uses <strong>EfficientNetB0</strong>, chosen for its compound scaling method that balances depth, width, and resolution. Training followed a three-phase progressive fine-tuning approach:
            <br><br>
            <strong>Phase 1:</strong> Trained classification head with frozen backbone (lr: 1e-4)<br>
            <strong>Phase 2:</strong> Unfroze last 50 layers for domain-specific learning (lr: 1e-5)<br>
            <strong>Phase 3:</strong> Final fine-tuning with last 20 layers (lr: 5e-6)
            <br><br>
            The model was trained on a combined dataset of <strong>LC25000</strong> (25,000 augmented images) and <strong>LungHist700</strong> (691 real clinical samples from Hospital Clínico Universitario de Valladolid), achieving <strong>98.25% test accuracy</strong>.
        </p>
        <div class="tech-stack">
            <span class="tech-badge">TensorFlow</span>
            <span class="tech-badge">Keras</span>
            <span class="tech-badge">EfficientNetB0</span>
            <span class="tech-badge">Gradio</span>
            <span class="tech-badge">OpenCV</span>
            <span class="tech-badge">NumPy</span>
            <span class="tech-badge">HuggingFace Spaces</span>
        </div>
    </div>

    <div class="about-section">
        <h2 class="about-section-title">Limitations & Ethics</h2>
        <p class="about-text">
            <strong>This is an educational demonstration tool, NOT for clinical diagnosis.</strong>
            <br><br>
            While the model achieved excellent results, several limitations must be acknowledged:
            <br><br>
            • <strong>Data Bias:</strong> LC25000 relies on synthetic augmentation which may not capture real-world variability<br>
            • <strong>Domain Shift:</strong> Performance on images from different hospitals/staining protocols is untested<br>
            • <strong>Uncertainty:</strong> The model outputs confident predictions even in ambiguous cases<br>
            • <strong>External Validation:</strong> The model needs testing on independent datasets before clinical consideration
        </p>
    </div>

    <div class="about-section">
        <h2 class="about-section-title">Created By</h2>
        <div class="creator-card">
            <div class="creator-info">
                <h3>Liviu Orehovschi</h3>
                <p>Student researcher exploring the intersection of deep learning and medical imaging. This project was developed as a capstone research project.</p>
                <div class="creator-links">
                    <a href="https://orehovschi.com" target="_blank" class="creator-link">Portfolio →</a>
                    <a href="https://github.com/liviuorehovschi" target="_blank" class="creator-link">GitHub →</a>
                    <a href="https://linkedin.com/in/liviuorehovschi" target="_blank" class="creator-link">LinkedIn →</a>
                </div>
            </div>
        </div>
    </div>
</div>
"""

# ============================================================
# BUILD THE GRADIO APP
# ============================================================
with gr.Blocks(
    title="Histomancer - AI Lung Cancer Classifier",
    theme=gr.themes.Base(),
    css=custom_css,
    head="""
    <link rel="icon" type="image/x-icon" href="https://huggingface.co/spaces/liviuorehovschi/histomancer/resolve/main/histo.ico">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    """
) as demo:

    # Boot screen
    gr.HTML(boot_screen_html)

    # Signature watermark
    gr.HTML(watermark_html)

    # Main navigation tabs
    with gr.Tabs() as tabs:

        # ===== HOME TAB =====
        with gr.Tab("Home", id="home"):
            gr.HTML(home_content)

        # ===== DIAGNOSTIC TOOL TAB =====
        with gr.Tab("Diagnostic Tool", id="diagnostic"):
            gr.HTML("""
            <div class="diagnostic-container">
                <div class="diagnostic-header">
                    <h1 class="diagnostic-title">Lung Tissue Analyzer</h1>
                    <p class="diagnostic-subtitle">Upload a histopathology image for AI-powered classification</p>
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        type="pil",
                        label="Upload Histopathology Image",
                        height=350,
                        elem_classes=["upload-zone"]
                    )
                    analyze_btn = gr.Button(
                        "Analyze Image",
                        variant="primary",
                        size="lg"
                    )
                    gr.Markdown("""
                    **Supported formats:** JPG, PNG, TIFF
                    **Best results:** H&E stained lung tissue at 20-40x magnification
                    """)

                with gr.Column(scale=1):
                    output_label = gr.Label(
                        label="Classification Results",
                        num_top_classes=3
                    )
                    gr.Markdown("""
                    **Classes:**
                    - 🔴 **Adenocarcinoma** — Glandular lung cancer
                    - 🟡 **Squamous Cell Carcinoma** — Keratinizing lung cancer
                    - 🟢 **Normal Tissue** — Healthy lung cells
                    """)

            gr.HTML("<h3 style='text-align:center; margin: 2rem 0 1rem; color: var(--text-primary);'>AI Explainability Visualizations</h3>")

            with gr.Row():
                saliency_output = gr.Image(
                    type="pil",
                    label="Saliency Map",
                    height=300
                )
                gradcam_output = gr.Image(
                    type="pil",
                    label="Grad-CAM Heatmap",
                    height=300
                )

            gr.Markdown("""
            ---
            **Understanding the Visualizations:**
            - **Saliency Map:** Shows pixel-level feature importance. Bright areas indicate regions that strongly influenced the classification.
            - **Grad-CAM:** Highlights which regions the model focused on. Red/yellow = high attention, blue = low attention.

            ⚠️ **Disclaimer:** This is an educational tool. NOT for clinical diagnostic use.
            """)

            # Event handlers
            analyze_btn.click(
                fn=analyze_image,
                inputs=input_image,
                outputs=[output_label, saliency_output, gradcam_output]
            )
            input_image.change(
                fn=analyze_image,
                inputs=input_image,
                outputs=[output_label, saliency_output, gradcam_output]
            )

        # ===== ABOUT TAB =====
        with gr.Tab("About", id="about"):
            gr.HTML(about_content)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
