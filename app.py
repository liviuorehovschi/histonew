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

print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
print("[SUCCESS] Model loaded!")


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


# ============================================================
# WRAPPER FUNCTIONS
# ============================================================
def analyze_image(image):
    if image is None:
        return {}, gr.update(visible=False), gr.update(visible=False)
    try:
        results = predict(image)
        return results, gr.update(visible=True), gr.update(visible=True)
    except Exception as e:
        print(f"[ERROR] {e}")
        return {}, gr.update(visible=False), gr.update(visible=False)

def get_gradcam(image):
    if image is None:
        return None
    try:
        return generate_gradcam(image)
    except:
        return None

def get_saliency(image):
    if image is None:
        return None
    try:
        return generate_saliency(image)
    except:
        return None


# ============================================================
# CSS
# ============================================================
css = """
* { box-sizing: border-box; }
body, .gradio-container {
    background: #09090b !important;
    color: #fafafa !important;
    font-family: system-ui, -apple-system, sans-serif !important;
}
footer { display: none !important; }

/* Tabs */
.tab-nav, [role="tablist"] {
    background: #18181b !important;
    border-bottom: 1px solid #27272a !important;
    justify-content: center !important;
    gap: 0 !important;
    padding: 0 !important;
}
.tab-nav button, [role="tab"] {
    background: transparent !important;
    color: #a1a1aa !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 16px 32px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
.tab-nav button:hover, [role="tab"]:hover { color: #fafafa !important; }
.tab-nav button.selected, [role="tab"][aria-selected="true"] {
    color: #fafafa !important;
    border-bottom-color: #fafafa !important;
}

/* Cards */
.gr-panel, .gr-box, .gr-form {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 8px !important;
}

/* Buttons */
.gr-button-primary, button.primary {
    background: #fafafa !important;
    color: #09090b !important;
    border: none !important;
    font-weight: 500 !important;
}
.gr-button-primary:hover, button.primary:hover {
    background: #e4e4e7 !important;
}
.gr-button-secondary, button.secondary, button:not(.primary) {
    background: #27272a !important;
    color: #fafafa !important;
    border: 1px solid #3f3f46 !important;
}
.gr-button-secondary:hover, button.secondary:hover {
    background: #3f3f46 !important;
}

/* Image component */
.image-container, [data-testid="image"] {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 8px !important;
}
.upload-button { background: #27272a !important; }

/* Labels */
.gr-label, .label-wrap {
    background: #18181b !important;
    border: 1px solid #27272a !important;
}

/* Remove blue accents */
:root {
    --color-accent: #fafafa !important;
    --color-accent-soft: #27272a !important;
}
*:focus { outline-color: #52525b !important; }

/* Watermark - fixed position, always visible */
.watermark {
    position: fixed;
    bottom: 20px;
    right: 20px;
    opacity: 0.15;
    transition: opacity 0.3s;
    z-index: 9999;
}
.watermark:hover { opacity: 0.4; }

/* Page layouts */
.page-container { max-width: 1000px; margin: 0 auto; padding: 48px 24px; }
.page-title { font-size: 32px; font-weight: 600; margin-bottom: 8px; }
.page-subtitle { color: #a1a1aa; font-size: 16px; margin-bottom: 32px; }
.section { margin-bottom: 32px; }
.section-title { font-size: 14px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px; }

/* Hero */
.hero { text-align: center; padding: 64px 24px; }
.hero img { width: 64px; height: 64px; margin-bottom: 24px; }
.hero h1 { font-size: 48px; font-weight: 700; margin: 0 0 16px 0; }
.hero p { color: #a1a1aa; font-size: 18px; max-width: 600px; margin: 0 auto 32px; line-height: 1.7; }

/* Features */
.features { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; padding: 48px 24px; max-width: 1000px; margin: 0 auto; }
.feature { background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 24px; }
.feature h3 { font-size: 18px; font-weight: 600; margin: 0 0 8px 0; }
.feature p { color: #a1a1aa; font-size: 14px; margin: 0; line-height: 1.6; }

/* About */
.about { max-width: 700px; margin: 0 auto; padding: 48px 24px; }
.about h1 { font-size: 36px; font-weight: 700; margin-bottom: 32px; }
.about h2 { font-size: 13px; color: #fafafa; text-transform: uppercase; letter-spacing: 0.1em; margin: 32px 0 12px 0; }
.about p { color: #a1a1aa; line-height: 1.8; margin-bottom: 16px; }
.about ul { color: #a1a1aa; padding-left: 20px; }
.about li { margin-bottom: 8px; }
.about a { color: #fafafa; }

/* Author */
.author { background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 24px; margin-top: 32px; }
.author h4 { font-size: 18px; margin: 0 0 12px 0; }
.author-links { display: flex; gap: 16px; }
.author-links a { color: #a1a1aa; font-size: 14px; text-decoration: none; }
.author-links a:hover { color: #fafafa; }

/* Viz section */
.viz-row { display: flex; gap: 16px; justify-content: center; margin-top: 16px; }
.viz-output { text-align: center; }
.viz-output img { max-width: 300px; border-radius: 8px; }
"""

# ============================================================
# HTML
# ============================================================
watermark = '''
<a href="https://orehovschi.com" target="_blank" class="watermark">
<svg viewBox="200 80 520 120" width="80" fill="none"><path d="M220.228 124.394C216.295 136.784 197.467 151.923 213.286 164.067C233.494 179.581 311.081 114.354 287.264 97.9612C267.04 84.0407 230.956 101.639 227.963 116.478M275.112 144.263C303.698 122.796 289.382 147.552 285.867 155.813C283.982 160.244 311.078 154.469 314.689 153.41M314.689 153.41C359.924 140.146 311.373 127.238 314.689 153.41ZM314.689 153.41C316.788 169.975 356.198 147.569 361.872 143.865M361.872 143.865C379.996 132.033 396.654 110.1 396.142 93.0151C395.587 74.4442 371.8 126.504 361.872 143.865ZM361.872 143.865C358.713 149.388 356.853 155.059 354.151 160.646C353.56 161.868 357.043 156.178 366.562 150.218C393.466 133.372 380.946 180.567 408.42 157.542M408.42 157.542C422.113 146.068 406.519 142.186 408.42 157.542ZM408.42 157.542C409.261 164.335 419.291 164.376 427.904 164.518C446.758 164.829 436.156 128.918 413.569 144.253M443.616 141.885C461.237 189.037 461.247 141.561 494.921 144.092M530.069 142.59C492.635 142.764 525.754 153.079 519.876 160.419C508.831 167.863 482.829 171.175 468.519 176M567.777 142.645C548.176 142.644 535.687 158.692 551.095 168.622C563.172 176.406 592.423 158.35 598.782 153.314M598.782 153.314C612.006 142.84 620.551 100.564 615.577 113.834C610.591 127.137 604.922 140.225 598.782 153.314ZM598.782 153.314C586.286 179.948 593.453 164.248 617.545 156.091C635.029 150.172 619.238 171.51 629.98 172.179C645.463 173.144 652.018 163.388 658.901 155.799M660.543 135.24C663.427 135.24 666.987 132.79 669 131.565" stroke="#fafafa" stroke-width="4" stroke-linecap="round"/></svg>
</a>
'''

home_html = '''
<div class="hero">
    <img src="https://huggingface.co/spaces/liviuorehovschi/histomancer/resolve/main/histo.ico" alt="Histomancer">
    <h1>Histomancer</h1>
    <p>Lung tissue classification from histopathology slides. Analyzes H&E stained tissue and predicts adenocarcinoma, squamous cell carcinoma, or normal lung tissue.</p>
</div>
<div class="features">
    <div class="feature">
        <h3>Classification</h3>
        <p>Classifies lung histopathology into three categories with confidence scores using EfficientNet B0.</p>
    </div>
    <div class="feature">
        <h3>Explainability</h3>
        <p>Generates Grad CAM and saliency maps to visualize which regions influenced the prediction.</p>
    </div>
    <div class="feature">
        <h3>Transparency</h3>
        <p>Open source, interactive, and designed for research and education rather than clinical use.</p>
    </div>
</div>
'''

about_html = '''
<div class="about">
    <h1>About Histomancer</h1>

    <h2>What It Is</h2>
    <p>Histomancer classifies lung histopathology images into adenocarcinoma, squamous cell carcinoma, or normal tissue. It was built to explore whether CNNs can learn histologically meaningful patterns and present predictions in an inspectable, explainable way.</p>

    <h2>Technical Details</h2>
    <p>Model: EfficientNet B0 with transfer learning. Input: 224x224 histopathology images. Output: Class prediction with confidence score. Explainability: Grad CAM and saliency mapping.</p>

    <h2>Datasets</h2>
    <p><strong>LC25000:</strong> ~25,000 images with synthetic augmentation. Provides scale but limited real world variability.</p>
    <p><strong>LungHist700:</strong> 691 real clinical samples added to improve generalization.</p>

    <h2>Limitations</h2>
    <ul>
        <li>Training data is partially synthetic</li>
        <li>Confidence scores are softmax probabilities, not calibrated uncertainty</li>
        <li>Not validated across multiple hospitals</li>
        <li>For research and education only, not clinical diagnosis</li>
    </ul>

    <div class="author">
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
# BUILD APP
# ============================================================
with gr.Blocks(css=css, title="Histomancer") as demo:
    gr.HTML(watermark)

    with gr.Tabs():
        # HOME
        with gr.Tab("Home"):
            gr.HTML(home_html)

        # DIAGNOSTIC
        with gr.Tab("Diagnostic"):
            gr.HTML('<div class="page-container"><h1 class="page-title">Analyze Tissue</h1><p class="page-subtitle">Upload a histopathology image or select a sample below</p></div>')

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Upload Image", height=280)

                    gr.Examples(
                        examples=[
                            ["test_images/adenocarcinoma.jpg"],
                            ["test_images/benign_tissue.png"],
                            ["test_images/squamous_cell_carcinoma.png"]
                        ],
                        inputs=input_image,
                        label="Sample Images"
                    )

                    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

                with gr.Column(scale=1):
                    output_label = gr.Label(label="Results", num_top_classes=3)

                    # These are hidden until analysis completes
                    with gr.Row(visible=False) as viz_buttons:
                        gradcam_btn = gr.Button("Show Grad CAM")
                        saliency_btn = gr.Button("Show Saliency Map")

                    with gr.Row(visible=False) as viz_outputs:
                        gradcam_out = gr.Image(type="pil", label="Grad CAM", height=250)
                        saliency_out = gr.Image(type="pil", label="Saliency Map", height=250)

            gr.HTML('<p style="color:#71717a;font-size:13px;text-align:center;margin-top:24px;">For research and education only. Not for clinical diagnosis.</p>')

            # Events
            analyze_btn.click(
                fn=analyze_image,
                inputs=input_image,
                outputs=[output_label, viz_buttons, viz_outputs]
            )
            gradcam_btn.click(fn=get_gradcam, inputs=input_image, outputs=gradcam_out)
            saliency_btn.click(fn=get_saliency, inputs=input_image, outputs=saliency_out)

        # ABOUT
        with gr.Tab("About"):
            gr.HTML(about_html)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
