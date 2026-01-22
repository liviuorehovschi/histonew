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


def get_confidence_label(confidence: float) -> str:
    """Return confidence interpretation."""
    if confidence >= 0.95:
        return "Very high confidence"
    elif confidence >= 0.85:
        return "High confidence"
    elif confidence >= 0.70:
        return "Moderate confidence"
    else:
        return "Lower confidence - consider additional analysis"


def predict(image: Image.Image) -> dict:
    """Classify lung histopathology image."""
    if image is None:
        return {}

    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)

    # Return all class probabilities for Gradio label component
    results = {}
    for i, class_name in enumerate(CLASS_NAMES):
        label = CLASS_LABELS[class_name]
        results[label] = float(predictions[0][i])

    return results


def generate_gradcam(image: Image.Image) -> Image.Image:
    """Generate Grad-CAM visualization."""
    if image is None:
        return None

    # Preprocess
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    resized = img_array.astype(np.uint8)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Get prediction
    preds = model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds[0])

    # Build gradient model
    base_model = model.get_layer('efficientnetb0')
    conv_layer = base_model.get_layer('top_conv')
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[conv_layer.output, base_model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Generate heatmap
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap and overlay
    cam = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    combined = cv2.addWeighted(resized, 0.6, cam, 0.4, 0)

    return Image.fromarray(combined)


def generate_saliency(image: Image.Image) -> Image.Image:
    """Generate saliency map visualization."""
    if image is None:
        return None

    # Preprocess
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Compute saliency
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

    # Apply colormap
    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return Image.fromarray(heatmap)


def analyze_image(image: Image.Image):
    """Main analysis function - returns prediction, saliency, GradCAM."""
    if image is None:
        return {}, None, None

    try:
        # Validate image
        if not isinstance(image, Image.Image):
            print(f"[ERROR] Invalid image type: {type(image)}")
            return {}, None, None

        # Check image size
        if image.size[0] < 50 or image.size[1] < 50:
            print("[ERROR] Image too small")
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
# GRADIO INTERFACE - MATCHING ORIGINAL FLASK APP DESIGN
# ============================================================
custom_css = """
/* Color Variables matching original Flask app */
:root {
    --deep-blue: #1a1f2e;
    --darker-gray: #161923;
    --dark-gray: #1e2330;
    --medium-gray: #8892b0;
    --light-gray: #a8b2d1;
    --accent-gray: #ccd6f6;
    --glass-bg: rgba(26, 31, 46, 0.92);
    --glass-border: rgba(255, 255, 255, 0.08);
    --shadow: rgba(0, 0, 0, 0.25);
}

/* Global Gradio overrides */
.gradio-container {
    background-color: var(--deep-blue) !important;
    font-family: Arial, sans-serif !important;
    max-width: 1400px !important;
    margin: auto !important;
}

.dark {
    background-color: var(--deep-blue) !important;
}

/* Headers and text */
h1, h2, h3, h4 {
    color: var(--accent-gray) !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
}

p, label, span {
    color: var(--medium-gray) !important;
    line-height: 1.6;
}

/* Main container styling */
.contain {
    background-color: var(--darker-gray) !important;
}

/* Upload box styling - glass morphism */
.image-container, .file-preview {
    background-color: var(--dark-gray) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    padding: 20px !important;
}

/* Button styling matching original */
.primary-btn, button[variant="primary"] {
    background-color: var(--dark-gray) !important;
    color: var(--accent-gray) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease !important;
}

.primary-btn:hover, button[variant="primary"]:hover {
    background-color: var(--darker-gray) !important;
    transform: translateY(-2px) !important;
}

/* Label/results card */
.label-container {
    background-color: var(--dark-gray) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    padding: 20px !important;
}

/* Image preview styling */
.image-preview {
    border-radius: 8px !important;
    border: 1px solid var(--glass-border) !important;
}

/* Adjust output images */
.output-image {
    border-radius: 8px !important;
    background-color: var(--dark-gray) !important;
}

/* Section headers */
#title-header {
    text-align: center;
    margin-bottom: 2rem;
    color: var(--accent-gray);
}

#subtitle {
    text-align: center;
    color: var(--medium-gray);
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Loading animation matching scanner effect */
@keyframes scanning {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.loading-scanner {
    position: relative;
    overflow: hidden;
}

.loading-scanner::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(168, 178, 209, 0.3), transparent);
    animation: scanning 1.5s infinite;
}
"""

with gr.Blocks(title="Lung Tissue Analysis Tool", theme=gr.themes.Base(), css=custom_css) as demo:
    gr.HTML("""
        <div id="title-header">
            <h1>LUNG TISSUE ANALYSIS TOOL</h1>
            <p id="subtitle">Upload a histopathological image for AI-powered classification</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Image")
            input_image = gr.Image(
                type="pil",
                label="Histopathology Image",
                height=300,
                elem_classes="upload-box"
            )

            gr.Markdown("### Sample Images")
            sample_dropdown = gr.Dropdown(
                choices=["Select sample...", "Adenocarcinoma", "Benign Tissue", "Squamous Cell Carcinoma"],
                label="Quick Test",
                value="Select sample..."
            )

            analyze_btn = gr.Button(
                "ANALYZE",
                variant="primary",
                elem_classes="primary-btn",
                size="lg"
            )

        with gr.Column(scale=1):
            gr.Markdown("### Classification Results")
            output_label = gr.Label(
                label="Diagnosis",
                num_top_classes=3,
                elem_classes="label-container"
            )

    gr.HTML("<h2 style='text-align: center; margin-top: 2rem;'>AI INTERPRETABILITY</h2>")

    with gr.Row():
        saliency_output = gr.Image(
            type="pil",
            label="Saliency Map",
            height=300,
            elem_classes="output-image"
        )
        gradcam_output = gr.Image(
            type="pil",
            label="Grad-CAM Visualization",
            height=300,
            elem_classes="output-image"
        )

    gr.Markdown("""
    ---
    **About the Visualizations:**
    - **Saliency Map**: Shows pixel-level importance for the classification decision
    - **Grad-CAM**: Highlights regions the model focused on (red = high attention)

    **Model Info:** EfficientNetB0 | 98.25% Test Accuracy | Trained on LC25000 + LungHist700

    *This is an educational tool - not for clinical diagnostic use*
    """)

    # Event handlers
    analyze_btn.click(
        fn=analyze_image,
        inputs=input_image,
        outputs=[output_label, saliency_output, gradcam_output]
    )

    # Auto-analyze on upload
    input_image.change(
        fn=analyze_image,
        inputs=input_image,
        outputs=[output_label, saliency_output, gradcam_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
