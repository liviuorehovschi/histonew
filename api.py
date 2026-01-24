"""
Histomancer Flask REST API
Serves lung histopathology classification with Grad-CAM and saliency visualizations.
Also serves the React frontend from /static folder.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import base64
import os

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
print("[SUCCESS] Model loaded and ready!")

# ============================================================
# ML FUNCTIONS
# ============================================================

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess image for EfficientNetB0 inference."""
    img = img.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


def predict(image: Image.Image) -> dict:
    """Run inference and return all class probabilities."""
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


def get_confidence_level(confidence: float) -> str:
    """Determine confidence level based on thresholds."""
    if confidence >= 0.90:
        return "high"
    elif confidence >= 0.60:
        return "moderate"
    else:
        return "low"


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# ============================================================
# FLASK APP
# ============================================================

STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})


# Serve React frontend
@app.route('/')
def serve_index():
    """Serve the React app index.html."""
    return send_from_directory(STATIC_FOLDER, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files or fallback to index.html for client-side routing."""
    if os.path.exists(os.path.join(STATIC_FOLDER, path)):
        return send_from_directory(STATIC_FOLDER, path)
    return send_from_directory(STATIC_FOLDER, 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Run inference on uploaded image.
    Returns top diagnosis with confidence level.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        image = Image.open(file.stream)

        if image.size[0] < 50 or image.size[1] < 50:
            return jsonify({"error": "Image too small (minimum 50x50)"}), 400

        all_predictions = predict(image)

        if not all_predictions:
            return jsonify({"error": "Prediction failed"}), 500

        # Get top prediction
        top_class = max(all_predictions, key=all_predictions.get)
        top_confidence = all_predictions[top_class]
        confidence_level = get_confidence_level(top_confidence)

        return jsonify({
            "diagnosis": top_class,
            "confidence": round(top_confidence, 4),
            "confidenceLevel": confidence_level,
            "allPredictions": all_predictions
        })

    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/gradcam', methods=['POST'])
def api_gradcam():
    """Generate and return Grad-CAM visualization."""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        image = Image.open(file.stream)

        if image.size[0] < 50 or image.size[1] < 50:
            return jsonify({"error": "Image too small (minimum 50x50)"}), 400

        gradcam_img = generate_gradcam(image)

        if gradcam_img is None:
            return jsonify({"error": "Grad-CAM generation failed"}), 500

        return jsonify({
            "image": image_to_base64(gradcam_img)
        })

    except Exception as e:
        print(f"[ERROR] Grad-CAM failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/saliency', methods=['POST'])
def api_saliency():
    """Generate and return saliency map visualization."""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        image = Image.open(file.stream)

        if image.size[0] < 50 or image.size[1] < 50:
            return jsonify({"error": "Image too small (minimum 50x50)"}), 400

        saliency_img = generate_saliency(image)

        if saliency_img is None:
            return jsonify({"error": "Saliency map generation failed"}), 500

        return jsonify({
            "image": image_to_base64(saliency_img)
        })

    except Exception as e:
        print(f"[ERROR] Saliency failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
