import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

import tempfile
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import cv2
import logging
from werkzeug.utils import secure_filename
from monai.transforms import ScaleIntensity
from flask import Flask, render_template, request, jsonify
from qiskit_machine_learning.algorithms import PegasosQSVC
from sklearn.metrics import confusion_matrix
from image_processing.data_loader import count_images
from image_processing.image_transformations import apply_grayscale, apply_gaussian_blur
from image_processing.dimensionality_reduction import (
    X_train_reduced, X_test_reduced,
    MinMaxScaler, reduce_to_n_dimensions
)
from image_processing.data_loader import (
    y_train, y_test,
    train_path_norm, train_path_pneu, test_path_norm, test_path_pneu
)
from quantum_classification.quantum_async_jobs import submit_quantum_job, check_quantum_job
from workflow.workflow_manager import WorkflowManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

workflow_manager = WorkflowManager()

MODEL_PATH = "models/PegasosQSVC_Fidelity_quantm_trainer_pneumonia.model"
if os.path.exists(MODEL_PATH):
    loaded_model = PegasosQSVC.load(MODEL_PATH)
    workflow_manager.model = loaded_model
    log.info("Loaded trained QSVC model for Pneumonia Detection.")
else:
    log.warning("No trained pneumonia model found! Training a new one...")
    workflow_manager.train_quantum_model()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dataset-info")
def dataset_info():
    return jsonify({
        "train_normal": count_images(train_path_norm),
        "train_pneumonia": count_images(train_path_pneu),
        "test_normal": count_images(test_path_norm),
        "test_pneumonia": count_images(test_path_pneu),
    })


@app.route("/xray-image")
def xray_image():
    folder = train_path_pneu
    if not os.path.exists(folder):
        return jsonify({"error": "Dataset folder not found!"}), 404

    images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        return jsonify({"error": "No X-ray images found!"}), 404

    selected = random.choice(images)
    path = os.path.join(folder, selected)
    log.info(f"Selected Chest X-ray Image: {selected}")

    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    scaled = ScaleIntensity(minv=0.0, maxv=1.0)(xray.astype(np.float32))
    colored = plt.cm.viridis(scaled / np.max(scaled))

    gray_path = os.path.join(STATIC_DIR, "xray_gray.jpg")
    colored_path = os.path.join(STATIC_DIR, "xray_colored.jpg")
    plt.imsave(gray_path, xray, cmap="gray")
    plt.imsave(colored_path, colored)

    return jsonify({
        "gray_image": "static/xray_gray.jpg",
        "colored_image": "static/xray_colored.jpg"
    })


@app.route("/pca-plot")
def pca_plot():
    try:
        plt.figure(figsize=(10, 8))
        plt.scatter(
            X_train_reduced[y_train == 0][:, 0],
            X_train_reduced[y_train == 0][:, 1],
            label="Normal",
            color="green", marker="o"
        )
        plt.scatter(
            X_train_reduced[y_train == 1][:, 0],
            X_train_reduced[y_train == 1][:, 1],
            label="Pneumonia",
            color="red", marker="x"
        )
        plt.title("PCA Visualization of X-ray Dataset")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.grid(True)
        path = os.path.join(STATIC_DIR, "pca_plot.jpg")
        plt.savefig(path)
        return jsonify({"pca_plot": "static/pca_plot.jpg"})
    except Exception as e:
        log.exception("PCA plot generation failed")
        return jsonify({"error": str(e)}), 500


@app.route("/predict-probabilities")
def predict_probabilities():
    try:
        preds = workflow_manager.model.predict(X_test_reduced[:30])
        probs = torch.tensor(preds, dtype=torch.float32).numpy()

        plt.figure(figsize=(10, 6))
        plt.hist(probs, bins=30, color="blue", alpha=0.7, label="Prediction Scores")
        plt.axvline(0.5, color="black", linestyle="--", label="Threshold = 0.5")
        plt.title("Prediction Probability Distribution")
        plt.xlabel("Prediction Score")
        plt.ylabel("Frequency")
        plt.legend()

        path = os.path.join(STATIC_DIR, "predicted_probs.jpg")
        plt.savefig(path)
        return jsonify({"predicted_probs_plot": "static/predicted_probs.jpg"})
    except Exception as e:
        log.exception("Probability histogram failed")
        return jsonify({"error": str(e)}), 500


@app.route("/confusion-matrix")
def confusion_matrix_plot():
    try:
        preds = workflow_manager.model.predict(X_test_reduced[:30])
        cm = confusion_matrix(y_test[:30], preds)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Pneumonia Detection")
        path = os.path.join(STATIC_DIR, "confusion_matrix.jpg")
        plt.savefig(path)
        return jsonify({"conf_matrix_plot": "static/confusion_matrix.jpg"})
    except Exception as e:
        log.exception("Confusion matrix error")
        return jsonify({"error": str(e)}), 500


@app.route("/classification-score")
def classification_score():
    try:
        train = workflow_manager.model.score(X_train_reduced[:10], y_train[:10])
        test = workflow_manager.model.score(X_test_reduced[:10], y_test[:10])
        return jsonify({
            "train_score": f"{train:.2f}",
            "test_score": f"{test:.2f}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/classify-image", methods=["POST"])
def classify_uploaded_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "No selected image"}), 400

    try:
        temp = os.path.join(tempfile.gettempdir(), secure_filename(image.filename))
        image.save(temp)

        img = cv2.imread(temp)
        if img is None:
            raise ValueError("Invalid or unreadable xray image.")

        gray = apply_grayscale(img)
        blurred = apply_gaussian_blur(gray)
        resized = cv2.resize(blurred, (256, 256)).flatten().astype(np.float32)

        if np.std(resized) < 1e-3:
            raise ValueError("Uniform image — likely invalid.")

        # Quantum Feature Prep
        num_qubits = 18
        layers = 3
        total_params = num_qubits * layers

        reduced = reduce_to_n_dimensions(resized.reshape(1, -1), num_qubits)
        scaled = MinMaxScaler((0, np.pi)).fit_transform(reduced).flatten()
        features = np.tile(scaled, layers)[:total_params]

        if len(features) != total_params:
            raise ValueError(f"Expected {total_params} features, got {len(features)}")

        prediction = workflow_manager.classify_with_quantum_circuit(features)
        return jsonify({"quantum_prediction": prediction})

    except Exception as e:
        log.exception("Image classification failed")
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400


@app.route("/quantum-job/submit", methods=["POST"])
def quantum_job_submit():
    try:
        data = request.get_json()
        features = np.array(data.get("features", []), dtype=np.float32)
        job_id = submit_quantum_job(features)
        return jsonify({"job_id": job_id, "message": "Quantum job submitted."})
    except Exception as e:
        log.exception("Error submitting quantum job")
        return jsonify({"error": str(e)}), 400


@app.route("/quantum-job/status/<job_id>", methods=["GET"])
def quantum_job_status(job_id):
    try:
        result = check_quantum_job(job_id)
        return jsonify(result)
    except Exception as e:
        log.exception("Error checking quantum job")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
