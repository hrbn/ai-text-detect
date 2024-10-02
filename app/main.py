import os
import torch
from flask import Flask, request, jsonify, render_template
from module import SequenceClassificationModule
from utils import best_checkpoint

import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load model and move it to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lit_module = SequenceClassificationModule.load_from_checkpoint(best_checkpoint()).to(device)


# Route to serve the UI
@app.route("/")
def index():
    return render_template("index.html")


# API endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if the request's content-type is application/json
        if not request.is_json:
            logging.error("Request content type is not application/json")
            return jsonify({"error": "Invalid content type. Expected application/json"}), 400

        data = request.get_json()

        # Verify 'text' key exists in the JSON payload
        if "text" not in data:
            logging.error("Missing 'text' key in JSON payload")
            return jsonify({"error": "Invalid request payload. Missing 'text' key."}), 400

        sequence = data["text"]

        # Run prediction
        with torch.no_grad():
            res = lit_module.predict_step(sequence)

        if "label" not in res:
            logging.error("Missing 'label' key in model output")
            return jsonify({"error": "Invalid response from model. Missing 'label' key."}), 400

        return jsonify({"prediction": res["label"], "probability": res["probability"]})

    except Exception as e:
        logging.exception("An error occurred during prediction:")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
