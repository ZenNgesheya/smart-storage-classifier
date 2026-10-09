"""
app.py

Flask web server that:
  - Serves a mobile-friendly web page for capturing photos with a phone camera
  - Accepts an uploaded photo, sends it to the trained Roboflow model
  - Returns the predicted storage category and confidence as JSON
"""

import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from roboflow import Roboflow

MODEL_ID = "michael-nghidengwa/grocery-storage-classifier/3"
UPLOAD_PATH = "uploaded_frame.jpg"
CONFIDENCE_THRESHOLD = 40  # percent

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise RuntimeError("ROBOFLOW_API_KEY not found. Check your .env file.")

rf = Roboflow(api_key=api_key)
workspace_id, project_id, version_number = MODEL_ID.split("/")
project = rf.workspace(workspace_id).project(project_id)
version = project.version(int(version_number))

trained_models = version.models()
if not trained_models:
    raise RuntimeError("No trained models found for this version.")

model = trained_models[0]
print("Connected to Roboflow model:", MODEL_ID)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_file.save(UPLOAD_PATH)

    try:
        prediction = model.predict(UPLOAD_PATH).json()
        predictions = prediction.get("predictions", [])

        if not predictions:
            return jsonify({"class": None, "confidence": 0, "message": "No prediction returned"})

        top = predictions[0]
        class_name = top.get("class", "Unknown")
        confidence = round(top.get("confidence", 0) * 100, 1)

        is_uncertain = confidence < CONFIDENCE_THRESHOLD

        return jsonify({
            "class": class_name,
            "confidence": confidence,
            "uncertain": is_uncertain
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # host="0.0.0.0" makes the server reachable from other devices on the
    # same Wi-Fi network (like your phone), not just this PC.
    app.run(host="0.0.0.0", port=5000, debug=True)