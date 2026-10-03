"""
classify_live.py

Captures live frames from a USB camera and classifies each one using
the trained Roboflow model (Fridge / Freezer / Cardboard-Pantry).

Controls:
  SPACE -> capture current frame and classify it
  q     -> quit
"""

import os
import cv2
from dotenv import load_dotenv
from roboflow import Roboflow

# ---- Configuration ----
# Replace with your actual model workspace/project/version, e.g.:
# "michael-nghidengwa/grocery-storage-classifier/2"
MODEL_ID = "michael-nghidengwa/grocery-storage-classifier/2"

TEMP_IMAGE_PATH = "temp_frame.jpg"
CONFIDENCE_THRESHOLD = 40  # percent; below this we show "Uncertain"

# ---- Load API key from .env ----
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    print("ERROR: ROBOFLOW_API_KEY not found. Check your .env file.")
    exit()

# ---- Connect to Roboflow model ----
rf = Roboflow(api_key=api_key)
workspace_id, project_id, version_number = MODEL_ID.split("/")
project = rf.workspace(workspace_id).project(project_id)
version = project.version(int(version_number))

trained_models = version.models()
if not trained_models:
    print("ERROR: No trained models found for this version.")
    exit()

model = trained_models[0]

print("Connected to Roboflow model:", MODEL_ID)

# ---- Open camera ----
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Press SPACE to classify current frame, q to quit.")

last_result_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    display_frame = frame.copy()

    if last_result_text:
        cv2.putText(display_frame, last_result_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(display_frame, "SPACE = classify | q = quit",
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Storage Classifier", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        # Save the current frame to disk and send it for classification
        cv2.imwrite(TEMP_IMAGE_PATH, frame)
        print("Classifying...")

        try:
            prediction = model.predict(TEMP_IMAGE_PATH).json()
            predictions = prediction.get("predictions", [])

            if predictions:
                top = predictions[0]
                class_name = top.get("class", "Unknown")
                confidence = round(top.get("confidence", 0) * 100, 1)

                if confidence < CONFIDENCE_THRESHOLD:
                    last_result_text = f"Uncertain ({class_name} {confidence}%)"
                else:
                    last_result_text = f"{class_name} ({confidence}%)"

                print("Result:", last_result_text)
            else:
                last_result_text = "No prediction returned"
                print(last_result_text)

        except Exception as e:
            last_result_text = "Error classifying"
            print("Error:", e)

cap.release()
cv2.destroyAllWindows()