"""
capture_images.py

Captures photos from a USB camera and saves them into the correct
dataset folder (fridge / freezer / cardboard) for training the
storage-location classifier.

Controls:
  1 -> save current frame into dataset/fridge/
  2 -> save current frame into dataset/freezer/
  3 -> save current frame into dataset/cardboard/
  q -> quit
"""

import cv2
import os
import time

DATASET_DIR = "dataset"
CLASSES = {
    ord("1"): "fridge",
    ord("2"): "freezer",
    ord("3"): "cardboard",
}

# Make sure folders exist
for class_name in CLASSES.values():
    os.makedirs(os.path.join(DATASET_DIR, class_name), exist_ok=True)

# Open the USB camera (0 is usually the default camera; change to 1 if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera. Try changing the camera index (0, 1, 2...).")
    exit()

print("Camera opened successfully.")
print("Press 1 = fridge | 2 = freezer | 3 = cardboard | q = quit")

counts = {name: len(os.listdir(os.path.join(DATASET_DIR, name))) for name in CLASSES.values()}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Show live counts on screen
    display_frame = frame.copy()
    y = 30
    for name, count in counts.items():
        cv2.putText(display_frame, f"{name}: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30

    cv2.putText(display_frame, "1=fridge 2=freezer 3=cardboard q=quit",
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Dataset Capture", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key in CLASSES:
        class_name = CLASSES[key]
        filename = f"{class_name}_{int(time.time() * 1000)}.jpg"
        filepath = os.path.join(DATASET_DIR, class_name, filename)
        cv2.imwrite(filepath, frame)
        counts[class_name] += 1
        print(f"Saved: {filepath}  (total {class_name}: {counts[class_name]})")

cap.release()
cv2.destroyAllWindows()

print("\nFinal counts:")
for name, count in counts.items():
    print(f"  {name}: {count}")