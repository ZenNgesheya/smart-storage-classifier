import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Press SPACE to save a photo, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Quick Snapshot - press SPACE to save, q to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        cv2.imwrite("quick_test_photo.jpg", frame)
        print("Saved as quick_test_photo.jpg")
        break
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()