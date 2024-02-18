import cv2
import numpy as np

# Load the image you want to overlay
overlay = cv2.imread(r"C:\Users\Nestoras\Pictures\wallpaper\wp5333295.jpg", cv2.IMREAD_UNCHANGED)

if overlay is None or overlay.size == 0:
    print("Error loading overlay image. Check the file path.")
    exit()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame from the camera.")
        break

    # Resize overlay to match the dimensions of the frame
    overlay_resized = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))

    # Blend the frame and overlay
    alpha = 0.5
    blended = cv2.addWeighted(frame, 1 - alpha, overlay_resized, alpha, 0)

    cv2.imshow("Overlay", blended)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
