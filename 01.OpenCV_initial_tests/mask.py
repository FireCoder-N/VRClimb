import cv2
import numpy as np

# Load the overlay image
overlay = cv2.imread(r"./wp5333295.jpg", cv2.IMREAD_UNCHANGED)

if overlay is None or overlay.size == 0:
    print("Error loading overlay image. Check the file path.")
    exit()

# Define the color range for red (you can adjust this based on the color of the object you want to track)

lower_red = np.array([0, 0, 100])
upper_red = np.array([100, 100, 255])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame from the camera.")
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the red color
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found, track the largest one
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # Resize the overlay to match the bounding box dimensions
        overlay_resized = cv2.resize(overlay, (w, h))

        # Blend the frame and overlay
        alpha = 0.5
        blended = frame.copy()
        blended[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 1 - alpha, overlay_resized, alpha, 0)

        cv2.imshow("Object Tracking", blended)
    else:
        cv2.imshow("Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
