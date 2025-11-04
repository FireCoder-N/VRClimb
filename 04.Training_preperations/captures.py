import cv2

"""
Capture screenshots using OpenCV from a regular webcam, to be later used for annotation.
"""

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

screenshot_count = 0
max_screenshots = 50

while screenshot_count < max_screenshots:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display the frame
    cv2.imshow('Webcam', frame)
    
    # Wait for keypress
    key = cv2.waitKey(1) & 0xFF
    
    # If 's' is pressed, save the screenshot
    if key == ord('s'):
        screenshot_filename = f"{screenshot_count}.png"
        cv2.imwrite(screenshot_filename, frame)
        print(f"Screenshot {screenshot_count} saved as {screenshot_filename}")
        screenshot_count += 1

    # Exit if 'q' is pressed
    if key == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

print(f"Program finished after capturing {screenshot_count} screenshots.")
