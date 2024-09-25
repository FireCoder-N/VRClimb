import cv2
from ultralytics import YOLO
# import pyzed.sl as sl
import numpy as np

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
                       

if not cap.isOpened():
    print("Cannot open camera")
    exit()


ret, frame = cap.read()

if not ret or frame is None:
    print("failed to capture")
    cap.release()
    exit()

cv2.imshow("test", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

if frame.dtype != np.uint8:
    frame = frame.astype(np.uint8)

temp_img = "temp_frame.jpg"
cv2.imwrite(temp_img, frame_rgb)

results = model(temp_img)
results.show()
 
# When everything done, release the capture
cap.release()
# cv2.destroyAllWindows()

