import cv2
from ultralytics import YOLO
# import pyzed.sl as sl
import numpy as np

model = YOLO("yolov8m-seg.pt")
cap = cv2.VideoCapture(0)
                       

if not cap.isOpened():
    print("Cannot open camera")
    exit()


ret, frame = cap.read()

if not ret or frame is None:
    print("failed to capture")
    cap.release()
    exit()

# cv2.imshow("test", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if frame.dtype != np.uint8:
    frame = frame.astype(np.uint8)



results = model(frame)
annot = results[0].plot()

# cv2.imshow("YOLO", annot)
cv2.imwrite("8mq-seg_pre.jpg", annot)
# cv2.waitKey(0)
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

