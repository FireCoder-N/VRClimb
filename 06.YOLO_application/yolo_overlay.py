import cv2
from ultralytics import YOLO
# import pyzed.sl as sl
import numpy as np
import torch

# model = YOLO("yolov8m-seg.pt")
model = YOLO("best.pt")
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

# ==========================================
#      proof of concept: random image
# ==========================================
frame = cv2.imread("wall_test.jpg")
frame = cv2.resize(frame, (640, 480))

results = model(frame)
annot = results[0].plot()

# ==========================================
#         edit: calculate binary mask
# ==========================================
masks = results[0].masks.data
combined_mask = torch.any(masks, dim=0).int()

# Convert the combined mask to 255 for visualization (binary mask)
combined_mask = combined_mask.cpu().numpy().astype(np.uint8) * 255


cv2.imshow("YOLO", annot)
cv2.waitKey(0)
cv2.imshow("YOLO", frame)
cv2.waitKey(0)
cv2.imshow("YOLO", combined_mask)
# cv2.imwrite("8m-seg_post_mask.jpg", combined_mask)
cv2.waitKey(0)
 
# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()

