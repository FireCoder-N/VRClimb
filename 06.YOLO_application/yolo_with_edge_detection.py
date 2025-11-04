import numpy as np
import cv2
from ultralytics import YOLO
import torch
import process

cap = cv2.VideoCapture(0)
                       
if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()

if not ret or frame is None:
    print("failed to capture")
    cap.release()
    exit()

# frame = cv2.imread("C:/Users/Mike/Documents/N/wall_test.jpg")
# frame = cv2.resize(frame, (640,480))
 
model = YOLO("best.pt")
results = model(frame, retina_masks=True) #Retina Masks result in a mask with finer details

mask = results[0].masks.data
mask = torch.any(mask, dim=0).int()
mask = mask.cpu().numpy().astype(np.uint8) * 255

# mask = cv2.GaussianBlur(mask, (1,1), 0)
mask = process.erode(mask)


edges = cv2.Canny(frame,100,200)

if edges.shape != mask.shape:
    mask = cv2.resize(mask, (edges.shape[1], edges.shape[0]))

temp = cv2.multiply(edges, mask)
inverted = cv2.bitwise_not(temp)

result = cv2.multiply(inverted, mask)

 
cv2.imshow('Result', result)
# cv2.imwrite("canny+yolo_l50_retina.png", result)
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
