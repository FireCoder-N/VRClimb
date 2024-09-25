import cv2
from ultralytics import YOLO
# import pyzed.sl as sl
import numpy as np

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
                       

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     print(frame.dtype)
 
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(1) == ord('q'):
#         break

# while cap.isOpened():
#     suc, frame = cap.read()

#     if suc:
#         if frame.dtype != np.uint8:
#             frame = frame.astype(np.uint8)

#         results = model(frame)

#         annotated = results[0].plot()

#         cv2.imshow("YOLOv8 Inference", annotated)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     else:
#         break


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

