import cv2

## Specify the index of the camera you want to use (0, 1, 2, etc.)
##camera_index = 0  # Change this according to your setup
##
##cap = cv2.VideoCapture(camera_index)

cap = cv2.VideoCapture(0)  # 0 indicates the default camera (you may need to change this number)
while True:
    ret, frame = cap.read()
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

