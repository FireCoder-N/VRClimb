import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Assuming you're capturing video from the default camera

#todo: Fine tune boundaries
boundaries = [([17, 15, 100], [50, 56, 200]),
              ([86, 31, 4], [220, 88, 50]),
              ([25, 146, 190], [62, 174, 250])]


while True:
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

    results = []
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)
        output = cv2.resize(output, (frame.shape[1], frame.shape[0]))

        results.append(output)


    # Concatenate the results into a 2x2 block
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    combined_output = np.block([
        [ [frame], 		[results[0]] ], 
        [ [results[1]], [results[2]] ]
    ])
    
	
    # show the images
    cv2.imshow("images", combined_output)

cap.release()
cv2.destroyAllWindows()
