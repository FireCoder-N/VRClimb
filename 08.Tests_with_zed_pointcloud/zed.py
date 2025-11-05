import cv2
import numpy as np
import pyzed.sl as sl

def processimage(image):
    # Step 1: Calculate the mean depth value from the grayscale image
    depth_mean = np.mean(image)

    # Step 2: Define a threshold to identify overly "white" regions (whiter = nearer in the depth map)
    # You can adjust this value based on how strict the clipping should be (closer to 255 means closer to the camera)
    threshold = 235  # Define a threshold near white (values from 0 to 255)

    # Step 3: Clip values above the threshold and replace them with the mean depth value
    # image = np.where(image > threshold, depth_mean, image)
    # image = np.where(image < 255 - threshold, depth_mean, image)

    k = 1/10
    image = 255 / (1 + np.exp(-k * (image - depth_mean)))

    image = np.where(image > threshold, depth_mean, image)
    image = np.where(image < 255 - threshold, depth_mean, image)

    # gamma = 5.0  # Change this value to exaggerate more or less
    # image = np.array(255 * (image / 255) ** gamma, dtype='uint8')

    image = np.array(image, dtype='uint8')

    return image




zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER
init_params.sdk_verbose = 0
init_params.depth_maximum_distance = 4
# init_params.depth_minimum_distance = 5 # Set the minimum depth perception distance to 15cm

runtime_parameters =sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Create an RGBA sl.Mat object
image_zed = sl.Mat() #zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
depth_zed = sl.Mat() #zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.F32_C1)
pointcloud_zed = sl.Mat()


if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
    # ---- get image ----
    # # Retrieve the left image in sl.Mat
    # zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    # # Use get_data() to get the numpy array
    # ocv = image_zed.get_data()

    # ---- get depth ----
    # Retrieve depth data (32-bit)
    zed.retrieve_image(image_zed, sl.VIEW.DEPTH)
    # Load depth data into a numpy array
    ocv = image_zed.get_data()
    ocv = cv2.cvtColor(ocv, cv2.COLOR_RGBA2GRAY)
    ocv = cv2.normalize(ocv, ocv, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    

    #  ---- display image ----
    ocv = processimage(ocv)
    cv2.imshow("Image", ocv)
    # cv2.imwrite("depthmask.png", ocv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # ---- measure depth ----
    # Retrieve depth data (32-bit)
    zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
    # Load depth data into a numpy array
    ocv = image_zed.get_data()
    
    # print(ocv[int(len(ocv)/2)][int(len(ocv[0])/2)])

    # ---- get poincloud ----
    zed.retrieve_measure(pointcloud_zed, sl.MEASURE.XYZRGBA) # Retrieve colored point cloud
    x = round(image_zed.get_width() / 2)
    y = round(image_zed.get_height() / 2)
    err, depth_value = depth_zed.get_value(x, y)
    print(f"Distance to Camera at ({x}, {y}): {depth_value} m")

zed.close()

