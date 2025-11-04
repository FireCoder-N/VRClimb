import cv2
import numpy
import pyzed.sl as sl


zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.sdk_verbose = 0

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Create an RGBA sl.Mat object
image_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)


if zed.grab() == sl.ERROR_CODE.SUCCESS :
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
    ocv = cv2.normalize(ocv, ocv, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


    # Display the left image from the numpy array
    cv2.imshow("Image", ocv)
    cv2.waitKey(0)

zed.close()

