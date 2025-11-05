import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt

# def boost_grayscale(image):
#     """
#     Boosts the grayscale image by first ignoring completely black pixels (value 0),
#     shifting other pixel values toward 128, and then exaggerating small differences.
    
#     Parameters:
#     image (numpy.ndarray): Grayscale image (2D array) to be processed.
    
#     Returns:
#     numpy.ndarray: The boosted grayscale image.
#     """

#     # Step 1: Ignore completely black pixels
#     mask = image > 0  # Mask to ignore black pixels (value 0)

#     # Step 2: Shift the pixel values toward 128
#     shifted_image = np.zeros_like(image, dtype=np.float32)
#     shifted_image[mask] = (image[mask] - 128) * -1 + 128  # Shift towards 128
    
#     # # Step 3: Exaggerate small changes
#     # # Apply contrast stretch to the non-black pixels
#     # min_value = np.min(shifted_image[mask])
#     # max_value = np.max(shifted_image[mask])

#     # if max_value > min_value:  # Avoid division by zero
#     #     # Stretch the contrast of the non-black pixels to exaggerate differences
#     #     exaggerated_image = np.zeros_like(shifted_image, dtype=np.float32)
#     #     exaggerated_image[mask] = 255 * (shifted_image[mask] - min_value) / (max_value - min_value)
#     # else:
#     #     exaggerated_image = shifted_image.copy()  # No change if the image is uniform

#     # Convert back to uint8 type for display or saving
#     final_image = shifted_image.astype(np.uint8)

#     return final_image


def sigmoid_process(image):
    # Calculate the mean depth value from the grayscale image
    depth_mean = np.mean(image)

    #Apply sigmoid to boost colors
    k = 1/10
    image = 255 / (1 + np.exp(-k * (image - depth_mean)))

    image = np.array(image, dtype='uint8')

    return image

def erode(image, erosion_level=3):
    structuring_kernel = np.full(shape=(erosion_level, erosion_level), fill_value=255)

    orig_shape = image.shape
    pad_width = erosion_level - 2

    # pad the matrix with `pad_width`
    image_pad = np.pad(array=image, pad_width=pad_width, mode='constant')
    pimg_shape = image_pad.shape
    h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])

    # sub matrices of kernel size
    flat_submatrices = np.array([
        image_pad[i:(i + erosion_level), j:(j + erosion_level)]
        for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)
    ])

    # condition to replace the values - if the kernel equal to submatrix then 255 else 0
    image_erode = np.array([255 if (i == structuring_kernel).all() else 0 for i in flat_submatrices])
    image_erode = image_erode.reshape(orig_shape)

    return image_erode.astype(np.uint8)

############
# ZED CAMERA
############

zed = sl.Camera()

# set configuration parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER
init_params.sdk_verbose = 0
init_params.depth_maximum_distance = 4

runtime_parameters =sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

image_zed = sl.Mat()
image_l_zed = sl.Mat()

############
# YOLO MODEL
############
model = YOLO("./best.pt")


if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
    # depthmap
    zed.retrieve_image(image_zed, sl.VIEW.DEPTH)
    ocv = image_zed.get_data()
    ocv = cv2.cvtColor(ocv, cv2.COLOR_RGBA2GRAY)
    ocv = cv2.normalize(ocv, ocv, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imshow("Image", ocv)
    # cv2.imwrite("depthmask.png", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for j in range(2):
        for col in ocv.T.astype(np.float32):
            me = np.mean(col)
            for i in range(len(col)):
                col[i] /= me

        for col in ocv.astype(np.float32):
            me = np.mean(col)
            for i in range(len(col)):
                col[i] /= me

    ocv *= 255

    cv2.imshow("Image", ocv.astype(np.uint8))
    # cv2.imwrite("depthmask.png", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # ocv = sigmoid_process(ocv)

    # yolomask
    zed.retrieve_image(image_l_zed, sl.VIEW.LEFT)
    ocv2 = image_l_zed.get_data()
    ocv2 = cv2.cvtColor(ocv2, cv2.COLOR_RGBA2RGB)

    results = model(ocv2, retina_masks=True)

    mask = results[0].masks.data
    mask = torch.any(mask, dim=0).int()
    mask = mask.cpu().numpy().astype(np.uint8) * 255
    mask = erode(mask)

    # canny
    edges = cv2.Canny(ocv,10,20)
    
    # yolo + canny
    yc = cv2.multiply(edges, mask)
    inverted_yc = cv2.bitwise_not(yc)
    result = cv2.multiply(inverted_yc, mask)

    # final
    final = cv2.multiply(ocv, result//255)

    # for i in range(3):
    #     final = boost_grayscale(final)

    cv2.imshow("Image", edges)
    # cv2.imwrite("depthmask.png", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

zed.close()

