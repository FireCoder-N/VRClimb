import numpy as np
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt


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

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

model = YOLO("best.pt")
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.sdk_verbose = 0

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

image_l = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
image_r = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)


if zed.grab() == sl.ERROR_CODE.SUCCESS :
    # ---- get left image ----
    zed.retrieve_image(image_l, sl.VIEW.LEFT)
    imgL = image_l.get_data()
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGRA2RGB)
    
    results_l = model(imgL, retina_masks=True)
    mask_l = results_l[0].masks.data
    mask_l = torch.any(mask_l, dim=0).int()
    mask_l = mask_l.cpu().numpy().astype(np.uint8) * 255

    mask_l = erode(mask_l)

    edges = cv2.Canny(imgL,100,200)

    if edges.shape != mask_l.shape:
        mask_l = cv2.resize(mask_l, (edges.shape[1], edges.shape[0]))

    temp = cv2.multiply(edges, mask_l)
    inverted = cv2.bitwise_not(temp)

    result_l = cv2.multiply(inverted, mask_l)
    # cv2.imshow("Image", results_l[0].plot())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ---- get right image ----
    zed.retrieve_image(image_r, sl.VIEW.RIGHT)
    imgR = image_r.get_data()
    imgR = cv2.cvtColor(imgR, cv2.COLOR_RGBA2RGB)
    
    results_r = model(imgR, retina_masks=True)
    mask_r = results_r[0].masks.data
    mask_r = torch.any(mask_r, dim=0).int()
    mask_r = mask_r.cpu().numpy().astype(np.uint8) * 255

    mask_r = erode(mask_r)

    edges = cv2.Canny(imgR,100,200)

    if edges.shape != mask_r.shape:
        mask_r = cv2.resize(mask_r, (edges.shape[1], edges.shape[0]))

    temp = cv2.multiply(edges, mask_r)
    inverted = cv2.bitwise_not(temp)

    result_r = cv2.multiply(inverted, mask_r)
    

    # Create empty color channels
    red_channel = np.zeros_like(mask_l)  # Red for the first mask
    blue_channel = np.zeros_like(mask_r)  # Blue for the second mask

    # Assign the whites in mask1 to red and in mask2 to blue
    red_channel[mask_l == 255] = 255
    blue_channel[mask_r == 255] = 255

    # Combine the channels to create the final image
    # (R, G, B) - red_channel in R, blue_channel in B, nothing in G
    final_image = np.stack([red_channel, np.zeros_like(red_channel), blue_channel], axis=2)

    # cv2.imshow("Image", final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    stereo = cv2.StereoSGBM_create(
        # minDisparity=0,
        # numDisparities=16*10,  # Larger range of disparities for finer depth granularity
        # blockSize=3,          # Small block size for capturing more detail
        # uniquenessRatio=5,    # Helps with eliminating ambiguous matches
        # speckleWindowSize=50,  # Filters out noise
        # speckleRange=4,
        # disp12MaxDiff=1,
        # P1=24,   # Penalty on the disparity smoothness
        # P2= 32   # Larger values for smoother disparity transitions
    )


    disparity = stereo.compute(mask_l, mask_r)
    cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    print(disparity)
    disparity = np.uint8(disparity)


    # Display the left image from the numpy array
    # cv2.imshow("Image", disparity)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    
    # Plot each image in the grid
    axes[0, 0].imshow(results_l[0].plot())
    axes[0, 1].imshow(result_l, cmap='gray')
    axes[0, 2].imshow(final_image)
    axes[1, 0].imshow(results_r[0].plot())
    axes[1, 1].imshow(result_r, cmap='gray')
    axes[1, 2].imshow(disparity, cmap='gray')

    # Turn off axes for each subplot
    for ax in axes.flat:
        ax.axis('off')

    # Display the plot
    plt.tight_layout()
    plt.show()



zed.close()


