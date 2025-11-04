import cv2
import numpy as np

def process_image(binary_image, X=50):
    # Ensure the image is binary (black and white), assuming 0 is black, 255 is white
    # _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Get the dimensions of the image (rows and columns)
    rows, cols = binary_image.shape

    # Process each row of the image
    for row in range(rows):
        count_black = 0
        erase = False

        # Traverse each pixel in the row
        for col in range(cols):
            if binary_image[row, col] == 0:  # Black pixel
                if erase:
                    count_black = 0
                erase = False
                count_black += 1
            else:  # White pixel
                if count_black >= X:
                    # We found at least X black pixels in a row, turn to black
                        binary_image[row, col] = 0
                        erase = True
                else:
                    count_black = 0

    # cv2.imshow('Result', binary_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return binary_image

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