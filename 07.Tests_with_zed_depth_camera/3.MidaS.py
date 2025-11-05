import cv2
import torch
import pyzed.sl as sl
import numpy as np

# ====================================
#               Model
# ==================================== 
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# ====================================
#               Image
# ==================================== 
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
    # Retrieve the left image in sl.Mat
    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    # Use get_data() to get the numpy array
    img = image_zed.get_data()
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
else:
    print("Failed")
    exit()

input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

# output = (output - output.min()) / (output.max() - output.min()) * 255
output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# output = output.astype(np.uint8)  # Convert to uint8

cv2.imshow("Image", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
zed.close()