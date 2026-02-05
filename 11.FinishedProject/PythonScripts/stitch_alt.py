import cv2
import os

def main(folder_path = "C:/Users/Mike/Documents/9.Scene/MyProject/WallAssets/captures", panorama_path="C:/Users/Mike/Documents/9.Scene/MyProject/WallAssets/panorama.png"):
    
    file_names = [f for f in os.listdir(folder_path) if (f.endswith(('.png', '.jpg', '.jpeg')))]
    imgs = [cv2.imread(os.path.join(folder_path, f)) for f in file_names]

    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch(imgs)

    if status != cv2.STITCHER_OK:
        raise RuntimeError("Stitching Failed:" + str(status)) 
    cv2.imwrite(panorama_path, pano)
