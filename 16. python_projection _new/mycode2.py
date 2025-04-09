import os
import json
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CAMERA INTRINSICS (from ZED)
K = np.array([
    [1386.08, 0, 980.449],
    [0, 1385.6, 535.083],
    [0, 0, 1]
], dtype=np.float32)

K_inv = np.linalg.inv(K)
K_torch = torch.from_numpy(K).float().to(device)
K_inv_torch = torch.from_numpy(K_inv).float().to(device)

# CORRECTION: Tracker is mounted 88° CCW around X to get camera pointing along -z
correction_rot = R.from_euler('x', np.radians(0)).as_matrix()

def load_images_and_poses(json_path, image_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    imgs, poses = [], []
    for entry in data:
        fname = entry["filename"]
        img_path = os.path.join(image_dir, fname)
        if not os.path.exists(img_path):
            print(f"WARNING: Missing image {img_path}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        imgs.append(img)

        # Extract position
        pos = np.array([
            entry["position"]["x"],
            entry["position"]["y"],
            entry["position"]["z"]
        ])

        # Extract and combine rotation
        rot = np.eye(3)
        for r in entry["rotation_sequence"]:
            angle = r["angle_rad"]
            axis = r["axis"]
            rot_part = R.from_euler(axis.lower(), angle).as_matrix()
            rot = rot @ rot_part

        # Apply tracker mount correction (88° CCW around X)
        rot = rot @ correction_rot
        poses.append((rot, pos))

    return imgs, poses


def get_homography(R_wc, t_wc, K, plane_z):
    n = np.array([0, 0, 1])
    d = plane_z  # distance from origin to plane (z = -1.2)
    
    # camera to world → invert to get world to camera
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    H = K @ (R_cw - np.outer(t_cw, n) / d) @ K_inv
    return H / H[2,2]

def warp_image_torch(img, H, output_shape):
    h_out, w_out = output_shape
    H_inv = torch.from_numpy(np.linalg.inv(H)).float().to(device)

    yy, xx = torch.meshgrid(
        torch.arange(h_out, dtype=torch.float32, device=device),
        torch.arange(w_out, dtype=torch.float32, device=device),
        indexing='ij'
    )

    ones = torch.ones_like(xx)
    pix_coords = torch.stack((xx, yy, ones), dim=-1)  # (H, W, 3)
    pix_coords = pix_coords.view(-1, 3).T  # (3, N)

    world_coords = H_inv @ pix_coords
    world_coords = world_coords / world_coords[2:3].clone()

    u = world_coords[0].view(h_out, w_out) / img.shape[1] * 2 - 1
    v = world_coords[1].view(h_out, w_out) / img.shape[0] * 2 - 1
    grid = torch.stack((u, v), dim=-1).unsqueeze(0)

    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    warped = torch.nn.functional.grid_sample(img_tensor, grid, align_corners=True)
    return warped.squeeze().cpu().numpy()

def stitch_images(json_file, K, image_dir, plane_z=-1.2):
    imgs, poses = load_images_and_poses(json_file, image_dir)

    canvas_size = (2000, 2000)
    offset = np.array([1000, 1000])
    stitched = np.zeros(canvas_size, dtype=np.float32)
    count = np.zeros(canvas_size, dtype=np.float32)

    for img, (R, t) in tqdm(zip(imgs, poses), total=len(imgs), desc="Stitching"):
        H = get_homography(R, t, K, plane_z)

        warped = warp_image_torch(img, H, canvas_size)

        mask = (warped > 0).astype(np.float32)
        stitched += warped * mask
        count += mask

    stitched = np.divide(stitched, count, out=np.zeros_like(stitched), where=(count > 0))
    return (stitched * 255).astype(np.uint8)

# Example usage:
result = stitch_images("./data.json", K, "./images", plane_z=-1.2)
cv2.imwrite("stitched_result_final.png", result)
