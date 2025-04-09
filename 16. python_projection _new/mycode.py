import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import cupy as cp
from math import pi

def load_camera_info(json_file):
    """Loads camera parameters from a JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)

    camera_info = []

    for entry in data:
        filename = entry["filename"]

        # Translation vector (t)
        t = np.array([[entry["position"]["x"]],
                      [entry["position"]["y"]],
                      [entry["position"]["z"]]])

        # Compute Rotation Matrix (R) from axis-angle representation
        R_mat = np.eye(3)
        for rot in entry["rotation_sequence"]:
            angle = rot["angle_rad"]
            axis = rot["axis"].upper()
            if axis == "X":
                r = R.from_rotvec([pi/2-angle, 0, 0])
            elif axis == "Y":
                r = R.from_rotvec([0, angle, 0])
            elif axis == "Z":
                r = R.from_rotvec([0, 0, angle])
            R_mat = r.as_matrix() @ R_mat  # Apply rotation in sequence

        camera_info.append({"filename": filename, "R": R_mat, "t": t})

    return camera_info

def backproject_pixel(K, R, t, pixel):
    """Backprojects a pixel (u, v) into a 3D ray in world coordinates."""
    u, v = pixel
    pixel_h = np.array([u, v, 1.0])  # Homogeneous pixel coordinate

    # Convert pixel to camera space
    K_inv = np.linalg.inv(K)
    cam_dir = K_inv @ pixel_h

    # Transform to world coordinates
    ray_direction = R.T @ cam_dir
    ray_direction /= np.linalg.norm(ray_direction)  # Normalize

    # Compute camera center in world space
    ray_origin = -R.T @ t

    return ray_origin, ray_direction

def project_pixels_cuda(K, R, t, z_plane, width, height):
    """CUDA-accelerated projection of all pixels onto plane"""
    K = cp.array(K)
    R = cp.array(R)
    t = cp.array(t)
    u, v = cp.meshgrid(cp.arange(width), cp.arange(height))
    pixels = cp.stack([u.ravel(), v.ravel(), cp.ones(u.size)], axis=-1)

    # Convert pixel to camera space
    K_inv = cp.linalg.inv(K)
    cam_dirs = (K_inv @ pixels.T).T

    # Transform to world coordinates
    R_inv = R.T
    ray_dirs = (R_inv @ cam_dirs.T).T
    ray_dirs /= cp.linalg.norm(ray_dirs, axis=1, keepdims=True)  # Normalize

    # Compute camera center in world space
    ray_origin = -R_inv @ t
    ray_origin = cp.tile(ray_origin.T, (pixels.shape[0], 1))

    dz = ray_dirs[:,2]
    valid = cp.abs(dz) > 1e-6

    lam = cp.where(valid, (z_plane - ray_origin[:,2]) / dz, cp.nan)
    X_proj = ray_origin[:,0] + lam*ray_dirs[:,0]
    Y_proj = ray_origin[:,1] + lam*ray_dirs[:,1]

    # print("X_proj min/max", np.min(X_proj), np.max(X_proj))
    # print("Y_proj min/max", np.min(Y_proj), np.max(Y_proj))

    return X_proj.get(), Y_proj.get()

def poisson_blending(stitched, warped, mask, center):
    return cv2.seamlessClone(warped, stitched, mask, center, cv2.NORMAL_CLONE)

def normalize_positions(positions, output_size):
    x_min, x_max = -2, 2
    z_min, z_max = -1, 2

    positions[:,0] = (positions[:,0] - x_min) / (x_max-x_min)
    positions[:,1] = (positions[:,1] - z_min) / (z_max-z_min)

    positions[:,0] = positions[:,0] * output_size[1]
    positions[:,1] = positions[:,1] * output_size[0]

    positions = positions.astype(np.int32)
    return positions

def project_to_plane(ray_origin, ray_direction, z_plane=1.2):
    """Projects the 3D ray onto the plane z = z_plane."""
    o_x, o_y, o_z = ray_origin.flatten()
    d_x, d_y, d_z = ray_direction.flatten()

    if abs(d_z) < 1e-6:  # Prevent division by zero if ray is parallel to plane
        return None

    # Compute lambda
    lam = (z_plane - o_z) / d_z

    # Compute projected (X, Y, 1.2)
    X_proj = o_x + lam * d_x
    Y_proj = o_y + lam * d_y

    return np.array([X_proj, Y_proj, z_plane])

def warp_images_new(json_file, K, output_size, z_plane=1.2):
    camera_data = load_camera_info(json_file)
    path = "images/"

    all_X, all_Y = [], []

    for cam in camera_data:
        img = cv2.imread(path + cam["filename"])
        if img is None:
            print(f"Error: Could not load {cam['filename']}")
            continue

        h, w, _ = img.shape
        R, t = cam["R"], cam["t"]

        X_proj, Y_proj = project_pixels_cuda(K, cam["R"], cam["t"], z_plane, w, h)
        all_X.append(X_proj)
        all_Y.append(Y_proj)

    all_X = np.concatenate(all_X)
    all_Y = np.concatenate(all_Y)

    scale_x = output_size[1] / (all_X.max() - all_X.min())
    scale_y = output_size[0] / (all_Y.max() - all_Y.min())
    scale = min(scale_x, scale_y)

    offset_x = -all_X.min() * scale
    offset_y = -all_Y.min() * scale

    stiched_float = np.zeros(output_size, dtype=np.float32)
    weight_map = np.zeros(output_size[:2], dtype=np.float32)

    for cam in tqdm(camera_data):
        img = cv2.imread(path + cam["filename"])
        if img is None:
            continue

        h, w, _ = img.shape
        X_proj, Y_proj = project_pixels_cuda(K, cam["R"], cam["t"], z_plane, w, h)

        X_proj_img = np.round(X_proj * scale + offset_x).astype(int)
        Y_proj_img = np.round(Y_proj * scale + offset_y).astype(int)

        for i, (px,py) in enumerate(zip(X_proj_img, Y_proj_img)):
            if (0 <= px <= output_size[1] -1 ) and (0 <= py <= output_size[0] -1):
                stiched_float[py,px] += img[i // w, i % w]
                weight_map[py,px] += 1

    weight_map[weight_map == 0] = 1
    stitched = (stiched_float /weight_map[..., None]).astype(np.uint8)
    return stitched


def warp_images_to_plane(json_file, K, output_size, z_plane=1.2):
    """
    Warps all images from the JSON file onto the given plane (z = 1.2).
    """
    camera_data = load_camera_info(json_file)
    stitched = np.zeros(output_size, dtype=np.uint8)
    path = "images/"

    scale = 200
    offset_x = 0 # output_size[1] // 2
    offset_y = output_size[0] // 2

    for cam in tqdm(camera_data):
        img = cv2.imread(path + cam["filename"])
        if img is None:
            print(f"Error: Could not load {cam['filename']}")
            continue

        h, w, _ = img.shape
        R, t = cam["R"], cam["t"]

        X_proj, Y_proj = project_pixels_cuda(K, R, t, z_plane, w, h)
        
        # Convert projected world coordinates to image coordinates
        X_proj_img = np.round(X_proj * scale + offset_x).astype(int)
        Y_proj_img = np.round(Y_proj * scale + offset_y).astype(int)

        # Map each pixel from the original image to the stitched output image
        for i, (px, py) in enumerate(zip(X_proj_img, Y_proj_img)):
            # Use explicit boundary check to ensure indices are in range
            if (0 <= px <= output_size[1] - 1) and (0 <= py <= output_size[0] - 1):
                stitched[py, px] = img[i // w, i % w]

    return stitched

    #     for v in tqdm(range(h)):
    #         for u in range(w):
    #             # Backproject pixel to world ray
    #             ray_origin, ray_direction = backproject_pixel(K, cam["R"], cam["t"], (u, v))

    #             # Project onto plane z = 1.2
    #             projected_point = project_to_plane(ray_origin, ray_direction, z_plane)
    #             if projected_point is None:
    #                 continue

    #             # Convert world (X, Y) to output image coordinates
    #             px, py = int(projected_point[0] * 200 + output_size[1] // 2), int(projected_point[1] * 200 + output_size[0] // 2)

    #             if 0 <= px < output_size[1] and 0 <= py < output_size[0]:
    #                 warped_img[py, px] = img[v, u]

    #     # Merge into stitched image
    #     mask = (warped_img > 0).astype(np.uint8)
    #     stitched = np.where(mask, warped_img, stitched)
    #     # break

    # return stitched

# --- Example Usage ---
if __name__ == "__main__":
    json_file = "data.json"  # Path to JSON file

    # Intrinsic Matrix
    K = np.array([
        [1386.08, 0, 980.449],
        [0, 1385.6, 535.083],
        [0, 0, 1]
    ])

    output_size = (2000, 3000, 3)  # Define final image size
    result = warp_images_new(json_file, K, output_size, z_plane=-1.2)

    cv2.imwrite("stitched_result2.jpg", result)