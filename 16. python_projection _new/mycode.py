import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

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
                r = R.from_rotvec([angle-90, 0, 0])
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

def warp_images_to_plane(json_file, K, output_size, z_plane=1.2):
    """
    Warps all images from the JSON file onto the given plane (z = 1.2).
    """
    camera_data = load_camera_info(json_file)
    stitched = np.zeros(output_size, dtype=np.uint8)
    path = "images/"

    for cam in tqdm(camera_data):
        img = cv2.imread(path + cam["filename"])
        if img is None:
            print(f"Error: Could not load {cam['filename']}")
            continue

        h, w, _ = img.shape
        warped_img = np.zeros(output_size, dtype=np.uint8)

        for v in tqdm(range(h)):
            for u in range(w):
                # Backproject pixel to world ray
                ray_origin, ray_direction = backproject_pixel(K, cam["R"], cam["t"], (u, v))

                # Project onto plane z = 1.2
                projected_point = project_to_plane(ray_origin, ray_direction, z_plane)
                if projected_point is None:
                    continue

                # Convert world (X, Y) to output image coordinates
                px, py = int(projected_point[0] * 200 + output_size[1] // 2), int(projected_point[1] * 200 + output_size[0] // 2)

                if 0 <= px < output_size[1] and 0 <= py < output_size[0]:
                    warped_img[py, px] = img[v, u]

        # Merge into stitched image
        mask = (warped_img > 0).astype(np.uint8)
        stitched = np.where(mask, warped_img, stitched)
        # break

    return stitched

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
    result = warp_images_to_plane(json_file, K, output_size, z_plane=1.2)

    cv2.imwrite("stitched_result.jpg", result)