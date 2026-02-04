import json
import numpy as np
import open3d as o3d
import imageio.v2 as imageio
import os
from skimage.transform import resize

def load_scene(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data["images"], np.array(data["intrinsics"])

def unproject(depth, K):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack((i, j, np.ones_like(i)), axis=-1).reshape(-1, 3).T  # 3 x N
    depths = depth.reshape(-1)
    K_inv = np.linalg.inv(K)
    cam_points = (K_inv @ pixels) * depths  # 3 x N
    return cam_points.T

def backproject_all(images_dict, K, image_dir="."):
    all_points = []
    all_colors = []

    for filename, extrinsics in images_dict.items():
        rgb_path = os.path.join(image_dir, filename)
        depth_path = os.path.join(image_dir, filename.replace("/rgb", "/depth"))

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"Missing files for {filename}, skipping...")
            continue

        rgb = imageio.imread(rgb_path) / 255.0
        depth = imageio.imread(depth_path).astype(np.float32) / 1000.0  # mm to meters

        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)

        # Match RGB to depth resolution
        if rgb.shape[:2] != depth.shape:
            rgb = resize(rgb, (*depth.shape, 3), preserve_range=True, anti_aliasing=True).astype(np.float32)

        R = np.array(extrinsics["R"])
        t = np.array(extrinsics["t"]).reshape(3, 1)

        valid = depth > 0
        cam_points = unproject(depth, K)
        cam_points = cam_points[valid.flatten()]

        world_points = (R.T @ (cam_points.T - t)).T
        all_points.append(world_points)

        rgb_vals = rgb.reshape(-1, 3)[valid.flatten()]
        all_colors.append(rgb_vals)

    return np.vstack(all_points), np.vstack(all_colors)

def visualize(points, colors):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pc])

if __name__ == "__main__":
    json_path = "C:/Users/Mike/Documents/N/14.cppstandalone/captures/data.json"
    images_dict, K = load_scene(json_path)
    points, colors = backproject_all(images_dict, K, "C:/Users/Mike/Documents/N/14.cppstandalone/captures/rgb")
    visualize(points, colors)
