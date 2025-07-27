import pyzed.sl as sl
import cv2
import numpy as np
import open3d as o3d
import os

# === Settings ===
save_dir = "captures"
os.makedirs(save_dir, exist_ok=True)

# === Initialize ZED ===
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD1080
init.depth_mode = sl.DEPTH_MODE.ULTRA
init.coordinate_units = sl.UNIT.METER
zed = sl.Camera()
if zed.open(init) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED")
    exit(1)

runtime = sl.RuntimeParameters()
image = sl.Mat()
depth = sl.Mat()
point_cloud = sl.Mat()

frame_id = 0
print("[INFO] Press any key in the OpenCV window to capture a frame. Press ESC to finish.")
while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        img = image.get_data()
        img_display = cv2.resize(img, (960, 540))
        cv2.imshow("ZED Capture", img_display)
        key = cv2.waitKey(10)

        if key == 27:  # ESC to exit
            break
        elif key != -1:
            rgb_path = os.path.join(save_dir, f"rgb_{frame_id:03d}.png")
            pcd_path = os.path.join(save_dir, f"pcd_{frame_id:03d}.ply")

            # Save RGB image
            cv2.imwrite(rgb_path, img)

            # Convert point cloud to Open3D format
            pc_np = point_cloud.get_data()[:, :, :3]  # Get XYZ
            valid = np.isfinite(pc_np).all(axis=2)
            xyz = pc_np[valid]
            colors = image.get_data()[..., :3][valid] / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            o3d.io.write_point_cloud(pcd_path, pcd)
            print(f"[CAPTURED] Frame {frame_id} saved: {rgb_path}, {pcd_path}")

            # Visualize the point cloud
            # print(f"[INFO] Displaying point cloud {frame_id}...")
            # o3d.visualization.draw_geometries([pcd], window_name=f"PointCloud {frame_id}")

            frame_id += 1

zed.close()
cv2.destroyAllWindows()