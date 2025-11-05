import cv2
import numpy as np
import pyzed.sl as sl
import open3d as o3d

##############################################################
#                          UTILITIES
##############################################################
def preprocess_point_cloud(pcd, voxel_size):
    """Downsample, estimate normals, and compute FPFH features."""
    print("[INFO] Preprocessing point cloud...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """Global alignment using FPFH features and RANSAC."""
    print("[INFO] Performing RANSAC global registration...")
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def refine_registration(source, target, initial_transformation, voxel_size):
    """Local ICP alignment for refinement."""
    print("[INFO] Refining with ICP...")
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold,
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


##############################################################
#                       INITIALIZATIONS
##############################################################
# Initialize ZED camera
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30
init_params.sdk_verbose = 1

runtime_parameters = sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera.")
    exit(1)

# Enable positional tracking
tracking_params = sl.PositionalTrackingParameters()
zed.enable_positional_tracking(tracking_params)

# Get intrinsics
cam_info = zed.get_camera_information()
width = cam_info.camera_resolution.width
height = cam_info.camera_resolution.height
left_cam = cam_info.calibration_parameters.left_cam
fx, fy, cx, cy = left_cam.fx, left_cam.fy, left_cam.cx, left_cam.cy
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
print(f"Using intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

combined_pcd = o3d.geometry.PointCloud()
image_color = sl.Mat()
image_depth = sl.Mat()
pose = sl.Pose()


##############################################################
#                          MAINLOOP
##############################################################
frame_count = 0
while True:
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        zed.retrieve_image(image_color, sl.VIEW.LEFT)
        img = image_color.get_data()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Show preview
        cv2.namedWindow("ZED Live View - Press Any Key to Capture, ESC to Exit", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ZED Live View - Press Any Key to Capture, ESC to Exit", 800, 600)
        cv2.imshow("ZED Live View - Press Any Key to Capture, ESC to Exit", img_rgb)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break
        elif key != -1:
            print(f"\n==================== Frame {frame_count + 1} ====================")
            try:
                zed.retrieve_image(image_color, sl.VIEW.LEFT)
                zed.retrieve_measure(image_depth, sl.MEASURE.DEPTH)
                color_np = image_color.get_data()
                depth_np = image_depth.get_data()
                color_np = cv2.cvtColor(color_np, cv2.COLOR_BGRA2RGB)

                o3d_color = o3d.geometry.Image(color_np.astype(np.uint8))
                o3d_depth = o3d.geometry.Image(depth_np.astype(np.float32))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color, o3d_depth,
                    depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
                )

                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
                pcd.transform([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])

                voxel_size = 0.05  # 5 cm

                if frame_count == 0:
                    print("[INFO] First frame — initializing combined point cloud.")
                    combined_pcd = pcd
                    combined_down, combined_fpfh = preprocess_point_cloud(combined_pcd, voxel_size)
                    frame_count += 1
                else:
                    source_pcd = pcd
                    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)

                    result_ransac = execute_global_registration(
                        source_down, combined_down,
                        source_fpfh, combined_fpfh,
                        voxel_size)

                    if result_ransac.fitness > 0.1:
                        print(f"[INFO] RANSAC succeeded. Fitness = {result_ransac.fitness:.4f}")
                        # source_pcd.estimate_normals(
                        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=50))
                        # combined_pcd.estimate_normals(
                        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))

                        # result_icp = refine_registration(
                        #     source_pcd, combined_pcd,
                        #     result_ransac.transformation, voxel_size)
                        
                        # print(f"[INFO] ICP refinement complete. Fitness = {result_icp.fitness:.4f}")
                        source_pcd.transform(result_ransac.transformation)
                        combined_pcd += source_pcd

                        # Update reference for next alignment
                        combined_down, combined_fpfh = preprocess_point_cloud(combined_pcd, voxel_size)
                        frame_count += 1
                    else:
                        print("[WARN] RANSAC failed to find a valid alignment (fitness too low).")

            except Exception as e:
                print(f"[ERROR] Failed to process frame {frame_count + 1}: {e}")




# Done capturing
cv2.destroyAllWindows()
zed.disable_positional_tracking()
zed.close()

if frame_count == 0:
    print("No frames captured. Exiting.")
    exit(0)

# Estimate normals
print("Estimating normals for combined point cloud...")
combined_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

# Poisson reconstruction
print("Performing Poisson surface reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(combined_pcd, depth=9)

# Filter low-density vertices
densities = np.asarray(densities)
threshold = np.percentile(densities, 5)
vertices_to_remove = densities < threshold
mesh.remove_vertices_by_mask(vertices_to_remove)

# Save and show
o3d.io.write_triangle_mesh("multi_frame_mesh.obj", mesh)
print("Mesh saved to 'multi_frame_mesh.obj'.")

o3d.visualization.draw_geometries([mesh], window_name="Reconstructed Mesh", width=960, height=540)

