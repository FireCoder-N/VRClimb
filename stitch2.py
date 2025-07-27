import cv2
import numpy as np
import os
import open3d as o3d
import pyzed.sl as sl
import PointcloudCrop as pcc

def detect_and_match_features(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    return np.float32(pts1), np.float32(pts2)

def compute_homographies(images):
    transforms = [np.eye(3)]
    for i in range(1, len(images)):
        pts1, pts2 = detect_and_match_features(images[i - 1], images[i])
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
        cumulative_H = transforms[i - 1] @ H
        transforms.append(cumulative_H)
    return transforms

def warp_images(images, transforms):
    # Determine output panorama size
    sizes = [img.shape[:2] for img in images]
    corners = []
    for img, H in zip(images, transforms):
        h, w = img.shape[:2]
        pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
        warped_pts = cv2.perspectiveTransform(pts, H)
        corners.append(warped_pts)

    all_pts = np.concatenate(corners, axis=0)
    [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

    panorama = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=np.uint8)
    for img, H in zip(images, transforms):
        warp_matrix = translation @ H
        warped = cv2.warpPerspective(img, warp_matrix, (panorama.shape[1], panorama.shape[0]))
        mask = (warped > 0)
        panorama[mask] = warped[mask]
    return panorama, translation

def save_transforms(transforms, output_dir="transforms"):
    os.makedirs(output_dir, exist_ok=True)
    for i, H in enumerate(transforms):
        np.save(os.path.join(output_dir, f"transform_{i}.npy"), H)

def merge_pcd(transforms, pcd_paths, K):
    merged_pcd = o3d.geometry.PointCloud()
    assert len(transforms) == len(pcd_paths) 
    for i, path in enumerate(pcd_paths):
        pcd = o3d.io.read_point_cloud(path)
        _, Rs, Ts, _ = cv2.decomposeHomographyMat(transforms[i], K)
        T = np.eye(4)
        ind = 0
        T[:3, :3] = Rs[ind]
        T[:3, 3] = Ts[ind].ravel()
        # result = refine_registration(merged_pcd, pcd, T)

        pcd.transform(T)
        merged_pcd += pcd

    return merged_pcd

def refine_registration(source, target, initial_transformation, distance_threshold=0.1):
    """Local ICP alignment for refinement."""
    print("[INFO] Refining with ICP...")
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=20))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=20))
    
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20, relative_fitness=1e-6, relative_rmse=1e-6)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold,
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria)
    return result

def cleanup(pcd):
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    cleaned_pcd = pcd.select_by_index(ind)

    # Compute the center of the point cloud
    center = np.mean(np.asarray(cleaned_pcd.points), axis=0)

    # Define the target camera view (this could be any vector you want the point cloud to face)
    camera_direction = np.array([0, 0, 1])

    # Vector from point cloud center to the camera direction
    view_direction = camera_direction - center

    # Compute the rotation matrix to align the point cloud's normal with the camera direction
    # First, we normalize the vectors
    view_direction = view_direction / np.linalg.norm(view_direction)

    # Get the transformation matrix (this is a rough approach, assuming you want the main normal to face the camera)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0])

    # Apply the transformation
    cleaned_pcd.rotate(rotation_matrix, center=center)

    # cleaned_pcd.transform([[1, 0, 0, 0],
    #                       [0, -1, 0, 0],
    #                       [0, 0, -1, 0],
    #                       [0, 0, 0, 1 ]])

    return cleaned_pcd

def crop_point_cloud_with_2d_box(pcd):
    cropper = pcc.PointCloudCropWindow(pointcloud=pcd, frame_aspect_ratio=1)
    cropped_pcd = cropper.run()
    return cropped_pcd


#############################################################################################
if __name__ == "__main__":
    data_dir = "captures"

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera for intrinsics.")
        exit(1)

    camera_info = zed.get_camera_information()
    width = camera_info.camera_resolution.width
    height = camera_info.camera_resolution.height
    fx = camera_info.calibration_parameters.left_cam.fx
    fy = camera_info.calibration_parameters.left_cam.fy
    cx = camera_info.calibration_parameters.left_cam.cx
    cy = camera_info.calibration_parameters.left_cam.cy
    zed.close()

    o3_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    K = o3_intrinsics.intrinsic_matrix
    zed.close()

    image_paths = sorted([data_dir + "\\" + f for f in os.listdir(data_dir) if f.startswith("rgb") and f.endswith(".png")])
    pcd_paths = sorted([data_dir + "\\" + f for f in os.listdir(data_dir) if f.startswith("pcd") and f.endswith(".ply")])
    images = [cv2.imread(p) for p in image_paths]

    print("Computing transforms...")
    transforms = compute_homographies(images)

    print("Warping images...")
    panorama, _ = warp_images(images, transforms)

    print("Saving panorama and transforms...")
    cv2.imwrite("panorama.png", panorama)

    print("merge pointclouds")
    merged_pcd = merge_pcd(transforms, pcd_paths, K)

    print("cleanup view and loose vertices")
    cleaned_pcd = cleanup(merged_pcd)
    
    cropped_pcd = crop_point_cloud_with_2d_box(cleaned_pcd)

    print("Estimating normals for clean point cloud...")
    cropped_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    print("Performing Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cropped_pcd, depth=9)

    # Filter low-density vertices
    densities = np.asarray(densities)
    threshold = np.percentile(densities, 5)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # o3d.io.write_point_cloud("stitched_output.ply", cropped_pcd)
    o3d.visualization.draw_geometries([mesh], window_name="Cleaned PointCloud")
