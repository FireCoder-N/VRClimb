import cv2
import numpy as np
import pyzed.sl as sl
import open3d as o3d

def view_pointcloud(point_cloud_np):
    point_cloud_np = point_cloud_np[:, :, :3]  # X, Y, Z

    # Reshape into a list of 3D points (N x 3 array)
    points = point_cloud_np.reshape((-1, 3))

    # Filter out points with no depth (Z value is zero or negative)
    points = points[points[:, 2] > 0]

    # Step 1: Extract the z coordinates
    z_coords = points[:, 2]

    # Step 2: Calculate the mean of the z coordinates
    z_mean = np.mean(z_coords)

    # Step 3: Define a threshold to clip outlier z values (tune this value to fit your needs)
    threshold = 0.2  # You can adjust the threshold based on how far from the mean is considered "off"

    # Step 4: Clip values outside the threshold (both above and below the threshold)
    # If the absolute difference from the mean is larger than the threshold, replace with z_mean
    z_coords_clipped = np.where(np.abs(z_coords - z_mean) > threshold, z_mean, z_coords)

    # Step 5: Update the point cloud with the modified z coordinates
    points[:, 2] = z_coords_clipped

    # Convert to Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud using Open3D
    o3d.visualization.draw_geometries([pcd])



zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER
init_params.sdk_verbose = 0
init_params.depth_maximum_distance = 4
# init_params.depth_minimum_distance = 5 # Set the minimum depth perception distance to 15cm

runtime_parameters =sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Create an RGBA sl.Mat object
image_zed = sl.Mat() #zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
depth_zed = sl.Mat() #zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.F32_C1)
pointcloud_zed = sl.Mat()


if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
    # ---- get image ----
    # # Retrieve the left image in sl.Mat
    # zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    # # Use get_data() to get the numpy array
    # ocv = image_zed.get_data()

    # ---- get depth ----
    # Retrieve depth data (32-bit)
    zed.retrieve_image(image_zed, sl.VIEW.DEPTH)
    # Load depth data into a numpy array
    ocv = image_zed.get_data()
    ocv = cv2.normalize(ocv, ocv, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    #  ---- display image ----
    cv2.imshow("Image", ocv)
    # cv2.imwrite("depthmask.png", ocv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # ---- measure depth ----
    # Retrieve depth data (32-bit)
    zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
    # Load depth data into a numpy array
    ocv = image_zed.get_data()
    
    # print(ocv[int(len(ocv)/2)][int(len(ocv[0])/2)])

    # ---- get poincloud ----
    zed.retrieve_measure(pointcloud_zed, sl.MEASURE.XYZRGBA) # Retrieve colored point cloud
    point_cloud_np = pointcloud_zed.get_data()
    view_pointcloud(point_cloud_np)

    x = round(image_zed.get_width() / 2)
    y = round(image_zed.get_height() / 2)
    err, depth_value = depth_zed.get_value(x, y)
    print(f"Distance to Camera at ({x}, {y}): {depth_value} m")

zed.close()

