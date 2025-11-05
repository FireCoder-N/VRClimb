import pyzed.sl as sl
import numpy as np
import cv2  # For visualizing depth mask

def get_depth_map(zed, runtime_params):
    depth_map = sl.Mat()
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        depth_data = depth_map.get_data()

        if depth_data is None:
            print("Depth map capture failed.")
            return None

        # Filter out NaN values
        # depth_data[np.isnan(depth_data)] = 255 #
        depth_data = depth_data[~np.isnan(depth_data)]
        
        if len(depth_data) == 0:
            print("Depth map is completely NaN.")
            return None
        
        print(f"Depth map captured. Min: {np.min(depth_data)}, Max: {np.max(depth_data)}")
        return depth_data
    else:
        print("Failed to grab depth image.")
        return None

def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Set up the ZED camera configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY  # High quality depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Depth in meters

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Unable to open ZED camera")
        exit()

    runtime_params = sl.RuntimeParameters()
    runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD

    # Capture the flat wall depth map for reference
    print("Capturing flat wall for reference...")
    wall_depth_map = get_depth_map(zed, runtime_params)
    if wall_depth_map is None:
        print("Error: Wall depth map is None.")
        zed.close()
        return

    # Capture the grips scene depth map
    print("Capturing scene with grips...")
    grips_depth_map = get_depth_map(zed, runtime_params)
    if grips_depth_map is None:
        print("Error: Grips depth map is None.")
        zed.close()
        return

    # Compute the absolute depth difference
    depth_difference = np.abs(grips_depth_map) # - wall_depth_map)

    # Filter out NaNs and apply a threshold to visualize
    depth_difference = np.nan_to_num(depth_difference, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize depth difference for visualization
    if np.max(depth_difference) > 0:
        depth_mask = (depth_difference / np.max(depth_difference)) * 255  # Scale to [0, 255]
    else:
        print("No valid depth differences detected.")
        zed.close()
        return

    depth_mask = depth_mask.astype(np.uint8)

    # Display the depth mask
    cv2.imshow("Depth Mask", depth_mask)
    print("Displaying depth mask. Press any key to close.")
    
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

    zed.close()

if __name__ == "__main__":
    main()
