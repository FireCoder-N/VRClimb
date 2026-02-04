import numpy as np
import cv2

# -----------------------------
# 1. Load your captured image
# -----------------------------
image = cv2.imread('C:/Users/Mike/Documents/N/14.cppstandalone/captures/rgb/1741173252.png')  # Replace with your image path
if image is None:
    raise ValueError("Image not found. Please check the path.")

# -----------------------------
# 2. Define camera intrinsics and extrinsics
# -----------------------------
# Example intrinsic parameters (adjust fx, fy, cx, cy as needed)
h, w = image.shape[:2]
fx = 1386.08  # focal length in pixels
fy = 1385.6
cx = 980.449
cy = 535.083
K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]])

# Example extrinsics: assume a rotation about the y-axis and a translation.
R = np.array([[ 0.9991388916969299, 0.04128009080886841, 0.0041617173701524734],
              [0.0059198252856731415, -0.0425599068403244, -0.9990763664245605],
              [-0.04106485843658447, 0.9982407093048096, -0.04276764392852783]])
t = np.array([-0.9851787090301514, 1.1147434711456299, 0.020821571350097656])  # camera position in world coordinates (x, y, z)

# Define the plane: z = c (for example, ground plane at z=0)
c = 1.2

# -----------------------------
# 3. Compute intersection of image rays with the plane
# -----------------------------
def intersect_with_plane(u, v):
    """ Given image pixel (u,v), compute (x,y) on plane z=c. """
    C = -R.T @ t
    # Convert pixel (u,v) to homogeneous coordinates
    pixel = np.array([u, v, 1])
    # Backproject into normalized camera coordinates
    p_cam = np.linalg.inv(K) @ pixel
    # Transform ray direction to world coordinates
    d = R.T @ p_cam
    # Camera center in world coordinates is t.
    # Solve for lambda in: t_z + lambda * d_z = c
    lambda_val = (c - C[2]) / d[2]
    # Intersection point in world coordinates
    X_world = C + lambda_val * d
    return X_world[:2]  # Return the (x,y) coordinates on the plane

# Define the four image corners: top-left, top-right, bottom-right, bottom-left
corners_img = np.array([[0,    0],
                        [w-1,  0],
                        [w-1, h-1],
                        [0,   h-1]], dtype=np.float32)

# Compute their corresponding positions on the plane
corners_plane = np.array([intersect_with_plane(u, v) for u, v in corners_img], dtype=np.float32)

# -----------------------------
# 4. Define a canvas and map world coordinates to canvas pixel coordinates
# -----------------------------
# Find bounding box of the projected corners to define canvas extents
min_xy = corners_plane.min(axis=0)
max_xy = corners_plane.max(axis=0)
margin = 50  # extra border in pixels

canvas_width = int(max_xy[0] - min_xy[0] + 2 * margin)
canvas_height = int(max_xy[1] - min_xy[1] + 2 * margin)

canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

def world_to_canvas(pt):
    """
    Map a point in world (plane) coordinates to canvas pixel coordinates.
    This simple mapping shifts the points so that the minimum point is at (margin, margin).
    """
    x_canvas = pt[0] - min_xy[0] + margin
    y_canvas = pt[1] - min_xy[1] + margin
    return int(round(x_canvas)), int(round(y_canvas))

colors = [(0,0,255),
          (0,255,0),
          (255,0,0),
          (255,255,0)]

for i, pt in enumerate(corners_plane):
    canvas_pt = world_to_canvas(pt)
    cv2.circle(canvas, canvas_pt, radius=5, color=colors[i], thickness=-1)

pts = np.array([world_to_canvas(pt) for pt in corners_plane], dtype=np.int32)
cv2.polylines(canvas, [pts], isClosed=True, color=(0,255,255), thickness=1)

# # Map the four plane corners to canvas coordinates
# dst_corners = np.array([world_to_canvas(pt) for pt in corners_plane], dtype=np.float32)

# # -----------------------------
# # 5. Compute homography and warp the image onto the canvas
# # -----------------------------
# # Compute homography from image coordinates to canvas coordinates
# H, _ = cv2.findHomography(corners_img, dst_corners)

# # Create a black canvas (representing the infinite plane)
# canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# # Warp the image using the computed homography
# warped_image = cv2.warpPerspective(image, H, (canvas_width, canvas_height))

# # Combine the warped image onto the canvas (in this simple case, the warped image replaces the black area)
# canvas = cv2.add(canvas, warped_image)

# -----------------------------
# 6. Display the result
# -----------------------------
cv2.imshow('Projected Image on Plane z=c', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()