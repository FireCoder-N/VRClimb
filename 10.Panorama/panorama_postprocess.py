import cv2
import numpy as np

# Load the stitched panorama
panorama = cv2.imread("output/panorama_rgb.png")
if panorama is None:
    raise FileNotFoundError("Could not load 'panorama_rgb.png'.")

h_pan, w_pan = panorama.shape[:2]

# Step 1: Create mask of non-black pixels
gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Step 2: Detect corner points on mask using Harris corner detector
# Convert mask to float32 for cornerHarris
mask_float = np.float32(mask)
harris = cv2.cornerHarris(mask_float, blockSize=2, ksize=3, k=0.04)
harris_dilated = cv2.dilate(harris, None)

# Threshold to get strong corners
corners = np.argwhere(harris_dilated > 0.02 * harris_dilated.max())
# corners are in (y, x) form; convert to (x, y)
corners = np.array([[x, y] for y, x in corners])

# Step 3: Classify corners into quadrants relative to image center
cx, cy = w_pan / 2, h_pan / 2
quadrants = {"tl": [], "tr": [], "br": [], "bl": []}
for (x, y) in corners:
    if x < cx and y < cy:
        quadrants["tl"].append((x, y))
    elif x >= cx and y < cy:
        quadrants["tr"].append((x, y))
    elif x >= cx and y >= cy:
        quadrants["br"].append((x, y))
    else:
        quadrants["bl"].append((x, y))

# Helper to pick the most extreme corner in each quadrant
def pick_extreme(points, quadrant):
    if not points:
        return None
    pts = np.array(points)
    if quadrant == "tl":
        # top-left: minimal x+y
        vals = pts[:,0] + pts[:,1]
        return tuple(pts[np.argmin(vals)])
    if quadrant == "tr":
        # top-right: minimal (w - x) + y
        vals = (w_pan - pts[:,0]) + pts[:,1]
        return tuple(pts[np.argmin(vals)])
    if quadrant == "br":
        # bottom-right: minimal (w - x) + (h - y)
        vals = (w_pan - pts[:,0]) + (h_pan - pts[:,1])
        return tuple(pts[np.argmin(vals)])
    if quadrant == "bl":
        # bottom-left: minimal x + (h - y)
        vals = pts[:,0] + (h_pan - pts[:,1])
        return tuple(pts[np.argmin(vals)])
    return None

corner_tl = pick_extreme(quadrants["tl"], "tl")
corner_tr = pick_extreme(quadrants["tr"], "tr")
corner_br = pick_extreme(quadrants["br"], "br")
corner_bl = pick_extreme(quadrants["bl"], "bl")

quad_src = np.array([corner_tl, corner_tr, corner_br, corner_bl], dtype=np.float32)

# Step 4: Compute dimensions for destination rectangle
(tl, tr, br, bl) = quad_src
widthA  = np.linalg.norm(br - bl)
widthB  = np.linalg.norm(tr - tl)
maxWidth  = int(max(widthA, widthB))

heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxHeight = int(max(heightA, heightB))

quad_dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype=np.float32)

# Step 5: Compute homography and warp
H_flat = cv2.getPerspectiveTransform(quad_src, quad_dst)
flattened = cv2.warpPerspective(panorama, H_flat, (maxWidth, maxHeight))

# save the flattened image
cv2.imwrite("output/pf.png", flattened)

# Step 6: Warp depthmap(s) accordingly
depth_min = np.load("output/panorama_depth_min.npy")  # float32
flattened_depth_min = cv2.warpPerspective(
    depth_min,
    H_flat,  # from the flatten script
    (flattened.shape[1], flattened.shape[0]),  # same as flattened RGB size
    flags=cv2.INTER_NEAREST
)
np.save("output/pdmf.npy", flattened_depth_min)

depth_avg = np.load("output/panorama_depth_avg.npy")
flattened_depth_avg = cv2.warpPerspective(
    depth_avg,
    H_flat,  # from the flatten script
    (flattened.shape[1], flattened.shape[0]),  # same as flattened RGB size
    flags=cv2.INTER_NEAREST
)
np.save("output/pdmf.npy", flattened_depth_min)
np.save("output/pdaf.npy", flattened_depth_avg)
