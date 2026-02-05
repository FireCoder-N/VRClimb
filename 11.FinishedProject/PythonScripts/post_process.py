import cv2
import numpy as np

# Helper to pick the most extreme corner in each quadrant
def pick_extreme(points, quadrant, h_pan, w_pan):
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

def flatten(panorama_path="C:/Users/Mike/Documents/9.Scene/MyProject/WallAssets/panorama.png"):
    panorama = cv2.imread(panorama_path)
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

    corner_tl = pick_extreme(quadrants["tl"], "tl", h_pan, w_pan)
    corner_tr = pick_extreme(quadrants["tr"], "tr", h_pan, w_pan)
    corner_br = pick_extreme(quadrants["br"], "br", h_pan, w_pan)
    corner_bl = pick_extreme(quadrants["bl"], "bl", h_pan, w_pan)

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
    cv2.imwrite(panorama_path, flattened)

if __name__ == "__main__":
    flatten("C:/Users/Mike/Documents/9.Scene/MyProject/WallAssets/panorama.png")
