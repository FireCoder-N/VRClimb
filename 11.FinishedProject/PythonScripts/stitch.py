import os
import glob
import numpy as np
import cv2
from collections import defaultdict, deque


# ------------------------------------------------------------
# 1. HELPER FUNCTIONS
# ------------------------------------------------------------

def parse_rc_from_name(basename: str):
    """
    Given a basename like 'r02_c05', return (r, c) = (2, 5) as integers.
    If parsing fails, raise ValueError.
    """
    if not (basename.startswith("r") and "_c" in basename):
        raise ValueError(f"Cannot parse (r,c) from '{basename}'")
    try:
        r_part, c_part = basename.split("_c")
        r = int(r_part[1:])
        c = int(c_part)
        return r, c
    except Exception as e:
        raise ValueError(f"Cannot parse (r,c) from '{basename}'") from e

def find_images(rgb_dir: str):
    """
    Scan rgb_dir for all rXX_cYY.png
    Returns a list of tuples: (rgb_path, basename, (r,c)).
    """
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "r[0-9][0-9]_c[0-9][0-9].png")))
    images = []
    for rgb_path in rgb_paths:
        base = os.path.splitext(os.path.basename(rgb_path))[0]  # e.g. "r02_c05"
        try:
            r, c = parse_rc_from_name(base)
        except ValueError:
            print(f"[WARNING] Could not parse r,c from '{base}'; skipping.")
            continue
        images.append((rgb_path, base, (r, c)))
    if len(images) < 2:
        raise RuntimeError(f"Need ≥2 matching RGB images. Found {len(images)}.")
    return images

def load_images(image_list):
    """
    Given a list of (rgb_path, base, (r,c)), load:
      - images: list of BGR uint8 arrays
      - bases:  list of str, e.g. 'r02_c05'
      - coords: list of (r,c) tuples
    """
    images, bases, coords = [], [], []
    for (rgb_path, base, (r,c)) in image_list:
        I = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if I is None:
            print(f"[WARNING] Could not read image '{rgb_path}'; skipping.")
            continue

        images.append(I)
        bases.append(base)
        coords.append((r, c))
    if len(images) < 2:
        raise RuntimeError(f"After loading, fewer than 2 valid pairs remain.")
    return images, bases, coords

def create_feature_detector(use_orb=False):
    """
    Returns (detector, matcher):
      - If use_orb=True: ORB + BFMatcher(HAMMING)
      - Else: SIFT + BFMatcher(L2)
    """
    if use_orb:
        detector = cv2.ORB_create(nfeatures=2000)
        matcher  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        detector = cv2.SIFT_create()
        matcher  = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    return detector, matcher

def detect_and_compute(detector, img):
    """
    Given a cv2 detector, run detectAndCompute on img (BGR).
    Returns (keypoints, descriptors).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, des = detector.detectAndCompute(gray, None)
    return kps, des

def match_features(matcher, des1, des2, ratio=0.75):
    """
    KNN‐match (k=2) from des1→des2, then apply Lowe’s ratio test.
    Returns list of good cv2.DMatch.
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m,n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def find_homography(kp1, kp2, matches, reproj_thresh=4.0):
    """
    Given keypoints kp1, kp2 and a list of cv2.DMatch,
    run RANSAC to find a 3×3 homography. Return (H, mask).
    If <4 matches, return (None, None).
    """
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)
    return H, mask

# ------------------------------------------------------------
# 2. MAIN PIPELINE
# ------------------------------------------------------------

def main(
        captures_path = "C:/Users/Mike/Documents/9.Scene/MyProject/WallAssets/captures",
        panorama_path = "C:/Users/Mike/Documents/9.Scene/MyProject/WallAssets/panorama.png",
        use_orb = True,
        min_inliers = 30):
    # 3.1. Discover & load all pairs
    images = find_images(captures_path)
    images, bases, coords = load_images(images)
    N = len(images)
    print(f"→ Loaded {N} RGB images.")

    # 3.2. Build maps: index → basename, index → (r,c)
    idx_to_rc   = {i: coords[i] for i in range(N)}

    # 3.3. Prepare feature detector & matcher
    detector, matcher = create_feature_detector(use_orb)

    # 3.4. Detect keypoints + descriptors for every image
    keypoints = [None]*N
    descriptors = [None]*N
    for i in range(N):
        kp, des = detect_and_compute(detector, images[i])
        keypoints[i] = kp
        descriptors[i] = des

    # 3.5. Build “grid‐adjacency” edges:
    #      Only attempt to match if |r1-r2| + |c1-c2| == 1 (neighbors in grid).
    pairwise_H = {}             # (i,j) → H_{i→j}
    adjacency  = defaultdict(list)  # graph: i → list of neighbors j

    for i in range(N):
        r_i, c_i = idx_to_rc[i]
        for j in range(i+1, N):
            r_j, c_j = idx_to_rc[j]
            # Allow direct and diagonal neighbors (|dr|<=1 and |dc|<=1, but not itself)
            dr, dc = abs(r_i - r_j), abs(c_i - c_j)
            if not ((dr + dc == 1) or (dr == 1 and dc == 1)):
                continue

            # Attempt feature‐matching between i and j
            kp1, des1 = keypoints[i], descriptors[i]
            kp2, des2 = keypoints[j], descriptors[j]
            matches = match_features(matcher, des1, des2)
            H_ij, mask = find_homography(kp1, kp2, matches)
            if H_ij is None or mask is None:
                continue

            inliers = int(mask.sum())
            if inliers < min_inliers:
                continue

            # Normalize H so that H[2,2] = 1
            H_ij = H_ij / H_ij[2,2]
            H_ji = np.linalg.inv(H_ij)
            H_ji = H_ji / H_ji[2,2]

            pairwise_H[(i, j)] = H_ij
            pairwise_H[(j, i)] = H_ji
            adjacency[i].append(j)
            adjacency[j].append(i)

    if not pairwise_H:
        raise RuntimeError("No valid neighbor overlaps found. Check your captures.")

    # 3.6. Choose an anchor – the (r,c) closest to the mean (center of the grid)
    coords_arr = np.array([idx_to_rc[i] for i in range(N)])
    center_coord = coords_arr.mean(axis=0)
    dists = np.linalg.norm(coords_arr - center_coord, axis=1)
    anchor_idx = int(np.argmin(dists))

    # 3.7. BFS to compute global homographies Hs[i] mapping i → anchor frame
    Hs = {i: None for i in range(N)}
    Hs[anchor_idx] = np.eye(3, dtype=np.float64)

    visited = set([anchor_idx])
    queue = deque([anchor_idx])

    while queue:
        cur = queue.popleft()
        for nbr in adjacency[cur]:
            if nbr in visited:
                continue
            # We have H_{nbr→cur}, so H_nbr = H_cur @ H_{nbr→cur}
            H_nbr_to_cur = pairwise_H.get((nbr, cur), None)
            if H_nbr_to_cur is None:
                continue
            Hs[nbr] = Hs[cur] @ H_nbr_to_cur
            Hs[nbr] = Hs[nbr] / Hs[nbr][2,2]
            visited.add(nbr)
            queue.append(nbr)

    # Only keep indices that have a valid homography
    valid_indices = [i for i in range(N) if Hs[i] is not None]

    # 3.8. Determine output canvas size (only over connected images)
    H_img, W_img = images[0].shape[:2]
    all_corners = []

    for i in valid_indices:
        corners = np.array([
            [0,       0,      1],
            [W_img,   0,      1],
            [W_img,   H_img,  1],
            [0,       H_img,  1]
        ], dtype=np.float64).T  # shape=3×4
        warped = Hs[i] @ corners      # 3×4
        warped /= warped[2:3, :]      # normalize
        warped_xy = warped[:2, :].T   # 4×2
        all_corners.append(warped_xy)

    all_corners = np.vstack(all_corners)  # (4N)×2
    x_min, y_min = all_corners.min(axis=0)
    x_max, y_max = all_corners.max(axis=0)

    x_min, y_min = float(x_min), float(y_min)
    x_max, y_max = float(x_max), float(y_max)

    tx = -x_min if x_min < 0 else 0.0
    ty = -y_min if y_min < 0 else 0.0

    W_out = int(np.ceil(x_max + tx))
    H_out = int(np.ceil(y_max + ty))
    print(f"→ Panorama canvas: {W_out}×{H_out} (WxH)")

    # Translation to shift all warped images into positive coordinates
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1 ]
    ], dtype=np.float64)

    # 3.9. Prepare accumulators for RGB
    panorama_rgb_acc    = np.zeros((H_out, W_out, 3), dtype=np.float32)
    panorama_weight_acc = np.zeros((H_out, W_out),      dtype=np.float32)

    for i in valid_indices:
        H_net = T @ Hs[i]

        # Warp RGB (as float32) + coverage mask
        I = images[i].astype(np.float32)
        warped_rgb = cv2.warpPerspective(I, H_net, (W_out, H_out))

        # Create binary mask of valid pixels
        valid_mask = np.ones((H_img, W_img), dtype=np.uint8)

        # Distance transform from edges inward
        dist = cv2.distanceTransform(valid_mask, distanceType=cv2.DIST_L2, maskSize=3)
        dist_norm = dist / (dist.max() + 1e-6)  # Normalize to [0, 1]

        # Feather weight mask for RGB
        feather_mask_rgb = dist_norm.astype(np.float32)
        warped_feather_rgb = cv2.warpPerspective(feather_mask_rgb, H_net, (W_out, H_out))

        # Warp RGB image
        warped_rgb = cv2.warpPerspective(I, H_net, (W_out, H_out))

        # Accumulate with soft mask
        for c in range(3):
            panorama_rgb_acc[..., c] += warped_rgb[..., c] * warped_feather_rgb
        panorama_weight_acc += warped_feather_rgb

    # 3.11. Finalize RGB panorama
    panorama_rgb = np.zeros_like(panorama_rgb_acc, dtype=np.uint8)
    nonzero = (panorama_weight_acc > 0)
    for c in range(3):
        channel = panorama_rgb_acc[..., c] / (panorama_weight_acc + 1e-6) 
        channel[~nonzero] = 0
        panorama_rgb[..., c] = np.clip(channel, 0, 255).astype(np.uint8)

    # 3.13. Save outputs
    cv2.imwrite(panorama_path, panorama_rgb)

if __name__ == "__main__":
    main()