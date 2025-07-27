"""
stitch.py

Reads all captures in <input_folder>/rgb (PNG) and <input_folder>/depth (NPY),
where each filename is prefixed by “rXX_cYY” (row and column). E.g.:

   rgb/r00_c00.png
   rgb/r00_c01.png
   rgb/r01_c00.png
   rgb/r01_c01.png
   ...

It will:

  1) Parse filenames to figure out “rows” and “columns.”
  2) For each row:
       • Load that row’s RGB images in increasing column order,
       • Stitch them (left→right) into one “row‐RGB-panorama.”
       • Simultaneously warp & blend that row’s depth-maps into a “row-depth-panorama”
         so it matches the row-RGB-panorama pixel-for-pixel.
  3) Now you have R “row panoramas.” Stitch **those** vertically
     (i.e. stitch row 0 with row 1, then with row 2, …) to form one big panorama.
     Do the same for the per-row depth panoramas (min‐depth & avg‐depth).

Outputs (in `<output_folder>/`):
   • rgb_full_pano.png
   • depth_full_pano_min.npy
   • depth_full_pano_avg.npy
   • depth_full_pano_min.png   (8-bit viridis visualization)
   • depth_full_pano_avg.png   (8-bit viridis visualization)
"""

import os
import re
import sys
import cv2
import numpy as np
import argparse

def load_row_col_map(folder, ext):
    """
    Scans `folder` for files ending in `ext` (e.g. '.png' or '.npy'),
    expecting names like 'rXX_cYY.ext'. Returns a dict:
       { row_index : { col_index : full_path, ... }, ... }
    """

    # Regex to match: r<digits>_c<digits>.<ext>
    pattern = re.compile(r"^r(\d+)_c(\d+)" + re.escape(ext) + r"$")

    row_map = {}
    for fname in os.listdir(folder):
        if not fname.lower().endswith(ext.lower()):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        r = int(m.group(1))
        c = int(m.group(2))
        full_path = os.path.join(folder, fname)
        row_map.setdefault(r, {})[c] = full_path

    return row_map  # { row0: { col0:path, col1:path, ... }, row1: {...}, ... }

def compute_pairwise_homographies(img_list, min_match_count=40):
    """
    Given a list of filepaths to grayscale images [I0, I1, ..., IN-1],
    compute homographies H_i→(i+1) between each consecutive pair using ORB+BFMatcher+RANSAC.
    Returns a list of length N-1: [H_0→1, H_1→2, …].
    """

    orb = cv2.ORB_create(5000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)

    H_steps = []
    prev_kp, prev_des = None, None
    prev_img = None

    for i, path in enumerate(img_list):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read {path}")

        kp, des = orb.detectAndCompute(img, None)

        if i == 0:
            prev_kp, prev_des, prev_img = kp, des, img
            continue

        # Match descriptors: prev → this
        matches = bf.knnMatch(prev_des, des, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < min_match_count:
            raise RuntimeError(
                f"Only {len(good)} matches between\n"
                f"  {img_list[i-1]}\n  and\n  {img_list[i]}\n"
                f"  (< min_match_count={min_match_count})"
            )

        src_pts = np.float32([ prev_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([     kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError(f"RANSAC failed between {img_list[i-1]} and {img_list[i]}")

        H_steps.append(H)
        prev_kp, prev_des, prev_img = kp, des, img

    return H_steps

def accumulate_homographies(H_steps):
    """
    Given [H_0→1, H_1→2, …, H_{N-2}→(N-1)], compute
    H_i→0 for i=0..N-1 (mapping image i into the coordinate frame of image 0).
    Returns list H_to_ref of length N, where H_to_ref[0] = I, and
      H_to_ref[i] = H_to_ref[i-1] @ H_{i-1→i}.
    """

    N_minus1 = len(H_steps)
    N = N_minus1 + 1
    H_to_ref = [np.eye(3, dtype=np.float64)]
    for i in range(1, N):
        H_prev = H_to_ref[i-1]
        H_step = H_steps[i-1]
        H_to_ref.append(H_prev.dot(H_step))
    return H_to_ref  # list of length N

def warp_and_blend_rgb(img_paths, H_to_ref, pano_size):
    """
    Warp each BGR image (img_paths[i]) via H_to_ref[i] into canvas pano_size=(W,H).
    Simple linear blending: at each pixel, sum up weighted colors and divide by #contributors.
    Returns one uint8 BGR panorama of shape (H, W, 3).
    """

    Wp, Hp = pano_size
    pano_bgr = np.zeros((Hp, Wp, 3), dtype=np.float32)
    weight_sum = np.zeros((Hp, Wp), dtype=np.float32)

    for idx, ipath in enumerate(img_paths):
        img_bgr = cv2.imread(ipath, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        if img_bgr is None:
            raise RuntimeError(f"Could not read RGB {ipath}")

        warped = cv2.warpPerspective(
            img_bgr, H_to_ref[idx], (Wp, Hp),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)
        )

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = (gray > 1e-5).astype(np.float32)  # where this image “contributes”

        pano_bgr[...,0] += warped[...,0] * mask
        pano_bgr[...,1] += warped[...,1] * mask
        pano_bgr[...,2] += warped[...,2] * mask
        weight_sum     += mask

    valid = (weight_sum > 0)
    pano_bgr[valid] /= weight_sum[valid][..., None]
    pano_uint8 = (np.clip(pano_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return pano_uint8

def stitch_depth_maps(depth_paths, H_to_ref, pano_size):
    """
    Similar to warp_and_blend_rgb, but for float32 depth arrays. Produces two outputs:
      • depth_pano_min : at each (u,v), the **minimum** (closest) positive depth among warped inputs
      • depth_pano_avg : at each (u,v), the **average** of all positive depths among warped inputs
    Any pixel out-of-bounds or no positive depth → 0 in both outputs.
    """

    Wp, Hp = pano_size
    depth_pano_min = np.full((Hp, Wp), np.inf, dtype=np.float32)
    depth_sum      = np.zeros((Hp, Wp), dtype=np.float32)
    depth_count    = np.zeros((Hp, Wp), dtype=np.uint32)

    for idx, dpath in enumerate(depth_paths):
        Di = np.load(dpath)  # (h,w) float32 in meters
        if Di.dtype != np.float32:
            Di = Di.astype(np.float32)

        warped = cv2.warpPerspective(
            Di, H_to_ref[idx].astype(np.float32),
            (Wp, Hp),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )

        valid = (warped > 0.0)
        # min-depth
        depth_pano_min[valid] = np.minimum(depth_pano_min[valid], warped[valid])
        # average-depth
        depth_sum[valid]   += warped[valid]
        depth_count[valid] += 1

    # Fix “never-seen” pixels for min-case (still = +inf → set to 0)
    mask_inf = (depth_pano_min == np.inf)
    depth_pano_min[mask_inf] = 0.0

    # Build average map
    depth_pano_avg = np.zeros_like(depth_sum, dtype=np.float32)
    nonzero = (depth_count > 0)
    depth_pano_avg[nonzero] = depth_sum[nonzero] / depth_count[nonzero]

    return depth_pano_min, depth_pano_avg

def make_depth_visualization(depth_map, vmax=None):
    """
    Convert float32 depth (meters) → 8-bit viridis colormap (RGB).
    Clips at the 95th percentile by default to avoid outlier skew.
    """
    if vmax is None:
        vals = depth_map[depth_map > 0]
        if vals.size == 0:
            vmax = 1.0
        else:
            vmax = np.percentile(vals, 95)

    scaled = np.clip(depth_map / vmax, 0.0, 1.0)
    # Convert to 0..255, then apply OpenCV’s COLORMAP_VIRIDIS
    scaled_8u = (scaled * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(scaled_8u, cv2.COLORMAP_VIRIDIS)
    return colored

def main():

    input_folder = "captures"
    min_match_count = 40
    output_folder = "output"

    # 1) Check for subfolders
    rgb_folder   = os.path.join(input_folder, "rgb")
    depth_folder = os.path.join(input_folder, "depth")
    if not os.path.isdir(rgb_folder) or not os.path.isdir(depth_folder):
        print("Error: expected 'rgb/' and 'depth/' subfolders under", input_folder)
        sys.exit(1)

    # 2) Build row→(col→path) maps for RGB and Depth
    rgb_rows   = load_row_col_map(rgb_folder,   ".png")
    depth_rows = load_row_col_map(depth_folder, ".npy")
    if set(rgb_rows.keys()) != set(depth_rows.keys()):
        print("Error: mismatch in row indices between rgb/ and depth/")
        sys.exit(1)

    # 3) For each row index (e.g. 0,1,2,...), ensure same columns exist
    for r in rgb_rows:
        if set(rgb_rows[r].keys()) != set(depth_rows[r].keys()):
            print(f"Error: row {r} has different columns in rgb vs. depth")
            sys.exit(1)

    # Sort row indices
    sorted_rows = sorted(rgb_rows.keys())
    num_rows = len(sorted_rows)
    print(f"Found {num_rows} rows: {sorted_rows}")

    # We will store per-row panoramas in these lists (in order of sorted_rows)
    row_rgb_panos      = []
    row_depth_min_panos = []
    row_depth_avg_panos = []

    # ------------------------------
    # 4) Stitch each **horizontal row** individually (left→right)
    # ------------------------------
    for r in sorted_rows:
        print(f"\nStitching ROW = {r} (left→right)...")

        # 4a) Sort columns
        sorted_cols = sorted(rgb_rows[r].keys())
        rgb_paths_row   = [rgb_rows[r][c]   for c in sorted_cols]
        depth_paths_row = [depth_rows[r][c] for c in sorted_cols]

        # 4b) Load one example to get (h,w)
        ex_img = cv2.imread(rgb_paths_row[0], cv2.IMREAD_COLOR)
        if ex_img is None:
            raise RuntimeError(f"Could not read {rgb_paths_row[0]}")
        h0, w0 = ex_img.shape[:2]

        # 4c) Compute homographies between each consecutive pair in this row
        #     (We only call compute_pairwise_homographies on pairs that do overlap left→right.)
        H_steps = compute_pairwise_homographies(
            rgb_paths_row, #[cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in rgb_paths_row],
            min_match_count=min_match_count
        )
        H_to_ref = accumulate_homographies(H_steps)  # length = #images_in_row

        # 4d) Figure out row-canvas size by warping corners of each image
        all_corners = []
        for i in range(len(rgb_paths_row)):
            corners = np.array([
                [0,   0,   1],
                [w0,  0,   1],
                [w0,  h0,  1],
                [0,   h0,  1]
            ], dtype=np.float64).T  # shape (3,4)
            warped = H_to_ref[i].dot(corners)   # (3,4)
            warped /= warped[2:3, :]
            all_corners.append(warped[:2, :].T)  # (4,2)

        all_corners = np.vstack(all_corners)  # (4·#images_in_row, 2)
        xmin, ymin = np.floor(all_corners.min(axis=0)).astype(int)
        xmax, ymax = np.ceil(all_corners.max(axis=0)).astype(int)

        width_row   = xmax - xmin
        height_row  = ymax - ymin

        # Build translation T so that everything shifts by (-xmin, -ymin)
        Trow = np.array([
            [1, 0, -xmin],
            [0, 1, -ymin],
            [0, 0, 1]
        ], dtype=np.float64)

        # Translate each homography
        H_translated = [ (Trow.dot(H_to_ref[i])).astype(np.float32)
                         for i in range(len(H_to_ref)) ]

        # 4e) Warp & blend **RGB** for this row
        rgb_row_pano = warp_and_blend_rgb(rgb_paths_row, H_translated, (width_row, height_row))
        row_rgb_panos.append((rgb_row_pano, (width_row, height_row), Trow, H_to_ref))

        # 4f) Warp & blend **Depth** for this row
        depth_min_pano, depth_avg_pano = stitch_depth_maps(depth_paths_row, H_translated, (width_row, height_row))
        row_depth_min_panos.append(depth_min_pano)
        row_depth_avg_panos.append(depth_avg_pano)

        print(f"  → Row {r} panorama size = (W={width_row}, H={height_row})")

    # ------------------------------
    # 5) Now we have N row-panoramas. Stitch them **vertically** (row0 over row1 over row2, …)
    # ------------------------------
    print("\nStitching vertical stack of row-panoramas...")

    # 5a) First, decide reference = row0. Prepare for homographies between rowPanos
    # We’ll compute H_rowSteps[i] = homography( row_i_pano → row_{i+1}_pano ) for i=0..num_rows-2
    H_row_steps = []
    for i in range(num_rows-1):
        # Convert to grayscale for feature matching
        grayA = cv2.cvtColor(row_rgb_panos[i][0], cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(row_rgb_panos[i+1][0], cv2.COLOR_BGR2GRAY)
        # Estimate H( row_i → row_{i+1} ) because they overlap vertically
        orb = cv2.ORB_create(5000)
        bf  = cv2.BFMatcher(cv2.NORM_HAMMING)
        kpA, desA = orb.detectAndCompute(grayA, None)
        kpB, desB = orb.detectAndCompute(grayB, None)
        matches = bf.knnMatch(desA, desB, k=2)
        good = [m for m,n in matches if m.distance < 0.75 * n.distance]
        if len(good) < min_match_count:
            raise RuntimeError(
                f"Not enough matches ({len(good)}) between row {i} pano and row {i+1} pano"
            )
        src_pts = np.float32([ kpA[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpB[m.trainIdx].pt  for m in good ]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError(f"RANSAC failed for rows {i}→{i+1}")
        H_row_steps.append(H)

    # 5b) Accumulate H_to_ref for row-panoramas, all into row0’s frame
    H_rows_to_ref = [np.eye(3, dtype=np.float64)]
    for i in range(1, num_rows):
        H_rows_to_ref.append(H_rows_to_ref[i-1].dot(H_row_steps[i-1]))

    # 5c) Find global bounding‐box across all row-panos by warping corners
    all_corners = []
    for i in range(num_rows):
        pano_img, (w_i, h_i), T_i, H_to_ref_i = row_rgb_panos[i]
        corners = np.array([
            [0,   0,   1],
            [w_i, 0,   1],
            [w_i, h_i, 1],
            [0,   h_i, 1]
        ], dtype=np.float64).T  # (3,4)

        warped = H_rows_to_ref[i].dot(corners)
        warped /= warped[2:3,:]
        all_corners.append(warped[:2,:].T)

    all_corners = np.vstack(all_corners)
    xmin_glob, ymin_glob = np.floor(all_corners.min(axis=0)).astype(int)
    xmax_glob, ymax_glob = np.ceil(all_corners.max(axis=0)).astype(int)

    width_full  = xmax_glob - xmin_glob
    height_full = ymax_glob - ymin_glob
    print(f"Full panorama canvas = (W={width_full}, H={height_full})")

    # 5d) Build global translation T_full to shift everything into [0..width_full)x[0..height_full)
    T_full = np.array([
        [1, 0, -xmin_glob],
        [0, 1, -ymin_glob],
        [0, 0, 1]
    ], dtype=np.float64)

    # 5e) Build final homographies for each row-pano: T_full @ H_rows_to_ref[i]
    H_rows_translated = [ (T_full.dot(H_rows_to_ref[i])).astype(np.float32)
                          for i in range(num_rows) ]

    # 5f) Collect the list of row-pano RGB images (as temporary files in memory)
    #     and the list of depth-panos for min/avg (in memory).
    #     We already have them in row_rgb_panos and row_depth_*_panos.

    # 5g) Warp & blend **RGB row-panoramas** onto the final canvas
    final_rgb_canvas = np.zeros((height_full, width_full, 3), dtype=np.float32)
    weight_sum_canvas = np.zeros((height_full, width_full), dtype=np.float32)

    for i in range(num_rows):
        row_img = row_rgb_panos[i][0].astype(np.float32) / 255.0  # [0..1]
        warped = cv2.warpPerspective(
            row_img, H_rows_translated[i],
            (width_full, height_full),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)
        )
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = (gray > 1e-5).astype(np.float32)

        final_rgb_canvas[...,0] += warped[...,0] * mask
        final_rgb_canvas[...,1] += warped[...,1] * mask
        final_rgb_canvas[...,2] += warped[...,2] * mask
        weight_sum_canvas += mask

    valid = (weight_sum_canvas > 0)
    final_rgb_canvas[valid] /= weight_sum_canvas[valid][..., None]
    final_rgb_uint8 = (np.clip(final_rgb_canvas, 0.0, 1.0) * 255.0).astype(np.uint8)

    # 5h) Warp & blend **Depth** row-panoramas (two cases: min‐depth & avg‐depth)
    depth_full_min = np.full((height_full, width_full), np.inf, dtype=np.float32)
    depth_sum_full = np.zeros((height_full, width_full), dtype=np.float32)
    depth_count_full = np.zeros((height_full, width_full), dtype=np.uint32)

    depth_full_avg = np.zeros((height_full, width_full), dtype=np.float32)

    for i in range(num_rows):
        dm_min = row_depth_min_panos[i]  # float32
        dm_avg = row_depth_avg_panos[i]

        warped_min = cv2.warpPerspective(
            dm_min, H_rows_translated[i],
            (width_full, height_full),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        warped_avg = cv2.warpPerspective(
            dm_avg, H_rows_translated[i],
            (width_full, height_full),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )

        valid_min = (warped_min > 0.0)
        depth_full_min[valid_min] = np.minimum(depth_full_min[valid_min], warped_min[valid_min])

        # For averaging, we only accumulate from the “avg‐of‐row” if row_depth_count>0.
        # But we don’t have a per-pixel count for row-depth_avg; instead, _row_ min/avg already blended.
        # So we’ll just treat warped_avg>0 as 1 contributor, and sum its value.
        valid_avg = (warped_avg > 0.0)
        depth_sum_full[valid_avg] += warped_avg[valid_avg]
        depth_count_full[valid_avg] += 1

    # Fix any +inf in min case → set to 0
    inf_mask = (depth_full_min == np.inf)
    depth_full_min[inf_mask] = 0.0

    # Build final average
    nonzero_mask = (depth_count_full > 0)
    depth_full_avg[nonzero_mask] = depth_sum_full[nonzero_mask] / depth_count_full[nonzero_mask]
    # leave depth_full_avg[...] = 0 where count==0

    # 5i) Save outputs
    os.makedirs(output_folder, exist_ok=True)
    rgb_out_path = os.path.join(output_folder, "rgb_full_pano.png")
    cv2.imwrite(rgb_out_path, final_rgb_uint8)
    print(f"Saved full RGB panorama → {rgb_out_path}")

    # Depth .npy files
    dmin_npy = os.path.join(output_folder, "depth_full_pano_min.npy")
    davg_npy = os.path.join(output_folder, "depth_full_pano_avg.npy")
    np.save(dmin_npy, depth_full_min)
    np.save(davg_npy, depth_full_avg)
    print(f"Saved depth-min panorama → {dmin_npy}")
    print(f"Saved depth-avg panorama → {davg_npy}")

    # Also save 8-bit visualizations
    vis_min = make_depth_visualization(depth_full_min)
    vis_avg = make_depth_visualization(depth_full_avg)
    dmin_png = os.path.join(output_folder, "depth_full_pano_min.png")
    davg_png = os.path.join(output_folder, "depth_full_pano_avg.png")
    cv2.imwrite(dmin_png, vis_min)
    cv2.imwrite(davg_png, vis_avg)
    print(f"Saved depth-min visualization → {dmin_png}")
    print(f"Saved depth-avg visualization → {davg_png}")

if __name__ == "__main__":
    main()
