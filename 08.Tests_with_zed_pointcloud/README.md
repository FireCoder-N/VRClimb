## Purpose and Differences Between the Files

### 1. `pointcloud.py` — Depth Data Validation
- Captures raw depth measurements (`sl.MEASURE.DEPTH`)
- Filters invalid (NaN) values and inspects numeric ranges
- Compares depth between two captures
- Used for sensor and SDK sanity checks

---

### 2. `zed.py` — Depth Visualization and Inspection
- Converts depth data into a 2D grayscale image
- Applies nonlinear filtering and thresholding
- Displays depth maps for human inspection
- Reads metric depth at individual pixels

---

### 3. `zed_pc.py` — 3D Reconstruction Pipeline
- Combines RGB and depth into point clouds
- Uses camera intrinsics for metric accuracy
- Aligns multiple frames using FPFH + RANSAC
- Merges point clouds and reconstructs a 3D mesh

---

### 4. `zed_pointcloud_stabilizer.py` — Pointcloud Stabilization, Cleaning, Visualization
- Reiteration of zed.py, replaces image logic entirely working only in pointclouds
- Clips Z-coordinate outliers
- Replaces extreme values with mean depth
- Converts NumPy → Open3D point clouds
- Visualizes cleaned geometry