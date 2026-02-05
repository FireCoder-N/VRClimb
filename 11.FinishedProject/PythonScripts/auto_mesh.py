import cv2
import torch
import open3d as o3d
import numpy as np
from ultralytics import YOLO

# =======================================================
# Plot
# =======================================================
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# from mpl_toolkits.mplot3d import Axes3D
from skimage.color import label2rgb

def plot_h(h1, h2, h3):
    """
    Plots a comparison between H_raw, H'_raw, and H_raw (final)
    """
    
    # Plot matrices side by side
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Display each matrix
    ax[0].imshow(h1, cmap='viridis')
    ax[0].set_title(r'$H_{raw}$', fontsize='medium')
    ax[0].axis('off')

    ax[1].imshow(h2, cmap='viridis')
    ax[1].set_title(r"$H_{raw}'$", fontsize='medium')
    ax[1].axis('off')

    ax[2].imshow(h3, cmap='viridis')
    ax[2].set_title(r'$\mathcal{H}_{raw}$', fontsize='medium')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


def plot_yolo_mask(mask, img=None, title="YOLO Mask"):
    """
    Plots the YOLO segmentation mask. Optionally overlay on the original RGB image.
    """
    plt.figure(figsize=(8,6))
    if img is not None:
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5, cmap='Reds')
    else:
        plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_depthmap(depth, title="MiDaS Depthmap"):
    """
    Visualizes the MiDaS raw depthmap with colormap.
    """
    plt.figure(figsize=(8,6))
    plt.imshow(depth, cmap='inferno')
    plt.colorbar(label="Depth (relative)")
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_connected_components(labels, num_labels=None, background=0, figsize=(8, 8), title=None, max_legend_items=None):
    """
    Visualize connected components using matplotlib and label2rgb, with a legend.
    
    Parameters:
        labels (ndarray): Output from cv2.connectedComponents (2D int array).
        num_labels (int, optional): Number of labels (for info only).
        background (int): Background label to mask out (usually 0).
        figsize (tuple): Size of the matplotlib figure.
        title (str, optional): Title for the plot.
        max_legend_items (int, optional): Max number of items to display in the legend.
    """
    # Generate color overlay
    overlay = label2rgb(labels, bg_label=background)

    # Create figure
    plt.figure(figsize=figsize)
    plt.imshow(overlay)
    
    # Default title
    if title is None:
        title = f"{num_labels if num_labels is not None else np.max(labels) + 1} Connected Components"
    plt.title(title)
    plt.axis('off')

    # Prepare legend
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != background]  # Exclude background

    if max_legend_items is not None:
        unique_labels = unique_labels[:max_legend_items]

    # # Limit number of legend items
    # if len(unique_labels) > max_legend_items:
    #     unique_labels = unique_labels[:max_legend_items]

    # Generate colors using the same mapping as label2rgb
    colormap = plt.cm.get_cmap('nipy_spectral', np.max(labels) + 1)
    legend_elements = [
        Patch(facecolor=colormap(label), edgecolor='black', label=f'Label {label}')
        for label in unique_labels
    ]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)
    plt.tight_layout()
    plt.show()


def plot_knn_neighbourhood(pcd, idx=0, k=30):
    """
    Plots a point and its kNN neighbourhood in 3D.
    pcd: open3d.geometry.PointCloud
    idx: index of the query point
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [_, idxs, _] = pcd_tree.search_knn_vector_3d(pcd.points[idx], k)

    pts = np.asarray(pcd.points)
    query_point = pts[idx]
    neigh_points = pts[idxs, :]

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, alpha=0.05, c='gray')
    ax.scatter(neigh_points[:,0], neigh_points[:,1], neigh_points[:,2], s=40, c='blue')
    ax.scatter(query_point[0], query_point[1], query_point[2], s=100, c='red', marker='x')
    ax.set_title(f"{k}-NN neighbourhood of point {idx}")
    print("done")
    plt.show()

def plot_normals(pcd_before, pcd_after, step=50):
    """
    Visualizes normals before and after orientation propagation.
    step: sample every 'step' points to avoid clutter.
    """
    pts_before = np.asarray(pcd_before.points)
    normals_before = np.asarray(pcd_before.normals)

    pts_after = np.asarray(pcd_after.points)
    normals_after = np.asarray(pcd_after.normals)

    fig = plt.figure(figsize=(14,6))

    # Before orientation
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.quiver(pts_before[::step,0], pts_before[::step,1], pts_before[::step,2],
               normals_before[::step,0], normals_before[::step,1], normals_before[::step,2],
               length=0.005, normalize=True, color='blue')
    ax1.set_title("Normals Before Orientation")

    # After orientation
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.quiver(pts_after[::step,0], pts_after[::step,1], pts_after[::step,2],
               normals_after[::step,0], normals_after[::step,1], normals_after[::step,2],
               length=0.005, normalize=True, color='green')
    ax2.set_title("Normals After Orientation")

    print("normals done")
    plt.show()


# =======================================================
# MiDaS
# =======================================================
def MiDaS_depthmap_generation(img_rgb, model="DPT_Hybrid"): # model = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model)
    midas.to("cpu")
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model == "DPT_Large" or model == "DPT_Hybrid":
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    imgbatch = transform(img_rgb).to("cpu")

    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img_rgb.shape[:2],
            mode = "bicubic",
            align_corners=False
        ).squeeze()

    return prediction.cpu().numpy()

# =======================================================
# YOLO
# =======================================================
def Holds_detection(img, model="C:/Users/Mike/Documents/9.Scene/MyProject/WallAssets/best.pt"):
    yolo = YOLO(model)
    yolo_results = yolo(img, retina_masks=True)
    mask = yolo_results[0].masks.data
    mask = torch.any(mask, dim=0).int()
    mask = mask.cpu().numpy().astype(np.uint8)

    return mask
    

def build_o3d_mesh(
        img_path="C:/Users/Mike/Documents/9.Scene/MyProject/WallAssets/panorama.png",
        gt_depth_path="C:/Users/Mike/Documents/9.Scene/MyProject/WallAssets/r01_c02.npy",
        save_path="C:/Users/Mike/Documents/9.Scene/MyProject/WallAssets/output.obj"):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_depth = np.load(gt_depth_path)
    gt_depth = gt_depth.astype(np.float32)
    valid_mask = np.isfinite(gt_depth) & (gt_depth > 0)
    valid_depth = gt_depth[valid_mask]
    Z_min, Z_max = np.min(valid_depth), np.max(valid_depth)

    output = MiDaS_depthmap_generation(img_rgb)
    mask = Holds_detection(img)

    plot_depthmap(output)
    plot_yolo_mask(mask)

    # mask invalid pixels
    valid_mask = np.isfinite(output) & (output > 0)

    d_min = output[valid_mask].min()
    d_max = output[valid_mask].max()

    # linear remap
    depth_norm = np.zeros_like(output, dtype=np.float32)
    depth_norm[valid_mask] = Z_min + (Z_max - Z_min) * (output[valid_mask] - d_min) / (d_max - d_min)

    depth_norm[~valid_mask] = np.nan


    # Fit plane to bg via SVD
    bg_pixels = np.where((mask == 0) & np.isfinite(depth_norm))
    v_bg, u_bg = bg_pixels
    Z_bg = depth_norm[v_bg, u_bg]

    # Convert to camera coords (pin‐hole). Suppose fx=fy=f, cx=cx, cy=cy.
    f = 800.   # example focal length in pixels
    W, H = img.shape[:2]
    cx, cy = W/2, H/2

    X_bg = (u_bg - cx) * (Z_bg / f)
    Y_bg = (v_bg - cy) * (Z_bg / f)
    P_bg = np.stack([X_bg, Y_bg, Z_bg], axis=1)   # shape = (N_bg, 3)

    # 1) Compute centroid:
    centroid = P_bg.mean(axis=0)
    P_centered = P_bg - centroid[None,:]

    # 2) SVD:
    U, S, Vt = np.linalg.svd(P_centered, full_matrices=False)
    normal = Vt[-1, :]              # last row of Vt  => a unit‐length normal vector (a,b,c)
    d = - normal.dot(centroid)      # so that (a,b,c)·(X,Y,Z)+d = 0

    # Now (a,b,c,d) define the best‐fit wall plane.
    a, b, c = normal

    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
    X_cam = (u_coords - cx) / f
    Y_cam = (v_coords - cy) / f
    C = np.ones_like(X_cam)

    den = a * X_cam + b * Y_cam + c * C
    den = den.T

    depth_plane = np.zeros_like(depth_norm) * np.nan
    eps = 1e-6
    background_pixels = (den != 0)
    valid_den = np.abs(den) > eps
    depth_plane[valid_den] = (-d / den[valid_den]).astype(np.float32)

    delta = depth_norm - depth_plane
    delta2 = depth_plane - depth_norm
    H_raw1 = np.maximum(delta, 0.0)
    H_raw2 = np.maximum(delta2, 0.0)
    H_raw = H_raw1 - H_raw2

    # plot_h(H_raw1, H_raw2, H_raw)

    num_labels, labels = cv2.connectedComponents((mask>0).astype(np.uint8))
    # plot_connected_components(labels, num_labels)
    # labels[u,v] ∈ {0,1,2,…,num_labels−1}, where 0 = background, 1..(num_labels−1)= each hold.

    # For convenience, let K = num_labels−1 (number of holds).
    # Pre‐allocate arrays:
    H_new = np.zeros_like(H_raw, dtype=np.float32)


    # (1) Compute per‐hold h_k_min, h_k_max:
    h_k_min = np.zeros(num_labels, dtype=np.float32) + np.inf
    h_k_max = np.zeros(num_labels, dtype=np.float32) - np.inf

    for k in range(1, num_labels):
        # Extract all H_raw within hold k:
        u_k, v_k = np.where(labels == k)
        if len(u_k)==0:
            # no pixels? skip
            h_k_min[k] = 0
            h_k_max[k] = 0
            continue

        values = H_raw[u_k, v_k]
        h_k_min[k] = values.min()
        h_k_max[k] = values.max()

    # (2) Compute each hold’s local span s_k = h_k_max - h_k_min:
    s_k = h_k_max - h_k_min

    # (3) Compute the global hold span:
    #     We want s_global = max_k( h_k_max ) – min_k( h_k_min ),
    #     but since each h_k_min >= 0, typically min_k h_k_min = 0.
    #     So s_global = max_k( h_k_max ).  Equivalently s_global = max(s_k).
    s_global = s_k.max()

    # (4) Now rescale each hold so that its new top = s_global:
    for k in range(1, num_labels):
        u_k, v_k = np.where(labels == k)
        if len(u_k)==0:
            continue

        if s_k[k] < 1e-6:
            # If this hold is essentially flat in H_raw, we can either leave it almost zero,
            # or give it a tiny epsilon height so it shows up as a “bump.” For example:
            for (u,v) in zip(u_k, v_k):
                H_new[u,v] = 1e-3   # 1 mm thick bump
            continue

        # Otherwise, do the per‐pixel formula:
        #    H_new(u,v) = ( H_raw(u,v) – h_k_min ) * ( s_global / s_k[k] ).
        scale = s_global / s_k[k]
        for (u,v) in zip(u_k, v_k):
            H_new[u, v] = (H_raw[u, v] - h_k_min[k]) * scale

    s = 0.005

    X = (u_coords - (W/2)) * s
    Y = (v_coords - (H/2)) * s
    Z = H_new.T

    # plot_depthmap(H_new, "")

    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size=0.002)

    pcd.transform([[0, 1, 0, 0],
                [-1, 0, 0, 0],                                                                                                                                                                  
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    
    # plot_knn_neighbourhood(pcd, 100)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    import copy
    pcd_before = copy.deepcopy(pcd)

    normals = np.asarray(pcd.normals)
    flip_mask = normals[:,2] < 0
    normals[flip_mask] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)

    pcd_after = copy.deepcopy(pcd)
    plot_normals(pcd_before, pcd_after, step=100)

    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=8,
        scale=1.1,
        linear_fit=False,
        n_threads=0
    )

    dens = np.asarray(densities)
    density_threshold = np.quantile(dens, 0.1)
    vertices_to_keep = np.where(dens > density_threshold)[0]

    mesh_poisson_clean = mesh_poisson.select_by_index(vertices_to_keep)
    mesh_poisson_clean.remove_unreferenced_vertices()
    mesh_poisson_clean.remove_degenerate_triangles()
    mesh_poisson_clean.remove_duplicated_triangles()
    mesh_poisson_clean.remove_duplicated_vertices()
    mesh_poisson_clean.remove_non_manifold_edges()
    mesh_poisson_clean.compute_vertex_normals()

    target_number_of_triangles = 10000  # Adjust as needed
    simplified_mesh = mesh_poisson_clean.simplify_quadric_decimation(target_number_of_triangles)


    o3d.io.write_triangle_mesh(save_path, simplified_mesh)
    return simplified_mesh


def main():
    mesh = build_o3d_mesh()
    return mesh

if __name__ == "__main__":
    main()