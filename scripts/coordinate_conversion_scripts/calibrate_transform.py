# calibrate_transform.py
"""
Compute the rigid-body transform (rotation + translation) between camera and map frames
using 3D–3D correspondences (Kabsch algorithm).
"""
import numpy as np

def compute_rigid_transform(cam_pts: np.ndarray, map_pts: np.ndarray) -> np.ndarray:
    """
    cam_pts, map_pts: (N,3) arrays of corresponding 3D points
    Returns 4x4 homogeneous transform T such that:
      [X_map; 1] = T @ [x_cam; 1]
    """
    assert cam_pts.shape == map_pts.shape and cam_pts.shape[0] >= 3

    # 1. centroids
    centroid_cam = cam_pts.mean(axis=0)
    centroid_map = map_pts.mean(axis=0)

    # 2. center the points
    A = cam_pts - centroid_cam
    B = map_pts - centroid_map

    # 3. covariance matrix
    H = A.T @ B

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # ensure a proper rotation (determinant = +1)
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    # 5. translation
    t = centroid_map - R @ centroid_cam

    # 6. assemble 4x4 transform
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

if __name__ == "__main__":
    # example usage: replace these with your measured correspondences
    # cam_points = np.array([
    #     [0.1, 0.2, 1.5],
    #     [0.5, 0.2, 1.5],
    #     [0.5, 0.6, 1.5],
    #     [0.1, 0.6, 1.5],
    #     [0.3, 0.4, 1.0],
    #     [0.7, 0.8, 1.2]
    # ])
    # map_points = np.array([
    #     [1.0, 2.0, 0.0],
    #     [2.0, 2.0, 0.0],
    #     [2.0, 3.0, 0.0],
    #     [1.0, 3.0, 0.0],
    #     [1.5, 2.5, 0.0],
    #     [2.5, 3.5, 0.0]
    # ])

    # Rover 1
    cam_points = np.array([
        [-1.11,0.32,5.24],
        [-1.01, 0.33,3.02],
        [0.85, 0.19, 3.46],
        [-0.48, 0.03, 3.83],
        [-0.09, 0.21, 3.24],
        [-0.56, -0.04, 3.82]
    ])
    map_points = np.array([
        [-0.413, 0.328, 0.0],
        [-0.478, 1.539, 0.0],
        [-2.116, 1.164, 0.0],
        [-0.987, 1.021, 0.0],
        [-1.309, 1.375, 0.0],
        [-0.829, 0.854, 0.0]
    ])



    T = compute_rigid_transform(cam_points, map_points)
    print("Computed camera→map transform:")
    print(T)
    