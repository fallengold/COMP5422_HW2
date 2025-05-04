"""
Homework2.
Replace 'pass' by your implementation.
"""

# Insert your package here
import cv2
import numpy as np
from homework.helper import getAbsoluteScale, camera2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from scipy.ndimage import gaussian_filter
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def find_matched_points(im1, im2, output_image_path="matches.jpg"):
    """
    Args: im1: First input image
          im2: Second input image
          output_image_path: Path to save the visualization of matches

    Returns:
        pts1: Nx2 array of matched points in the first image
        pts2: Nx2 array of matched points in the second image
    """
    # Ensure images are grayscale for SIFT
    if len(im1.shape) == 3:
        gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = im1
    if len(im2.shape) == 3:
        gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = im2

    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)

    # gray1 = im1
    # gray2 = im2

    # Initialize SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # bf matcher
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    print(f"Keypoints in im1: {len(keypoints1)}")
    print(f"Keypoints in im2: {len(keypoints2)}")
    print(f"Initial matches: {len(matches)}")

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    distances = [m.distance for m in matches]
    print(
        f"Match distances: min={min(distances)}, max={max(distances)}, mean={np.mean(distances)}"
    )

    model, inliers = ransac(
        (pts1, pts2),
        FundamentalMatrixTransform,
        min_samples=8,
        residual_threshold=1,
        max_trials=10000,
    )

    if inliers is not None:

        pts1 = pts1[inliers]
        pts2 = pts2[inliers]

    matches = [m for i, m in enumerate(matches) if inliers[i]]
    print(f"RANSAC matches: {len(matches)}")
    top_matches = matches[:100]
    match_img = cv2.drawMatches(
        im1,
        keypoints1,
        im2,
        keypoints2,
        top_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite(output_image_path, match_img)
    return pts1, pts2, model


"""
Q2.2.2: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: the fundamental matrix
"""


def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    T = np.array([[1 / M, 0.0, 0.0], [0.0, 1 / M, 0.0], [0.0, 0.0, 1.0]])
    N = pts1.shape[0]

    ones = np.ones((N, 1))
    pts1_h = np.hstack([pts1 / M, ones])
    pts2_h = np.hstack([pts2 / M, ones])  # N x 3

    # Construct A matrix
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1, _ = pts1_h[i]
        x2, y2, _ = pts2_h[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    # Solve for the SVD
    _, _, V = np.linalg.svd(A)
    F_normalized = V[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F_normalized)
    S[-1] = 0
    F_normalized = U @ np.diag(S) @ Vt
    F = T.T @ F_normalized @ T
    return F


"""
Q2.3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
"""


def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K2.T @ F @ K1
    return E


"""
Q2.3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
"""


def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    N = pts1.shape[0]
    P = np.zeros((N, 3))

    err = 0.0

    for i in range(N):
        A = np.zeros((4, 4))

        p1 = np.array([pts1[i][0], pts1[i][1], 1])
        p2 = np.array([pts2[i][0], pts2[i][1], 1])

        A[0] = p1[0] * C1[2] - C1[0]
        A[1] = p1[1] * C1[2] - C1[1]
        A[2] = p2[0] * C2[2] - C2[0]
        A[3] = p2[1] * C2[2] - C2[1]

        _, _, Vt = np.linalg.svd(A)
        Pi = Vt[-1]
        Pi = Pi / Pi[3]
        P[i] = Pi[:3]

        # Reprojection error
        p1_hat = C1 @ Pi
        p1_hat = p1_hat / p1_hat[2]
        p2_hat = C2 @ Pi
        p2_hat = p2_hat / p2_hat[2]

        err += np.linalg.norm(p1_hat[:2] - p1[:2], cv2.NORM_L2) + np.linalg.norm(
            p2_hat[:2] - p2[:2], cv2.NORM_L2
        )
    return P, err


"""
Q2.4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

"""


def compute_intersections(a, b, c, W, H):
    intersections = []
    eps = 1e-5
    if abs(b) > eps:
        y = (-c) / b
        if 0 <= y < H:
            intersections.append((0, y))
    if abs(b) > eps:
        y = (-a * (W - 1) - c) / b
        if 0 <= y < H:
            intersections.append((W - 1, y))
    if abs(a) > eps:
        x = (-c) / a
        if 0 <= x < W:
            intersections.append((x, 0))
    if abs(a) > eps:
        x = (-b * (H - 1) - c) / a
        if 0 <= x < W:
            intersections.append((x, H - 1))
    intersections = list(set(intersections))
    if len(intersections) < 2:
        return []
    sorted_pts = sorted(intersections, key=lambda p: (p[0], p[1]))
    return [sorted_pts[0], sorted_pts[-1]]


def epipolarCorrespondence(im1, im2, F, x1, y1):
    PATCH_DIM = 15
    GAUSS_SPREAD = 8.0
    LINE_SAMPLING_RATIO = 1.0
    MAX_DISPLACEMENT = 250
    MIN_SEARCH_WIDTH = 20

    x1, y1 = int(round(x1)), int(round(y1))
    H, W, _ = im2.shape
    half = PATCH_DIM // 2
    min_error = float("inf")
    opt_match = (x1, y1)

    kernel_1d = cv2.getGaussianKernel(PATCH_DIM, GAUSS_SPREAD)
    base_kernel = kernel_1d @ kernel_1d.T
    weighted_kernel = np.repeat(base_kernel[:, :, np.newaxis], 3, axis=2)
    top_y, bottom_y = max(0, y1 - half), min(H, y1 + half + 1)
    left_x, right_x = max(0, x1 - half), min(W, x1 + half + 1)
    if (bottom_y - top_y) != PATCH_DIM or (right_x - left_x) != PATCH_DIM:
        return opt_match
    ref_patch = im1[top_y:bottom_y, left_x:right_x, :].astype(float)
    src_point_homog = np.array([x1, y1, 1.0])
    epi_line = F @ src_point_homog
    line_a, line_b, line_c = epi_line[0], epi_line[1], epi_line[2]

    line_ends = compute_intersections(line_a, line_b, line_c, W, H)
    if not line_ends:
        return opt_match

    start_x, start_y = line_ends[0]
    end_x, end_y = line_ends[1]
    epi_length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    search_radius = max(int(epi_length * LINE_SAMPLING_RATIO), MIN_SEARCH_WIDTH)
    possible_matches = []
    param_t = np.linspace(0, 1, int(epi_length))
    x_coords = start_x + param_t * (end_x - start_x)
    y_coords = start_y + param_t * (end_y - start_y)

    for x_val, y_val in zip(x_coords, y_coords):
        offset = np.hypot(x_val - x1, y_val - y1)
        if offset > search_radius:
            continue
        match_x = int(round(x_val))
        match_y = int(round(y_val))
        if abs(match_x - x1) > MAX_DISPLACEMENT or abs(match_y - y1) > MAX_DISPLACEMENT:
            continue
        if 0 <= match_x < W and 0 <= match_y < H:
            possible_matches.append((match_x, match_y))

    for match_x, match_y in possible_matches:
        top_y, bottom_y = max(0, match_y - half), min(H, match_y + half + 1)
        left_x, right_x = max(0, match_x - half), min(W, match_x + half + 1)
        if (bottom_y - top_y) != PATCH_DIM or (right_x - left_x) != PATCH_DIM:
            continue
        cand_patch = im2[top_y:bottom_y, left_x:right_x, :].astype(float)
        match_error = np.sum(weighted_kernel * (ref_patch - cand_patch) ** 2)
        if match_error < min_error:
            min_error = match_error
            opt_match = (match_x, match_y)

    return opt_match


"""
Q3.1: Decomposition of the essential matrix to rotation and translation.
    Input:  im1, the first image
            im2, the second image
            k1, camera intrinsic matrix of the first frame
            k1, camera intrinsic matrix of the second frame
    Output: R, rotation
            r, translation

"""


def essentialDecomposition(im1, im2, k1, k2):
    pts1, pts2, _ = find_matched_points(im1, im2)
    M = np.max([im1.shape[0], im1.shape[1], im2.shape[0], im2.shape[1]])
    F = eightpoint(pts1, pts2, M)
    E = essentialMatrix(F, k1, k2)
    M2s = camera2(E)
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    C1 = k1 @ M1
    best_R, best_t = None, None
    best_count = 0
    best_error = float("inf")
    for i in range(4):
        M2 = M2s[:, :, i]
        C2 = k2 @ M2
        P1, err = triangulate(C1, pts1, C2, pts2)
        front1 = P1[:, 2] > 0
        R = M2[:, :3]
        t = M2[:, 3]
        P2 = np.zeros_like(P1)
        for j in range(P1.shape[0]):
            P2[j] = R @ P1[j] + t
        front2 = P2[:, 2] > 0
        front = np.logical_and(front1, front2)
        count = np.sum(front)
        if count > best_count or (count == best_count and err < best_error):
            best_count = count
            best_error = err
            best_R = R
            best_t = t
    return best_R, best_t


"""
Q3.2: Implement a monocular visual odometry.
    Input:  datafolder, the folder of the provided monocular video sequence
            GT_pose, the provided ground-truth (GT) pose for each frame
            plot=True, draw the estimated and the GT camera trajectories in the same plot Output: trajectory, the estimated camera trajectory (with scale aligned)        

"""


def visualOdometry(datafolder, GT_Pose, plot=True):
    # Replace pass by your implementation

    def load_intrinsics(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            K = np.array(
                [list(map(float, line.split())) for line in lines if line.strip()]
            )
            K1 = K[0].reshape(3, 3)
            K2 = K[1].reshape(3, 3)

        return K1, K2

    K1, K2 = load_intrinsics("data/COMP5422_HW2_DATA/Intrinsic4Recon.npz")

    gt_pose = GT_Pose  ## N x 3 x 4
    image_files = sorted(
        [f for f in os.listdir(datafolder) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )

    num_frames = len(image_files)
    images = []
    for files in image_files:
        img = cv2.imread(os.path.join(datafolder, files))
        images.append(img)

    # max_num = 50
    # images = images[:max_num]
    # gt_pose = gt_pose[:max_num]
    # num_frames = len(images)

    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = np.eye(3, 4)
    current_pose = np.eye(3, 4)

    iterable = tqdm(range(1, num_frames - 1), desc="Processing frames", unit="frame")
    gt_translations = gt_pose[:, :3, 3]
    try:
        for i in iterable:
            im1 = images[i - 1]
            im2 = images[i]

            R_rel, t_rel = essentialDecomposition(im1, im2, K1, K2)
            R_rel = R_rel.T
            t_rel = t_rel * -1
            cur_gt_pose = gt_pose[i].reshape(3, 4)
            prev_gt_pose = gt_pose[i - 1].reshape(3, 4)
            cur_gt_trans = cur_gt_pose[:, 3]
            prev_gt_trans = prev_gt_pose[:, 3]
            scale = getAbsoluteScale(prev_gt_trans, cur_gt_trans)
            t_rel_scaled = scale * (current_pose[:3, :3] @ t_rel)
            current_pose[:3, 3] += t_rel_scaled
            current_pose[:3, :3] = R_rel @ current_pose[:3, :3]
            # print(f"scale {scale}")
            # print(f"estimated t_update:\n {t_rel_scaled}")
            # print(f"GT t_update:\n {gt_translations[i + 1] - gt_translations[i]}")
            print("R_rel: ", R_rel)
            print("t_rel: ", t_rel)
            trajectory[i] = current_pose
    except Exception as e:
        print(f"Error processing frame {i}: {e}")
        pass
    est_translations = trajectory[:, :3, 3]

    np.savez("q3_2.npz", trajectory=trajectory)
    print("Saved trajectory to q3_2.npz")

    # Visualize trajectories
    if plot:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot estimated trajectory
        ax.plot(
            est_translations[:, 0],
            est_translations[:, 1],
            est_translations[:, 2],
            label="Estimated Trajectory",
            color="blue",
            marker="o",
        )
        # Plot GT trajectory
        ax.plot(
            gt_translations[:, 0],
            gt_translations[:, 1],
            gt_translations[:, 2],
            label="Ground Truth Trajectory",
            color="red",
            marker="x",
        )

        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Estimated vs Ground Truth Camera Trajectories")
        ax.legend()

        # Save plot
        plt.savefig("trajectory_plot.png")
        print("Saved trajectory plot to trajectory_plot.png")
        plt.show()

    return trajectory
