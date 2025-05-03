"""
Homework2.
Replace 'pass' by your implementation.
"""

# Insert your package here
import cv2
import numpy as np
from homework.helper import refineF
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


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


def sim(im1, im2, x1, y1, x2, y2, window_size):
    # Replace pass by your implementation
    sigma = window_size / 4

    half_w = window_size // 2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    y1_start = max(0, y1 - half_w)
    y1_end = min(im1.shape[0], y1 + half_w + 1)
    x1_start = max(0, x1 - half_w)
    x1_end = min(im1.shape[1], x1 + half_w + 1)

    y2_start = max(0, y2 - half_w)
    y2_end = min(im2.shape[0], y2 + half_w + 1)
    x2_start = max(0, x2 - half_w)
    x2_end = min(im2.shape[1], x2 + half_w + 1)

    # Adjust boundaries to ensure same size
    y_start = max(y1_start, y2_start)
    y_end = min(y1_end, y2_end)
    x_start = max(x1_start, x2_start)
    x_end = min(x1_end, x2_end)

    w1 = im1[y_start:y_end, x_start:x_end].astype(np.float32)
    w2 = im2[y_start:y_end, x_start:x_end].astype(np.float32)

    gaussian_kernel = np.outer(
        np.exp(-np.arange(-half_w, half_w + 1) ** 2 / (2 * sigma**2)),
        np.exp(-np.arange(-half_w, half_w + 1) ** 2 / (2 * sigma**2)),
    )

    gaussian_kernel /= np.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel[: w1.shape[0], : w1.shape[1]]

    diff = (w1 - w2) ** 2
    dist = 0
    for c in range(3):
        dist += np.sum(diff[:, :, c] * gaussian_kernel)
    dist = np.sqrt(dist / 3)
    return dist


def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    # if len(im1.shape) == 3:
    #     im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    # if len(im2.shape) == 3:
    #     im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    im1 = im1.astype(float)
    im2 = im2.astype(float)
    # for c in range(3):
    #     im1[:, :, c] = cv2.GaussianBlur(im1[:, :, c], (5, 5), 0)
    #     im2[:, :, c] = cv2.GaussianBlur(im2[:, :, c], (5, 5), 0)

    # Normalize intensity per channel (reduce lighting effects)
    for c in range(3):
        im1[:, :, c] -= np.mean(im1[:, :, c])
        im2[:, :, c] -= np.mean(im2[:, :, c])
        im1[:, :, c] /= 255.0
        im2[:, :, c] /= 255.0

    window_size = 11
    search_range = 75
    delta = 5

    # H, W = im1.shape[:2]
    # im1 = im1.astype(np.float32)
    # im1 -= np.mean(im1)
    # im1 /= 255.0
    # im2 = im2.astype(np.float32)
    # im2 -= np.mean(im2)
    # im2 /= 255.0

    # Calculate the epipolar line in im2
    H, W = im2.shape[:2]
    l = F @ np.array([x1, y1, 1])
    a, b, c = l

    best_x2, best_y2 = 0, 0
    min_dist = float("inf")
    # lambda_dist = 0.000001  # Weight for displacement penalty

    for x2 in range(max(0, int(x1 - search_range)), min(W, int(x1 + search_range))):
        y2_base = -(a * x2 + c) / b

        for dy in range(-delta, delta + 1):
            y2 = int(y2_base + dy)
            if y2 < 0 or y2 >= H:
                continue
            dist = sim(im1, im2, x1, y1, x2, y2, window_size)
            # dist += lambda_dist * abs(y2 - y2_base)  # Add displacement penalty
            if dist < min_dist:
                min_dist = dist
                best_x2, best_y2 = x2, y2

    return best_x2, best_y2


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
    # Replace pass by your implementation
    pass


"""
Q3.2: Implement a monocular visual odometry.
    Input:  datafolder, the folder of the provided monocular video sequence
            GT_pose, the provided ground-truth (GT) pose for each frame
            plot=True, draw the estimated and the GT camera trajectories in the same plot
    Output: trajectory, the estimated camera trajectory (with scale aligned)        

"""


def visualOdometry(datafolder, GT_Pose, plot=True):
    # Replace pass by your implementation
    pass
