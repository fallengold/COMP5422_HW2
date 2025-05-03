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
    # F_normalized = refineF(F_normalized, pts1 / M, pts2 / M)
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
    pass


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
    pass


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


def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pass


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
