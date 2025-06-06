"""
Homework2.
Helper functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import homework.submission as sub


def _epipoles(E):
    U, S, V = np.linalg.svd(E)
    e1 = V[-1, :]
    U, S, V = np.linalg.svd(E.T)
    e2 = V[-1, :]
    return e1, e2


def displayEpipolarF(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title("Select a point in this image")
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title(
        "Verify that the corresponding point \n is on the epipolar line in this image"
    )
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc = x
        yc = y
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        if s == 0:
            error("Zero line vector in displayEpipolar")

        l = l / s

        if l[0] != 0:
            ye = sy - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        ax1.plot(x, y, "*", markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)
        plt.draw()


def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))
    return F


def _objective_F(f, pts1, pts2):
    F = _singularize(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1)) ** 2 * (
            1 / (fp1[0] ** 2 + fp1[1] ** 2) + 1 / (fp2[0] ** 2 + fp2[1] ** 2)
        )
    return r


def refineF(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2),
        F.reshape([-1]),
        maxiter=100000,
        maxfun=10000,
    )
    return _singularize(f.reshape([3, 3]))


def camera2(E):
    U, S, V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m, 0, 0], [0, m, 0], [0, 0, 0]])).dot(V)
    U, S, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    if np.linalg.det(U.dot(W).dot(V)) < 0:
        W = -W

    M2s = np.zeros([3, 4, 4])
    M2s[:, :, 0] = np.concatenate(
        [U.dot(W).dot(V), U[:, 2].reshape([-1, 1]) / abs(U[:, 2]).max()], axis=1
    )
    M2s[:, :, 1] = np.concatenate(
        [U.dot(W).dot(V), -U[:, 2].reshape([-1, 1]) / abs(U[:, 2]).max()], axis=1
    )
    M2s[:, :, 2] = np.concatenate(
        [U.dot(W.T).dot(V), U[:, 2].reshape([-1, 1]) / abs(U[:, 2]).max()], axis=1
    )
    M2s[:, :, 3] = np.concatenate(
        [U.dot(W.T).dot(V), -U[:, 2].reshape([-1, 1]) / abs(U[:, 2]).max()], axis=1
    )
    return M2s


def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title("Select a point in this image")
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title(
        "Verify that the corresponding point \n is on the epipolar line in this image"
    )
    ax2.set_axis_off()

    pts1, pts2 = [], []
    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        # if s == 0:
        #     error("Zero line vector in displayEpipolar")

        l = l / s

        if l[0] != 0:
            ye = sy - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        ax1.plot(x, y, "*", markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = sub.epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, "ro", markersize=8, linewidth=2)
        pts1.append([xc, yc])
        pts2.append([x2, y2])
        pts1_arr = np.array(pts1).reshape(-1, 2)
        pts2_arr = np.array(pts2).reshape(-1, 2)
        save_dict = {"pts1": pts1_arr, "pts2": pts2_arr, "F": F}
        np.savez("homework/q2.4_1.npz", **save_dict)
        plt.draw()


def getAbsoluteScale(pos_frame_prev, pose_frame_curr):
    """
    Estimation of scale for multiplying translation vectors
    :return: Scalar multiplier
    pos_frame_prev: the position (i.e., absolute translation) of the previous frame of GT
    pos_frame_curr: the position (i.e., absolute translation) of the current frame of GT
    """
    x_prev = pos_frame_prev[0]
    y_prev = pos_frame_prev[1]
    z_prev = pos_frame_prev[2]

    x = pose_frame_curr[0]
    y = pose_frame_curr[1]
    z = pose_frame_curr[2]

    true_vect = np.array([[x], [y], [z]])
    prev_vect = np.array([[x_prev], [y_prev], [z_prev]])

    return np.linalg.norm(true_vect - prev_vect)
