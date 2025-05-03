"""
Q2.3.3:
    1. Load point correspondences calculated and saved in Q2.2.1
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q2.3_3.npz
"""

from homework.submission import (
    find_matched_points,
    eightpoint,
    essentialMatrix,
    triangulate,
)
from homework.helper import camera2
import os
import numpy as np


def load_intrinsics(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        K = np.array([list(map(float, line.split())) for line in lines if line.strip()])
        K1 = K[0].reshape(3, 3)
        K2 = K[1].reshape(3, 3)

    return K1, K2


intrinsics_path = "/home/fallengold/Documents/COMP5422/HW2/COMP5422_HW2/data/COMP5422_HW2_DATA/Intrinsic4Recon.npz"
K1, K2 = load_intrinsics(intrinsics_path)
# Load the points
points = np.load("homework/q2.2_1.npz", allow_pickle=True)
pts1 = points["pts1"]
pts2 = points["pts2"]

# Load fundamental matrix and M
F_data = np.load("homework/q2.2_2.npz", allow_pickle=True)
F = F_data["F"]
M = F_data["M"]


E = essentialMatrix(F, K1, K2)

print("E: ", E)

M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
M2s = camera2(E)

C1 = K1 @ M1
C2s = []
for i in range(4):
    C2s.append(K2 @ M2s[:, :, i])

best_M2, best_C2, best_P = None, None, None
max_in_front = 0
best_err = float("inf")

for i, M2 in enumerate(C2s):

    C2 = C2s[i]

    P, err = triangulate(C1, pts1, C2, pts2)
    # Count the number of points in front of both cameras
    in_front = 0
    for j in range(P.shape[0]):
        P_hom = np.hstack((P[j], 1))
        p1 = C1 @ P_hom
        p2 = C2 @ P_hom

        if p1[2] > 0 and p2[2] > 0:
            in_front += 1
    print(M2)
    print(f"M2 {i}: {in_front} points in front of both cameras")

    if in_front > max_in_front or (in_front == max_in_front and err < best_err):
        max_in_front = in_front
        best_M2 = M2
        best_C2 = C2
        best_P = P
        best_err = err

np.savez("homework/q2.3_3.npz", M2=best_M2, C1=C1, C2=best_C2, P=best_P)
print("Best M2: ", best_M2)
print("Best C2: ", best_C2)
print("Points number in front of both cameras: ", max_in_front)
# print("Best P: ", best_P)
print("Best reprojection error: ", best_err)
