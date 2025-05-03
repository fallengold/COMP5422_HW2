"""
Q2.4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from homework.submission import triangulate, epipolarCorrespondence
import numpy as np
import os
import cv2


def load_vis_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        D = np.array([list(map(float, line.split())) for line in lines if line.strip()])
    return D


# Load F
F_data = np.load("homework/q2.2_2.npz", allow_pickle=True)
F = F_data["F"]
M = F_data["M"]

# Load C1, C2
C_data = np.load("homework/q2.3_3.npz", allow_pickle=True)
C1 = C_data["C1"]
C2 = C_data["C2"]
M2 = C_data["M2"]


# Load data
pts1: np.ndarray = load_vis_data("data/COMP5422_HW2_DATA/VisPts.npz")
img_dir = "data/COMP5422_HW2_DATA/"
img1_path = os.path.join(img_dir, "img1.jpg")
img2_path = os.path.join(img_dir, "img2.jpg")
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# plt.figure(figsize=(10, 8))
# plt.imshow(img1)

# # Plot the points from pts1
# plt.scatter(pts1[:, 0], pts1[:, 1], c="red", s=50, marker="o", label="Points")

# # Add labels and legend
# plt.title("Points on Image 1")
# plt.legend()
# plt.axis("off")  # Hide axes for better visualization
# plt.show()

pts2 = np.zeros_like(pts1)
N = pts1.shape[0]
for i in range(N):
    x1, y1 = int(pts1[i][0]), int(pts1[i][1])
    x2, y2 = epipolarCorrespondence(img1, img2, F, x1, y1)
    pts2[i] = np.array([x2, y2])

plt.figure(figsize=(10, 8))
plt.imshow(img2)

# Plot the points from pts1
plt.scatter(pts2[:, 0], pts2[:, 1], c="red", s=50, marker="o", label="Points")

# Add labels and legend
plt.title("Points on Image 1")
plt.legend()
plt.axis("off")  # Hide axes for better visualization
plt.show()


# Triangulate to get 3D points
points_3d, error = triangulate(C1, pts1, C2, pts2)  # Shape: (N, 3)
print("Projection error:", error)
pts1_filtered = pts1

# Print 3D point range for debugging
print("X range:", points_3d[:, 0].min(), points_3d[:, 0].max())
print("Y range:", points_3d[:, 1].min(), points_3d[:, 1].max())
print("Z range:", points_3d[:, 2].min(), points_3d[:, 2].max())

# Extract colors from img1 (optional)
colors = []
for pt in pts1_filtered:
    x, y = int(pt[0]), int(pt[1])
    if 0 <= y < img1.shape[0] and 0 <= x < img1.shape[1]:
        color = img1[y, x, :]  # BGR
        colors.append(color[::-1] / 255.0)  # Convert to RGB and normalize to [0, 1]
    else:
        colors.append([0, 0, 0])  # Default to black for out-of-bounds
colors = np.array(colors)  # Shape: (N_filtered, 3)

# Visualize 3D point cloud
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Scatter plot with colors
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=colors, s=20)
# Set labels and limits (reference Figure 5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim([-4.0, 4.0])
ax.set_ylim([-3.0, 3.0])
ax.set_zlim([-0.0, 15.0])

# Capture multiple views
views = [
    (30, 30),  # Elevation, Azimuth
    (30, 120),
    (60, 30),
]
for i, (elev, azim) in enumerate(views):
    ax.view_init(elev=elev, azim=azim)
    plt.savefig(f"3d_view_{i+1}.png")
    print(f"Saved 3d_view_{i+1}.png (elev={elev}, azim={azim})")
plt.show()
# Save F, C1, C2 to q2.4_2.npz
np.savez("q2.4_2.npz", F=F, C1=C1, C2=C2)
print("Saved F, C1, C2 to q2.4_2.npz")
