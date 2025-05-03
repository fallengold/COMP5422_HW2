from homework.submission import (
    find_matched_points,
    eightpoint,
    essentialMatrix,
    triangulate,
)
import os
import cv2
from homework.helper import displayEpipolarF, epipolarMatchGUI
import numpy as np
from homework.helper import camera2


## test find matches points
input_dir = (
    "/home/fallengold/Documents/COMP5422/HW2/COMP5422_HW2/data/COMP5422_HW2_DATA"
)
path1, path2 = os.path.join(input_dir, "img1.jpg"), os.path.join(input_dir, "img2.jpg")
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

pts1, pts2, model = find_matched_points(img1, img2)
save_dict = {
    "pts1": pts1,
    "pts2": pts2,
}
np.savez("homework/q2.2_1.npz", **save_dict)

## Output
M = np.max([img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1]])
F = eightpoint(pts1, pts2, M)
print("F: ", F)
print("M: ", M)

save_dict = {
    "F": F,
    "M": M,
}
np.savez("homework/q2.2_2.npz", **save_dict)
# displayEpipolarF(img1, img2, F)

epipolarMatchGUI(img1, img2, F)
