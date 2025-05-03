from homework.submission import find_matched_points, eightpoint
import os
import cv2
from homework.helper import displayEpipolarF
import numpy as np


## test find matches points
input_dir = (
    "/home/fallengold/Documents/COMP5422/HW2/COMP5422_HW2/data/COMP5422_HW2_DATA"
)
path1, path2 = os.path.join(input_dir, "img1.jpg"), os.path.join(input_dir, "img2.jpg")
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

pts1, pts2, model = find_matched_points(img1, img2)
F = eightpoint(pts1, pts2, max(img1.shape))
displayEpipolarF(img1, img2, F)
