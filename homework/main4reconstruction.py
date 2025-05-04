"""
It is a main function that can be used for you to conduct the tasks in 3D reconstruction.
You run this main function to generate the expected outputs and results described in the Instruction.pdf,
by calling functions implemented in submission.py and helper.py
You are free to write it in your own style.
"""

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


if __name__ == "__main__":
    ## test find matches points
    input_dir = "data/COMP5422_HW2_DATA"
    path1, path2 = os.path.join(input_dir, "img1.jpg"), os.path.join(
        input_dir, "img2.jpg"
    )
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
    try:
        displayEpipolarF(img1, img2, F)
    except Exception as e:
        pass

    try:
        epipolarMatchGUI(img1, img2, F)
    except Exception as e:
        pass

    print("Terminate")
