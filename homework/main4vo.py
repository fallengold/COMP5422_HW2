"""
It is a main function that can be used for you to conduct the tasks in Visual Odometry.
You run this main function to generate the expected outputs and results described in the Instruction.pdf,
by calling functions implemented in submission.py and helper.py
You are free to write it in your own style.
"""

# Insert your package here

if __name__ == "__main__":
    import numpy as np
    import os
    import cv2
    from homework.submission import essentialDecomposition, visualOdometry

    input_dir = (
        "data/COMP5422_HW2_DATA"
    )
    path1, path2 = os.path.join(input_dir, "img1.jpg"), os.path.join(
        input_dir, "img2.jpg"
    )
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    def load_intrinsics(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            K = np.array(
                [list(map(float, line.split())) for line in lines if line.strip()]
            )
            K1 = K[0].reshape(3, 3)
            K2 = K[1].reshape(3, 3)

        return K1, K2

    def load_gt_pose(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            gt_pose = np.array(
                [list(map(float, line.split())) for line in lines if line.strip()]
            )
            N = gt_pose.shape[0]
            gt_pose = gt_pose.reshape(N, 3, 4)

        return gt_pose

    intrinsics_path = "data/COMP5422_HW2_DATA/Intrinsic4Recon.npz"
    K1, K2 = load_intrinsics(intrinsics_path)

    R, t = essentialDecomposition(img1, img2, K1, K2)
    print("R: ", R)
    print("t: ", t)

    pose_path = "data/COMP5422_HW2_DATA/GTPoses.npz"

    gt_pose = load_gt_pose(pose_path)
    vid_seq_path = "data/COMP5422_HW2_DATA/vid_seq/data"

    visualOdometry(vid_seq_path, gt_pose, plot=True)
