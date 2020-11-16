'''
This code is an example to demo camera pose estimation via homography.
To run this example, you need to prepare a planar target, change the settings of PHY_WIDTH and PHY_HEIGHT, and provide a calibration.yaml file containing camera intrinsic parameters.
Author: Zong-Chao Cheng
'''

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as rot
import yaml
import util

MIN_MATCH_COUNT = 10
CAMERA_PARAMETERS_INPUT_FILE = "cam1.yaml"

# Load an image of the planar target.
img1 = cv2.imread('box.png', 0)
img1_height, img1_width = img1.shape[:2]

# Specify the physical size of the planar target.
PHY_WIDTH = 85
PHY_HEIGHT = 58

# Open a camera for video capturing.
cap = cv2.VideoCapture(0)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

# Initiate SIFT detector.
sift = cv2.xfeatures2d.SIFT_create()
# Find the keypoints and descriptors with SIFT.
kp1, des1 = sift.detectAndCompute(img1, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

kps1 = [cv2.KeyPoint(f.pt[0]*PHY_WIDTH/img1_width, f.pt[1]*PHY_HEIGHT/img1_height, f.size) for f in kp1]

axes = np.float32([[0,0,0],[10, 0, 0], [0, 10, 0], [0, 0, -10]]).reshape(-1, 3)

# Load camera intrinsic parameters.
with open(CAMERA_PARAMETERS_INPUT_FILE) as f:
    loadeddict = yaml.load(f)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)

while cap.isOpened():
    ret, img = cap.read()
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des2 is not None and len(des2) > 1:
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_l_norm = cv2.undistortPoints(dst_pts, cameraMatrix=mtx, distCoeffs=dist, P=mtx)
            H, mask = cv2.findHomography(src_pts,pts_l_norm, cv2.RANSAC, 5.0)
            if H is not None:
                (R, T) = util.camera_pose_from_homography(mtx, H)
                rvec_, _ = cv2.Rodrigues(R.T)
                r = rot.from_rotvec(rvec_.T).as_euler('xyz', degrees=True)
                cv2.putText(img, 'Rotation(Euler angles): X: {:0.2f} Y: {:0.2f} Z: {:0.2f}'.format(r[0][0], r[0][1], r[0][2]), (20, int(frame_height) - 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
                cv2.putText(img, 'Translation(mm): X: {:0.2f} Y: {:0.2f} Z: {:0.2f}'.format(T[0], T[1], T[2]), (20, int(frame_height) - 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
                # Project the boundary of the planar target to the image.
                pts = np.float32([[0, 0, 0], [0, PHY_HEIGHT - 1, 0], [PHY_WIDTH - 1, PHY_HEIGHT - 1, 0], [PHY_WIDTH - 1, 0, 0]]).reshape(-1, 1, 3)
                dst, jac = cv2.projectPoints(pts, rvec_, T, mtx, dist)
                img = cv2.polylines(img, [np.int32(dst)], True, (100, 230, 240), 3, cv2.LINE_AA)
                # Project 3D axes points to the image.
                img_axes_pts, jac = cv2.projectPoints(axes, rvec_, T, mtx, dist)
                img = util.draw_axes(img, img_axes_pts)
    cv2.imshow('img', img)
    key = cv2.waitKey(delay=1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
