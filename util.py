import numpy as np
import cv2

def draw_axes(img, img_axes_pts):
    img = cv2.line(img, tuple(img_axes_pts[0].ravel()), tuple(img_axes_pts[1].ravel()), (0, 0, 255), 2)
    img = cv2.line(img, tuple(img_axes_pts[0].ravel()), tuple(img_axes_pts[2].ravel()), (0, 255, 0), 2)
    img = cv2.line(img, tuple(img_axes_pts[0].ravel()), tuple(img_axes_pts[3].ravel()), (255, 0, 0), 2)
    return img

def camera_pose_from_homography(Kinv, H):
    '''Calculate camera pose from Homography.

    Args:
       Kinv: inverse intrinsic camera matrix
       H: homography matrix
    Returns:
       R: rotation matrix
       T: translation vector
    '''
    H = np.transpose(H)
    # the scale factor
    l = 1 / np.linalg.norm(np.dot(Kinv, H[0]))
    r1 = l * np.dot(Kinv, H[0])
    r2 = l * np.dot(Kinv, H[1])
    r3 = np.cross(r1, r2)
    T = l * np.dot(Kinv, H[2])
    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))
    U, S, V = np.linalg.svd(R, full_matrices=True)
    U = np.matrix(U)
    V = np.matrix(V)
    R = U * V
    return (R, T)
