import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

def find_corner_points(images, corners_x, corners_y):

    points = np.zeros((corners_x*corners_y,3), np.float32)
    points[:,:2] = np.mgrid[0:corners_x,0:corners_y].T.reshape(-1,2)
    points_3d = [] 
    points_2d = [] 

    fig, axs = plt.subplots(5,4, figsize=(16, 11))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (corners_x,corners_y),None)
        if ret == True:
            points_3d.append(points)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            points_2d.append(corners2)
            img = cv2.drawChessboardCorners(img, (corners_x,corners_y), corners, ret)
            
            axs[i].axis('off')
            axs[i].imshow(img)
    
    return points_3d, points_2d

def find_3d_axis(fname, corners_x, corners_y, mtx, dist):

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (corners_x,corners_y), None)
    objp = np.zeros((corners_x*corners_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:corners_x,0:corners_y].T.reshape(-1,2)
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        return draw_axis(img,corners2,imgpts)
    else:
        return img
    
    
def find_3d_cube(fname, corners_x, corners_y, mtx, dist):

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (corners_x,corners_y), None)
    objp = np.zeros((corners_x*corners_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:corners_x,0:corners_y].T.reshape(-1,2)
    axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        return draw_cube(img,corners2,imgpts)
    else:
        return img
        