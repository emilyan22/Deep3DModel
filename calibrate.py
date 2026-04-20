import cv2
import numpy as np

# CONFIGURATION
CHECKERBOARD = (9, 6) # Inner corners (Width - 1, Height - 1)
SQUARE_SIZE = 25      # Size of one square in mm

# Termination criteria for sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store points
objpoints = [] # 3D points in real world
imgpointsL = [] # 2D points in Left image
imgpointsR = [] # 2D points in Right image

# Prepare object points (0,0,0), (25,0,0), (50,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Load your videos
capL = cv2.VideoCapture('cam_L_32mm.avi')
capR = cv2.VideoCapture('cam_R_32mm.avi')

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR: break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Find corners in both images
    foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if foundL and foundR:
        objpoints.append(objp)
        # Refine corners for sub-pixel accuracy
        cornersL2 = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        cornersR2 = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL2)
        imgpointsR.append(cornersR2)

# PERFORM STEREO CALIBRATION
print("Calculating Calibration... please wait.")
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, 
    None, None, None, None, grayL.shape[::-1])

# SAVE THE MATRICES
np.savez('stereo_calib.npz', mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T)
print("Calibration Complete! T[0] should be near 32mm.")