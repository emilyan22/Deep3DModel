import cv2
import numpy as np
import os

# CONFIGURATION
CHECKERBOARD = (9, 6) # Inner corners (Width - 1, Height - 1)
SQUARE_SIZE = 96      # Size of one square in mm

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
if not capL.isOpened() or not capR.isOpened():
    raise RuntimeError("Could not open cam_L_32mm.avi or cam_R_32mm.avi")

debug_dir = "calibration_debug"
os.makedirs(debug_dir, exist_ok=True)

frame_count = 0
matched_pairs = 0
saved_debug = 0
img_size = None
max_debug_frames = 8
classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE


def mean_reprojection_error(obj_pts, img_pts, rvecs, tvecs, mtx, dist):
    total = 0.0
    for i, obj in enumerate(obj_pts):
        projected, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_pts[i], projected, cv2.NORM_L2) / len(projected)
        total += error
    return total / len(obj_pts)

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    img_size = grayL.shape[::-1]
    frame_count += 1

    # Find corners in both images
    foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, classic_flags)
    foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, classic_flags)

    # Fallback for harder images (OpenCV 4+)
    if not foundL and hasattr(cv2, "findChessboardCornersSB"):
        foundL, cornersL = cv2.findChessboardCornersSB(grayL, CHECKERBOARD, None)
    if not foundR and hasattr(cv2, "findChessboardCornersSB"):
        foundR, cornersR = cv2.findChessboardCornersSB(grayR, CHECKERBOARD, None)

    if foundL and foundR:
        matched_pairs += 1
        objpoints.append(objp)
        # Refine corners for sub-pixel accuracy
        cornersL2 = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        cornersR2 = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL2)
        imgpointsR.append(cornersR2)
    elif saved_debug < max_debug_frames:
        cv2.imwrite(os.path.join(debug_dir, f"miss_left_{frame_count:04d}.png"), frameL)
        cv2.imwrite(os.path.join(debug_dir, f"miss_right_{frame_count:04d}.png"), frameR)
        saved_debug += 1

capL.release()
capR.release()

# PERFORM STEREO CALIBRATION
print(f"Processed frames: {frame_count}")
print(f"Detected stereo pairs: {matched_pairs}")
if img_size is None:
    raise RuntimeError("No frames read from videos.")
if matched_pairs < 10:
    raise RuntimeError(
        "Not enough detected checkerboard pairs for calibration. "
        "Check CHECKERBOARD and review images in calibration_debug/."
    )

print("Calculating Calibration... please wait.")
# 1) Calibrate each camera independently first
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpointsL, img_size, None, None
)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpointsR, img_size, None, None
)

errL = mean_reprojection_error(objpoints, imgpointsL, rvecsL, tvecsL, mtxL, distL)
errR = mean_reprojection_error(objpoints, imgpointsR, rvecsR, tvecsR, mtxR, distR)
print(f"Left reprojection error:  {errL:.4f} px")
print(f"Right reprojection error: {errR:.4f} px")

# 2) Stereo calibration with fixed intrinsics for stability
stereo_criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-5,
)
stereo_flags = cv2.CALIB_FIX_INTRINSIC
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    mtxL,
    distL,
    mtxR,
    distR,
    img_size,
    criteria=stereo_criteria,
    flags=stereo_flags,
)
baseline = float(np.linalg.norm(T))
print(f"Stereo RMS error: {ret:.4f}")
print(f"Estimated baseline: {baseline:.2f} mm")

# SAVE THE MATRICES
np.savez('stereo_calib.npz', mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T)
print("Calibration complete. Saved to stereo_calib.npz")