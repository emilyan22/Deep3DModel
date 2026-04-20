import cv2
import numpy as np
import os

# CONFIGURATION
CHECKERBOARD = (7, 9)  # Inner corners (cols, rows)
SQUARE_SIZE = 20.5  # mm
FRAME_STEP = 1  # Use every Nth frame to avoid near-duplicates
MAX_PAIRS = 80  # Upper bound on accepted stereo pairs
MIN_PAIRS = 8  # Minimum accepted pairs required to calibrate
MIN_BOARD_AREA_RATIO = 0.012  # Reject tiny checkerboard detections
MIN_BORDER_PX = 8  # Reject detections too close to image borders

# Preferred input names (first existing pair is used)
LEFT_VIDEO_CANDIDATES = [
    "/Users/emilyan/Downloads/videos 2/cam_L.avi",
    "videos 2/cam_L.avi",
    "calibration/videos 2/cam_L.avi",
    "cam_L_32mm.avi",
    "calibration/videos/cam_L.avi",
    "cam_L.avi",
]
RIGHT_VIDEO_CANDIDATES = [
    "/Users/emilyan/Downloads/videos 2/cam_R.avi",
    "videos 2/cam_R.avi",
    "calibration/videos 2/cam_R.avi",
    "cam_R_32mm.avi",
    "calibration/videos/cam_R.avi",
    "cam_R.avi",
]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)
classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
sb_flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY

debug_dir = "calibration_debug"
os.makedirs(debug_dir, exist_ok=True)

# Avoid OpenCL binary-cache/runtime issues on macOS during calibration.
cv2.ocl.setUseOpenCL(False)


def resolve_video_path(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def detect_corners(gray):
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, classic_flags)
    if not found and hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, sb_flags)
    if not found:
        return False, None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def board_quality(corners, width, height):
    pts = corners.reshape(-1, 2)
    cols, rows = CHECKERBOARD
    quad = np.array([pts[0], pts[cols - 1], pts[-1], pts[-cols]], dtype=np.float32)
    area = float(abs(cv2.contourArea(quad)))
    area_ratio = area / float(width * height)
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    border = min(min_x, min_y, width - max_x, height - max_y)
    return area_ratio, border


def mean_reprojection_error(obj_pts, img_pts, rvecs, tvecs, mtx, dist):
    total = 0.0
    for i, obj in enumerate(obj_pts):
        projected, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_pts[i], projected, cv2.NORM_L2) / len(projected)
        total += error
    return total / len(obj_pts)


left_video = resolve_video_path(LEFT_VIDEO_CANDIDATES)
right_video = resolve_video_path(RIGHT_VIDEO_CANDIDATES)
if left_video is None or right_video is None:
    raise RuntimeError("Could not find input videos for left/right camera.")

capL = cv2.VideoCapture(left_video)
capR = cv2.VideoCapture(right_video)
if not capL.isOpened() or not capR.isOpened():
    raise RuntimeError(f"Could not open videos: {left_video}, {right_video}")

print(f"Using left video:  {left_video}")
print(f"Using right video: {right_video}")

objpoints = []
imgpointsL = []
imgpointsR = []
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

frame_count = 0
checked_frames = 0
detected_pairs = 0
accepted_pairs = 0
rejected_quality = 0
saved_debug = 0
max_debug_frames = 12
img_size = None

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    frame_count += 1
    if frame_count % FRAME_STEP != 0:
        continue

    checked_frames += 1
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    img_size = grayL.shape[::-1]
    h, w = grayL.shape

    foundL, cornersL = detect_corners(grayL)
    foundR, cornersR = detect_corners(grayR)
    if not (foundL and foundR):
        if saved_debug < max_debug_frames:
            cv2.imwrite(os.path.join(debug_dir, f"miss_left_{frame_count:04d}.png"), frameL)
            cv2.imwrite(os.path.join(debug_dir, f"miss_right_{frame_count:04d}.png"), frameR)
            saved_debug += 1
        continue

    detected_pairs += 1
    areaL, borderL = board_quality(cornersL, w, h)
    areaR, borderR = board_quality(cornersR, w, h)
    if (
        areaL < MIN_BOARD_AREA_RATIO
        or areaR < MIN_BOARD_AREA_RATIO
        or borderL < MIN_BORDER_PX
        or borderR < MIN_BORDER_PX
    ):
        rejected_quality += 1
        if saved_debug < max_debug_frames:
            badL = frameL.copy()
            badR = frameR.copy()
            cv2.drawChessboardCorners(badL, CHECKERBOARD, cornersL, True)
            cv2.drawChessboardCorners(badR, CHECKERBOARD, cornersR, True)
            cv2.putText(
                badL,
                f"Rejected area={areaL:.3f} border={borderL:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                badR,
                f"Rejected area={areaR:.3f} border={borderR:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.imwrite(os.path.join(debug_dir, f"reject_left_{frame_count:04d}.png"), badL)
            cv2.imwrite(os.path.join(debug_dir, f"reject_right_{frame_count:04d}.png"), badR)
            saved_debug += 1
        continue

    objpoints.append(objp.copy())
    imgpointsL.append(cornersL)
    imgpointsR.append(cornersR)
    accepted_pairs += 1

    if saved_debug < max_debug_frames:
        okL = frameL.copy()
        okR = frameR.copy()
        cv2.drawChessboardCorners(okL, CHECKERBOARD, cornersL, True)
        cv2.drawChessboardCorners(okR, CHECKERBOARD, cornersR, True)
        cv2.imwrite(os.path.join(debug_dir, f"accept_left_{frame_count:04d}.png"), okL)
        cv2.imwrite(os.path.join(debug_dir, f"accept_right_{frame_count:04d}.png"), okR)
        saved_debug += 1

    if accepted_pairs >= MAX_PAIRS:
        break

capL.release()
capR.release()

print(f"Processed frames: {frame_count}")
print(f"Checked frames (every {FRAME_STEP}): {checked_frames}")
print(f"Detected stereo pairs: {detected_pairs}")
print(f"Accepted stereo pairs: {accepted_pairs}")
print(f"Rejected on quality: {rejected_quality}")

if img_size is None:
    raise RuntimeError("No frames read from videos.")
if accepted_pairs < MIN_PAIRS:
    raise RuntimeError(
        "Not enough high-quality stereo pairs. "
        "Review calibration_debug/ and verify CHECKERBOARD."
    )

print("Calculating Calibration... please wait.")

# 1) Calibrate each camera independently
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

# 2) Stereo calibration with fixed intrinsics
stereo_criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-5,
)
retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    mtxL,
    distL,
    mtxR,
    distR,
    img_size,
    criteria=stereo_criteria,
    flags=cv2.CALIB_FIX_INTRINSIC,
)
baseline = float(np.linalg.norm(T))
print(f"Stereo RMS error: {retS:.4f}")
print(f"Estimated baseline: {baseline:.2f} mm")

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY
)

np.savez(
    "stereo_calib.npz",
    mtxL=mtxL,
    distL=distL,
    mtxR=mtxR,
    distR=distR,
    R=R,
    T=T,
    E=E,
    F=F,
    R1=R1,
    R2=R2,
    P1=P1,
    P2=P2,
    Q=Q,
    reproj_err_left=errL,
    reproj_err_right=errR,
    stereo_rms=retS,
    accepted_pairs=accepted_pairs,
)
print("Calibration complete. Saved to stereo_calib.npz")