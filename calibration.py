import cv2 as cv
import numpy as np
import glob

# =========================
# Settings
# =========================
pattern_size = (10, 7)      # 내부 코너 수
square_size = 25.0         # mm

image_paths = glob.glob("best_chessboards/*.jpg")

# =========================
# Prepare Object Points
# =========================
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

criteria = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# =========================
# Find Corners
# =========================
for path in image_paths:
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    found, corners = cv.findChessboardCorners(gray, pattern_size)

    if found:
        corners = cv.cornerSubPix(
            gray,
            corners,
            (11,11),
            (-1,-1),
            criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, pattern_size, corners, found)
        cv.imshow("Corners", img)
        cv.waitKey(200)

cv.destroyAllWindows()

# =========================
# Calibration
# =========================
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("RMS Error:", ret)
print("Camera Matrix:\n", camera_matrix)
print("Distortion:\n", dist_coeffs)

np.savez(
    "calibration_data.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

print("Calibration saved.")