import cv2 as cv
import numpy as np
import glob
import os
import shutil

pattern_size = (10, 7)

src_dir = "raw_chessboards"
dst_dir = "best_chessboards"

TOP_K = 30

os.makedirs(dst_dir, exist_ok=True)

criteria = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

scores = []

for path in glob.glob(f"{src_dir}/*.jpg"):
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    found, corners = cv.findChessboardCorners(
        gray,
        pattern_size,
        flags=
            cv.CALIB_CB_ADAPTIVE_THRESH +
            cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if not found:
        continue

    corners = cv.cornerSubPix(
        gray,
        corners,
        (11,11),
        (-1,-1),
        criteria
    )

    blur_score = cv.Laplacian(gray, cv.CV_64F).var()

    x, y, w, h = cv.boundingRect(corners)
    area_ratio = (w * h) / (img.shape[0] * img.shape[1])

    score = blur_score * area_ratio

    scores.append((score, path))

scores.sort(reverse=True)

selected = scores[:TOP_K]

for i, (_, path) in enumerate(selected):
    dst_path = os.path.join(dst_dir, f"{i:03d}.jpg")
    shutil.copy(path, dst_path)

print(f"Selected {len(selected)} best images.")