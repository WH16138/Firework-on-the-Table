import cv2 as cv
import numpy as np
import os

pattern_size = (10, 7)
target_count = 100
save_dir = "raw_chessboards"

os.makedirs(save_dir, exist_ok=True)

cap = cv.VideoCapture(0)

criteria = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

saved = 0
last_corners = None

MIN_SHIFT = 20
MIN_BLUR_SCORE = 100
MIN_BOARD_AREA_RATIO = 0.08

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    found, corners = cv.findChessboardCorners(
        gray,
        pattern_size,
        flags=
            cv.CALIB_CB_ADAPTIVE_THRESH +
            cv.CALIB_CB_NORMALIZE_IMAGE
    )

    display = frame.copy()

    if found:
        corners = cv.cornerSubPix(
            gray,
            corners,
            (11,11),
            (-1,-1),
            criteria
        )

        cv.drawChessboardCorners(display, pattern_size, corners, found)

        # Blur Score
        blur_score = cv.Laplacian(gray, cv.CV_64F).var()

        # Board Area Ratio
        x, y, w, h = cv.boundingRect(corners)
        area_ratio = (w * h) / (frame.shape[0] * frame.shape[1])

        # Diversity Check
        shift_ok = False
        if last_corners is None:
            shift_ok = True
        else:
            shift = np.mean(
                np.linalg.norm(
                    corners.reshape(-1,2) - last_corners.reshape(-1,2),
                    axis=1
                )
            )
            shift_ok = shift > MIN_SHIFT

        if (
            blur_score > MIN_BLUR_SCORE and
            area_ratio > MIN_BOARD_AREA_RATIO and
            shift_ok
        ):
            path = os.path.join(save_dir, f"{saved:03d}.jpg")
            cv.imwrite(path, frame)

            print(
                f"Saved {saved} | "
                f"Blur={blur_score:.1f} | "
                f"Area={area_ratio:.3f}"
            )

            last_corners = corners.copy()
            saved += 1

    cv.putText(
        display,
        f"Saved: {saved}/{target_count}",
        (20,40),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv.imshow("Smart Capture", display)

    if saved >= target_count:
        break

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()