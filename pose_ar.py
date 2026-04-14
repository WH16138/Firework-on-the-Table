import cv2 as cv
import numpy as np
import random

# =========================
# Calibration
# =========================
data = np.load("calibration_data.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# =========================
# Chessboard
# =========================
pattern_size = (10, 7)
square_size = 25.0

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size


# =========================
# Utility
# =========================
def draw_glow(img, center, color, radius):
    overlay = img.copy()

    for i in range(2, 0, -1):
        cv.circle(overlay, center, radius * i, color, -1)

    cv.addWeighted(overlay, 0.12, img, 0.88, 0, img)


# =========================
# Particle
# =========================
class Particle:
    def __init__(self, origin, base_color):
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)

        self.pos = origin.copy()
        self.vel = direction * random.uniform(0.4, 1.1)

        self.life = random.randint(30, 50)

        noise_strength = 25
        self.color = tuple(
            int(np.clip(
                c + random.randint(-noise_strength, noise_strength),
                0, 255
            ))
            for c in base_color
        )

        self.trail = []
        self.trail_timer = 0

    def update(self):
        self.trail_timer += 1

        if self.trail_timer % 2 == 0:
            self.trail.append(self.pos.copy())

            if len(self.trail) > 4:
                self.trail.pop(0)

        self.pos += self.vel
        self.vel[2] += 0.025
        self.life -= 1

        if self.life < 10:
            self.trail.clear()

    def alive(self):
        return self.life > 0


# =========================
# Firework
# =========================
class Firework:
    def __init__(self):
        cx = (pattern_size[0] - 1) * square_size / 2
        cy = (pattern_size[1] - 1) * square_size / 2

        self.pos = np.array([
            random.uniform(cx - 30, cx + 30),
            random.uniform(cy - 30, cy + 30),
            0
        ], dtype=np.float32)

        self.vel = np.array([
            random.uniform(-1.5, 1.5),
            random.uniform(-1.5, 1.5),
            random.uniform(-5.5, -3.5)
        ], dtype=np.float32)

        self.base_color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )

        self.exploded = False
        self.trail = []
        self.trail_timer = 0

    def update(self):
        self.trail_timer += 1

        if self.trail_timer % 2 == 0:
            self.trail.append(self.pos.copy())

            if len(self.trail) > 6:
                self.trail.pop(0)

        self.pos += self.vel
        self.vel[2] += 0.10

        if self.vel[2] >= 0:
            self.exploded = True


# =========================
# State
# =========================
fireworks = []
particles = []

spawn_cooldown = 0
MAX_FIREWORKS = 4

last_pose_valid = False
last_rvec = None
last_tvec = None

lost_frames = 0
MAX_LOST = 5


# =========================
# Webcam
# =========================
cap = cv.VideoCapture(0)

criteria = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # Physics Update
    # =========================
    spawn_cooldown = max(0, spawn_cooldown - 1)

    if spawn_cooldown == 0 and len(fireworks) < MAX_FIREWORKS:
        fireworks.append(Firework())
        spawn_cooldown = random.randint(6, 20)

    new_fireworks = []

    for fw in fireworks:
        fw.update()

        if fw.exploded:
            for _ in range(40):
                particles.append(Particle(fw.pos, fw.base_color))
        else:
            new_fireworks.append(fw)

    fireworks = new_fireworks

    for p in particles:
        p.update()

    particles = [p for p in particles if p.alive()]

    # =========================
    # Pose Estimation
    # =========================
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    found, corners = cv.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        corners = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        success, rvec, tvec = cv.solvePnP(
            objp,
            corners,
            camera_matrix,
            dist_coeffs
        )

        if success:
            last_pose_valid = True
            last_rvec = rvec
            last_tvec = tvec
            lost_frames = 0
        else:
            lost_frames += 1

    else:
        lost_frames += 1

    if lost_frames >= MAX_LOST:
        last_pose_valid = False

    # =========================
    # Render
    # =========================
    if last_pose_valid:

        # ---- Batch Project Main Points ----
        all_main_points = []

        for fw in fireworks:
            all_main_points.append(fw.pos)

        for p in particles:
            all_main_points.append(p.pos)

        projected_main = None
        if all_main_points:
            projected_main, _ = cv.projectPoints(
                np.array(all_main_points, dtype=np.float32),
                last_rvec,
                last_tvec,
                camera_matrix,
                dist_coeffs
            )
            projected_main = np.int32(projected_main).reshape(-1, 2)

        # ---- Batch Project Trail Points ----
        all_trail_points = []
        trail_meta = []

        render_objects = fireworks + particles

        for obj in render_objects:
            if len(obj.trail) >= 2:
                start_idx = len(all_trail_points)
                all_trail_points.extend(obj.trail)
                trail_meta.append((obj, start_idx, len(obj.trail)))

        projected_trails = None
        if all_trail_points:
            projected_trails, _ = cv.projectPoints(
                np.array(all_trail_points, dtype=np.float32),
                last_rvec,
                last_tvec,
                camera_matrix,
                dist_coeffs
            )
            projected_trails = np.int32(projected_trails).reshape(-1, 2)

        # ---- Render Trails ----
        if projected_trails is not None:
            for obj, start, length in trail_meta:
                pts = projected_trails[start:start + length]

                for i in range(len(pts) - 1):
                    alpha = (i + 1) / len(pts)

                    if isinstance(obj, Firework):
                        color = (0, 180, 255)
                        thickness = int(1 + 2 * alpha)
                    else:
                        color = tuple(int(c * alpha) for c in obj.color)
                        thickness = 2

                    cv.line(
                        frame,
                        tuple(pts[i]),
                        tuple(pts[i + 1]),
                        color,
                        thickness
                    )

        # ---- Render Main Objects ----
        idx = 0

        for fw in fireworks:
            pt = tuple(projected_main[idx])
            draw_glow(frame, pt, (0, 220, 255), 4)
            cv.circle(frame, pt, 4, (255, 255, 255), -1)
            idx += 1

        for p in particles:
            pt = tuple(projected_main[idx])
            radius = max(2, p.life // 15)

            draw_glow(frame, pt, p.color, radius)
            cv.circle(frame, pt, radius, (255, 255, 255), -1)
            idx += 1

    # =========================
    # Status UI
    # =========================
    status_text = "Chessboard Detected" if found else "Chessboard Not Detected"
    status_color = (0, 255, 0) if found else (0, 0, 255)

    cv.putText(
        frame,
        status_text,
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )

    cv.imshow("AR Fireworks Optimized", frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()