import glob
import os
import sys
import numpy as np
import cv2 as cv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, "camera_cal")
IMG_DIR = os.path.join(BASE_DIR, "test_images")
OUT_DIR = os.path.join(BASE_DIR, "examples")
OUT_VIDEO_DIR = os.path.join(BASE_DIR, "output_videos")
CALIB_FILE = os.path.join(CALIB_DIR, "calib.npz")

CHESSBOARD_PATTERNS = [(9, 6), (9, 5), (6, 5), (7, 6), (9, 4), (8, 6), (5, 6)]

ROI = np.array([[(100, 720), (540, 430), (760, 430), (1200, 720)]], dtype=np.int32)
SRC = np.float32([[611, 450], [697, 450], [1092, 720], [260, 720]])
DST = np.float32([[300, 0], [980, 0], [980, 720], [300, 720]])

YELLOW_LOWER = np.array([15, 80, 90])
YELLOW_UPPER = np.array([35, 255, 255])
WHITE_LOWER = np.array([0, 0, 200])
WHITE_UPPER = np.array([180, 40, 255])
CANNY_LOW, CANNY_HIGH = 100, 170

HOUGH_THRESHOLD, HOUGH_MIN_LENGTH, HOUGH_MAX_GAP = 30, 30, 40
LEFT_LIMIT, RIGHT_LIMIT, BOTTOM_LIMIT = 80, 1200, 660
LANE_TOLERANCE = 150

MIN_POINTS = 10
MIN_SPAN = 0.3
MAX_RESIDUAL = 50
LANE_WIDTH_PX = 680

MIN_LANE_PX, MAX_LANE_PX = 450, 950
N_SMOOTH = 5
SHOW_VIDEO_TILES = False

YM_PER_PIX = 30 / 720
XM_PER_PIX = 3.7 / 680

FONT = cv.FONT_HERSHEY_SIMPLEX


def read_image(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv.imdecode(data, cv.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


def calibrate():
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.001)
    objectPointsArray, imgPointsArray = [], []
    gray = None

    for path in sorted(glob.glob(os.path.join(CALIB_DIR, "calibration*.jpg"))):
        image = read_image(path)
        if image is None:
            continue
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret = False
        for (cols, rows) in CHESSBOARD_PATTERNS:
            ret, corners = cv.findChessboardCorners(gray, (cols, rows), None)
            if ret:
                break
        if not ret:
            continue
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objectPoints = np.zeros((rows * cols, 3), np.float32)
        objectPoints[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)

    if not objectPointsArray:
        raise SystemExit("Sahovska tabla nije pronadjena ni na jednoj slici u camera_cal/")

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
    np.savez(CALIB_FILE, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"Kalibracija: {len(objectPointsArray)} slika, RMS = {ret:.3f} px")
    return mtx, dist


def load_calibration(force=False):
    if force or not os.path.exists(CALIB_FILE):
        return calibrate()
    data = np.load(CALIB_FILE)
    return data["mtx"], data["dist"]


def undistort(img, mtx, dist):
    h, w = img.shape[:2]
    newCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    return cv.undistort(img, mtx, dist, None, newCameraMtx)


def find_edges(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    yellow = cv.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    white = cv.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    colorMask = cv.bitwise_or(yellow, white)
    filtered = cv.bitwise_and(img, img, mask=colorMask)

    gray = cv.cvtColor(filtered, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, CANNY_LOW, CANNY_HIGH, apertureSize=3)

    roiMask = np.zeros(edges.shape[:2], dtype="uint8")
    cv.fillPoly(roiMask, ROI, 255)
    return cv.bitwise_and(edges, edges, mask=roiMask)


def warper(img, src, dst, flags=cv.INTER_LINEAR):
    img_size = (img.shape[1], img.shape[0])
    M = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(img, M, img_size, flags=flags)


def find_lane_points(warped):
    mask = warped.copy()
    mask[:, :LEFT_LIMIT] = 0
    mask[:, RIGHT_LIMIT:] = 0
    mask[BOTTOM_LIMIT:, :] = 0

    lines = cv.HoughLinesP(mask, 1, np.pi / 180, threshold=HOUGH_THRESHOLD,
                           minLineLength=HOUGH_MIN_LENGTH, maxLineGap=HOUGH_MAX_GAP)

    left_x, left_y, right_x, right_y = [], [], [], []
    view = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = [int(v) for v in np.asarray(line).reshape(-1)[:4]]
            if abs(x2 - x1) > 0.35 * abs(y2 - y1) or y1 == y2:
                continue
            x_bottom = x1 + (x2 - x1) * (warped.shape[0] - 1 - y1) / (y2 - y1)
            if abs(x_bottom - DST[0][0]) <= LANE_TOLERANCE:
                is_left = True
            elif abs(x_bottom - DST[1][0]) <= LANE_TOLERANCE:
                is_left = False
            else:
                continue
            n = int(np.hypot(x2 - x1, y2 - y1) / 2) + 1
            xs = np.linspace(x1, x2, n)
            ys = np.linspace(y1, y2, n)
            if is_left:
                left_x += list(xs)
                left_y += list(ys)
            else:
                right_x += list(xs)
                right_y += list(ys)
            cv.line(view, (x1, y1), (x2, y2), (0, 0, 255) if is_left else (255, 0, 0), 3)

    return dict(left_x=np.array(left_x), left_y=np.array(left_y),
                right_x=np.array(right_x), right_y=np.array(right_y), view=view)


def fit_line(xs, ys, height):
    if len(xs) < MIN_POINTS or (ys.max() - ys.min()) < MIN_SPAN * height:
        return None
    fit = np.polyfit(ys, xs, 2)
    keep = np.abs(np.polyval(fit, ys) - xs) < MAX_RESIDUAL
    if keep.sum() >= MIN_POINTS:
        fit = np.polyfit(ys[keep], xs[keep], 2)
    return fit


def fit_lanes(points, height):
    left_fit = fit_line(points["left_x"], points["left_y"], height)
    right_fit = fit_line(points["right_x"], points["right_y"], height)
    if left_fit is None and right_fit is None:
        return None
    found_left, found_right = left_fit is not None, right_fit is not None

    if left_fit is None:
        left_fit = right_fit.copy()
        left_fit[2] -= LANE_WIDTH_PX
    if right_fit is None:
        right_fit = left_fit.copy()
        right_fit[2] += LANE_WIDTH_PX
    return dict(left_fit=left_fit, right_fit=right_fit, found_left=found_left, found_right=found_right)


def measure(left_fit, right_fit, height, width):
    ploty = np.linspace(0, height - 1, height)
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    left_m = np.polyfit(ploty * YM_PER_PIX, left_fitx * XM_PER_PIX, 2)
    right_m = np.polyfit(ploty * YM_PER_PIX, right_fitx * XM_PER_PIX, 2)

    y_eval = height * YM_PER_PIX
    left_radius = ((1 + (2 * left_m[0] * y_eval + left_m[1]) ** 2) ** 1.5) / max(abs(2 * left_m[0]), 1e-9)
    right_radius = ((1 + (2 * right_m[0] * y_eval + right_m[1]) ** 2) ** 1.5) / max(abs(2 * right_m[0]), 1e-9)

    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
    offset = (width / 2 - lane_center) * XM_PER_PIX
    return (left_radius + right_radius) / 2, offset, right_fitx[-1] - left_fitx[-1]


def draw_lane(undistorted, left_fit, right_fit, radius, offset):
    h = undistorted.shape[0]
    ploty = np.linspace(0, h - 1, h)
    left = np.column_stack([np.polyval(left_fit, ploty), ploty])
    right = np.column_stack([np.polyval(right_fit, ploty), ploty])

    layer = np.zeros_like(undistorted)
    cv.fillPoly(layer, [np.int32(np.vstack([left, right[::-1]]))], (0, 255, 0))
    cv.polylines(layer, [np.int32(left)], False, (0, 0, 255), 25)
    cv.polylines(layer, [np.int32(right)], False, (255, 0, 0), 25)

    result = cv.addWeighted(undistorted, 1.0, warper(layer, DST, SRC), 0.4, 0)

    side = "levo" if offset < 0 else "desno"
    radius_txt = f"Radius krivine: {radius:.0f} m" if radius < 5000 else "Radius krivine: > 5000 m (prava)"
    cv.putText(result, radius_txt, (30, 50), FONT, 1.2, (255, 255, 255), 3)
    cv.putText(result, f"Pozicija vozila: {abs(offset):.2f} m {side} od centra", (30, 100),
               FONT, 1.2, (255, 255, 255), 3)
    return result


def process(img, mtx, dist, history=None):
    h, w = img.shape[:2]
    undistorted = undistort(img, mtx, dist)
    binary = find_edges(undistorted)
    warped_edges = warper(binary, SRC, DST)
    ret, warped_edges = cv.threshold(warped_edges, 50, 255, cv.THRESH_BINARY)
    warped_color = warper(undistorted, SRC, DST)
    points = find_lane_points(warped_edges)
    fit = fit_lanes(points, h)

    left_fit = right_fit = None
    found_left = found_right = True
    if fit is not None:
        left_fit, right_fit = fit["left_fit"], fit["right_fit"]
        found_left, found_right = fit["found_left"], fit["found_right"]

    if history is not None:
        if fit is not None:
            ploty = np.linspace(0, h - 1, h)
            widths = np.polyval(right_fit, ploty) - np.polyval(left_fit, ploty)
            if len(history) == 0 or (widths.min() > MIN_LANE_PX and widths.max() < MAX_LANE_PX):
                history.append((left_fit, right_fit))
                if len(history) > N_SMOOTH:
                    history.pop(0)
        if len(history) > 0:
            left_fit = np.mean([f[0] for f in history], axis=0)
            right_fit = np.mean([f[1] for f in history], axis=0)
        else:
            left_fit = right_fit = None

    fit_view = points["view"].copy()
    steps = dict(undistorted=undistorted, binary=binary, warped_color=warped_color,
                 hough=points["view"], fit_view=fit_view)

    if left_fit is None:
        cv.putText(fit_view, "Traka nije pronadjena", (30, 50), FONT, 1.2, (255, 255, 255), 3)
        return undistorted, None, None, None, steps

    radius, offset, lane_width_px = measure(left_fit, right_fit, h, w)

    ploty = np.linspace(0, h - 1, h)
    for curve, found in ((left_fit, found_left), (right_fit, found_right)):
        pts = np.int32(np.column_stack([np.polyval(curve, ploty), ploty]))
        cv.polylines(fit_view, [pts], False, (0, 255, 255) if found else (255, 0, 255), 4)
    cv.putText(fit_view, f"Radius: {radius:.0f} m", (30, 50), FONT, 1.2, (255, 255, 255), 3)
    cv.putText(fit_view, f"Offset: {offset:.2f} m", (30, 100), FONT, 1.2, (255, 255, 255), 3)

    return draw_lane(undistorted, left_fit, right_fit, radius, offset), radius, offset, lane_width_px, steps


def make_tiles(img, steps):
    road = steps["undistorted"].copy()
    cv.polylines(road, [SRC.astype(np.int32)], True, (0, 0, 255), 3)
    return [("1 Original", img),
            ("2 Undistorted + src trapezoid", road),
            ("3 Binary", cv.cvtColor(steps["binary"], cv.COLOR_GRAY2BGR)),
            ("4 Birds-eye view", steps["warped_color"]),
            ("5 Hough lines (red = left, blue = right)", steps["hough"]),
            ("6 Polynomial fit", steps["fit_view"])]


def show_tiles(tiles):
    for i, (title, image) in enumerate(tiles):
        cv.namedWindow(title, cv.WINDOW_NORMAL)
        cv.resizeWindow(title, 520, 292)
        cv.moveWindow(title, (i % 3) * 530, (i // 3) * 332)
        cv.imshow(title, image)


def process_video(in_path, mtx, dist):
    cap = cv.VideoCapture(in_path)
    if not cap.isOpened():
        raise SystemExit(f"Video se ne moze otvoriti: {in_path}")
    os.makedirs(OUT_VIDEO_DIR, exist_ok=True)
    out_path = os.path.join(OUT_VIDEO_DIR, os.path.splitext(os.path.basename(in_path))[0] + "_out.mp4")
    writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*"mp4v"),
                            cap.get(cv.CAP_PROP_FPS) or 25, (1280, 720))
    history = []
    frames = missing = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, (1280, 720))
        result, radius, offset, width, steps = process(frame, mtx, dist, history)
        if radius is None:
            missing += 1
        writer.write(result)
        frames += 1
        if frames % 50 == 0:
            print(f"Obradjeno kadrova: {frames}")
        cv.imshow("Lane finding", result)
        if SHOW_VIDEO_TILES:
            show_tiles(make_tiles(frame, steps))
        if (cv.waitKey(1) & 0xff) == 27:
            break
    cap.release()
    writer.release()
    cv.destroyAllWindows()
    print(f"Gotovo: {frames} kadrova, bez detektovane trake: {missing}. Izlaz: {out_path}")


def main():
    args = sys.argv[1:]
    mtx, dist = load_calibration(force="--recalibrate" in args)

    if "--video" in args:
        i = args.index("--video")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            video_path = args[i + 1]
        else:
            video_path = os.path.join(BASE_DIR, "test_videos", "project_video.mp4")
            if not os.path.exists(video_path):
                video_path = os.path.join(BASE_DIR, "project_video.mp4")
        process_video(video_path, mtx, dist)
        return

    paths = [a for a in args if not a.startswith("--")]
    if not paths:
        paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not paths:
        raise SystemExit(f"Nema slika u {IMG_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"{'slika':28s} {'radijus [m]':>12s} {'pozicija [m]':>13s} {'sirina [px]':>12s}")
    for path in paths:
        name = os.path.basename(path)
        img = read_image(path)
        if img is None:
            print(f"{name:28s} slika se ne moze ucitati")
            continue
        img = cv.resize(img, (1280, 720))
        result, radius, offset, width, steps = process(img, mtx, dist)
        if radius is None:
            print(f"{name:28s} traka nije pronadjena")
        else:
            print(f"{name:28s} {radius:12.0f} {offset:13.2f} {width:12.0f}")

        tiles = make_tiles(img, steps)
        base = os.path.splitext(name)[0]
        cv.imwrite(os.path.join(OUT_DIR, base + "_result.jpg"), result)
        small = [cv.resize(image, (640, 360)) for _, image in tiles]
        cv.imwrite(os.path.join(OUT_DIR, base + "_diagnostics.jpg"),
                   np.vstack((np.hstack(small[:3]), np.hstack(small[3:]))))

        show_tiles(tiles)
        if (cv.waitKey(0) & 0xff) == 27:
            break
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()