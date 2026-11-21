"""
Soccer broadcast footage (camera pans/zooms) -> per-frame pitch coordinates (JSON) for Unity.

A pitch-keypoint model finds known field landmarks in every frame and the image->pitch
homography is recomputed per frame. When the landmarks are too few or look wrong, the last good
calibration is carried forward by tracking how the camera moves between frames ("coasting"), so
short dropouts no longer turn into "view": false. Every calibration is then snapped onto
the white lines visible in the frame (whole centre circle, penalty arcs, touchlines, box edges), and
whichever candidate (fresh landmarks or camera-motion prediction) fits the lines best is kept. Without
usable lines, a new calibration that disagrees with the prediction by more than --gate-tol metres is
treated as a glitch and ignored. The ball is chosen over the whole clip at the end, not frame by frame.

pip install ultralytics supervision opencv-python numpy scikit-learn scipy
pip install inference        # only if you use Roboflow-hosted model IDs
pip install trackers         # only needed with supervision >= 0.31 (sv.ByteTrack was removed)

Models (each arg takes a local Ultralytics weights file, e.g. best.pt, or a Roboflow model ID
like "workspace-project/3" + ROBOFLOW_API_KEY env var):
  --players  detector with classes ball / goalkeeper / player / referee
  --pitch    pose/keypoint model with the 32 landmarks in the order of pitch_vertices() below
  --ball     optional dedicated ball detector (recommended for broadcast)

Example:
  python track_match.py match.mp4 --players football-player-detection.pt \
      --pitch football-pitch-detection.pt --ball football-ball-detection.pt --stride 2

Coordinates (metres): origin at the kickoff spot by default (--origin center): x along the touchline
in [-L/2, L/2] (goals at x = -L/2 and +L/2), y across in [-W/2, W/2]. --origin corner gives 0..L, 0..W.
Output per frame: {"t", "view", "calib": "kp"|"coast", "n_kp", "players": [...], "ball": [x, y] | null}
"view_polygon": [[x, y], ...] = the part of the pitch (plus 5 m around it) the camera shows in that frame.
("ball_interp": true marks a ball position filled in across a short dropout; a player entry with
"interp": true was filled in across a short gap in that player's track.)
(--save-h also stores the 3x3 image->pitch matrix as "H" so you can overlay the pitch on the video.)
"""
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

import argparse
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans

BALL, GOALKEEPER, PLAYER, REFEREE = "ball", "goalkeeper", "player", "referee"
MAX_COLOR_SAMPLES = 30
MOTION_SCALE = 0.5      # camera motion is estimated on a half-size image (faster, just as accurate)
GATE_FRAMES = 15        # a calibration is only vetoed while the prediction is this fresh (frames)
BALL_MEMORY = 5         # frames during which the ball is expected near its last position
MAX_BALL_STEP = 4.0     # metres per processed frame the ball may move (generous: lofted balls project badly)
BALL_MARGIN = 5.0       # metres outside the pitch where a ball detection is still accepted
OUTLIER_MIN_DIST = 40.0  # LAB distance from both team colours that marks a track as a keeper-like outlier
LIGHTNESS_WEIGHT = 0.5  # weight of LAB L (brightness) vs a/b (hue) when comparing shirt colours
TEAM_ORDER_MIN_DL = 15.0  # OpenCV L units (0-255): kits closer than this in brightness are ordered by hue
MAX_COLOR_OVERLAP = 0.15  # skip colour samples when this much of the box is covered by another person
LOCAL_MODEL_SUFFIXES = (".pt", ".onnx", ".engine", ".mlpackage", ".mlmodel", ".torchscript",
                        "_openvino_model", "_ncnn_model", "_saved_model", ".tflite")
BALL_NAMES = {"ball", "football", "sports ball", "soccer ball", "soccer-ball"}
BALL_LINK_GAP = 3       # frames a ball tracklet may skip and still continue
BALL_LINK_PX = 30.0     # image-space link radius per frame, in pixels at 1920 px width (after camera motion)
BALL_TELEPORT_COST = 0.15  # path cost per metre the ball would have to jump beyond what is physically possible
UPPER_BODY = 0.7        # ball candidates inside the top 70% of a person box are shirts/heads, not the ball
LINE_RADII = (64, 32, 16, 8)  # px search radii (at 1920 px width) for the coarse-to-fine line refinement
SPRINT = 8.0            # m/s: a player can't cover more than this (+ a little noise) between sightings
MAX_SHOWN_SPEED = 11.0  # m/s: final cap on frame-to-frame movement in the output
AMBIGUITY_RATIO = 0.9   # track stitching: a link must be this much cheaper than any alternative
LINE_SKIP_KP = 0.92     # prediction fits the lines this well -> don't spend time fitting the landmark one
UNSETTLED_FIT = 0.7     # line fit below this at the start of a shot = calibration not settled yet (dropped)
LINE_CONFIRM = 0.8      # a line fit this good counts as a fresh calibration


def pitch_vertices(L, W, pb_w=40.32, pb_l=16.5, gb_w=18.32, gb_l=5.5, cc_r=9.15, spot=11.0):
    """The 32 landmarks in meters, in the same order as Roboflow's pitch keypoint model.
    Origin = corner of one goal line; x along the touchline (0..L), y across (0..W).
    Box/circle sizes are regulation; L and W vary by stadium (World Cup: 105 x 68)."""
    return np.array([
        (0, 0), (0, (W - pb_w) / 2), (0, (W - gb_w) / 2), (0, (W + gb_w) / 2), (0, (W + pb_w) / 2), (0, W),
        (gb_l, (W - gb_w) / 2), (gb_l, (W + gb_w) / 2), (spot, W / 2),
        (pb_l, (W - pb_w) / 2), (pb_l, (W - gb_w) / 2), (pb_l, (W + gb_w) / 2), (pb_l, (W + pb_w) / 2),
        (L / 2, 0), (L / 2, W / 2 - cc_r), (L / 2, W / 2 + cc_r), (L / 2, W),
        (L - pb_l, (W - pb_w) / 2), (L - pb_l, (W - gb_w) / 2), (L - pb_l, (W + gb_w) / 2), (L - pb_l, (W + pb_w) / 2),
        (L - spot, W / 2), (L - gb_l, (W - gb_w) / 2), (L - gb_l, (W + gb_w) / 2),
        (L, 0), (L, (W - pb_w) / 2), (L, (W - gb_w) / 2), (L, (W + gb_w) / 2), (L, (W + pb_w) / 2), (L, W),
        (L / 2 - cc_r, W / 2), (L / 2 + cc_r, W / 2),
    ], dtype=np.float32)


class Model:
    """Thin wrapper so every model can be local Ultralytics weights or a Roboflow model ID."""

    def __init__(self, spec, device=None):
        spec = spec.rstrip("/")
        # Ultralytics weights and exports: .pt, ONNX, TensorRT, Core ML (.mlpackage is a folder), OpenVINO...
        local = spec.lower().endswith(LOCAL_MODEL_SUFFIXES)
        if not local and os.path.exists(spec + ".pt"):  # forgot the extension
            spec, local = spec + ".pt", True
        if local and not os.path.exists(spec):
            raise FileNotFoundError(f"weights file not found: {spec}")
        self.roboflow = not local
        # Exported models have a fixed input size and choose their own hardware (Core ML -> Neural Engine),
        # so imgsz/device are only passed to PyTorch .pt models.
        self.pytorch = spec.lower().endswith(".pt")
        self.device = device
        if self.roboflow:
            from inference import get_model
            self.m = get_model(model_id=spec, api_key=os.environ.get("ROBOFLOW_API_KEY"))
        else:
            from ultralytics import YOLO
            self.m = YOLO(spec)

    def _run(self, frame, conf, imgsz):
        if self.roboflow:
            return self.m.infer(frame, confidence=conf)[0]
        if not self.pytorch:
            return self.m(frame, conf=conf, verbose=False)[0]
        kw = {"imgsz": imgsz} if imgsz else {}
        return self.m(frame, conf=conf, verbose=False, device=self.device or DEVICE, **kw)[0]

    def detections(self, frame, conf=0.25, imgsz=None, tiles=1):
        """tiles > 1 also runs the model on a tiles x tiles grid of overlapping crops and merges the results.
        Small objects (the ball, far players) are a few pixels wide after the model shrinks a 1080p+
        frame to its input size; on crops they keep their full resolution."""
        r = self._run(frame, conf, imgsz)
        det = sv.Detections.from_inference(r) if self.roboflow else sv.Detections.from_ultralytics(r)
        if tiles <= 1:
            return det
        h, w = frame.shape[:2]
        th, tw = int(h / tiles * 1.25), int(w / tiles * 1.25)  # 25% overlap so nothing is cut in half
        parts = [det]
        for gy in range(tiles):
            for gx in range(tiles):
                y0 = min(int(gy * h / tiles), h - th) if th < h else 0
                x0 = min(int(gx * w / tiles), w - tw) if tw < w else 0
                crop = frame[y0:y0 + th, x0:x0 + tw]
                r = self._run(crop, conf, imgsz)
                d = sv.Detections.from_inference(r) if self.roboflow else sv.Detections.from_ultralytics(r)
                if len(d):
                    d.xyxy = d.xyxy + np.array([x0, y0, x0, y0], dtype=d.xyxy.dtype)
                    parts.append(d)
        merged = sv.Detections.merge(parts)
        return merged.with_nms(threshold=0.5, class_agnostic=True) if len(merged) else merged

    def keypoints(self, frame, conf=0.3, imgsz=None):
        r = self._run(frame, conf, imgsz)
        return sv.KeyPoints.from_inference(r) if self.roboflow else sv.KeyPoints.from_ultralytics(r)


class Tracker:
    """ByteTrack that works with both the old sv.ByteTrack (deprecated, removed in supervision 0.31)
    and the newer `trackers` package (pip install trackers)."""

    def __init__(self, fps, thresh):
        fr = max(1, int(round(fps)))
        self.new = not hasattr(sv, "ByteTrack")
        if self.new:
            from trackers import ByteTrackTracker
            self.t = ByteTrackTracker(frame_rate=fr, track_activation_threshold=thresh,
                                      minimum_consecutive_frames=3, high_conf_det_threshold=thresh)
        else:
            self.t = sv.ByteTrack(track_activation_threshold=thresh, minimum_consecutive_frames=3, frame_rate=fr)

    def update(self, det, frame):
        if not self.new:
            return self.t.update_with_detections(det)
        out = self.t.update(det, frame)
        return out[out.tracker_id >= 0] if len(out) else out  # -1 = not confirmed yet

    def reset(self):
        self.t.reset()


def estimate_homography(kp, verts, min_conf=0.3, min_pts=6, min_area=50.0, px_thresh=6.0):
    """Image->pitch homography from detected landmarks. Returns (H or None, landmarks used).

    The fit is done pitch->image so the RANSAC threshold is in pixels, then inverted."""
    if len(kp) == 0 or kp.xy[0].shape[0] != len(verts):
        return None, 0
    xy = kp.xy[0]
    mask = (xy[:, 0] > 1) & (xy[:, 1] > 1)  # undetected keypoints come back as (0, 0)
    # supervision >= 0.32 calls it keypoint_confidence; older versions call it confidence.
    conf = getattr(kp, "keypoint_confidence", None)
    if conf is None:
        conf = getattr(kp, "confidence", None)
    if conf is not None:
        mask &= conf[0] > min_conf
    n = int(mask.sum())
    if n < min_pts:
        return None, n
    # Reject near-collinear sets (e.g. only the halfway line visible) -> degenerate homography.
    if cv2.contourArea(cv2.convexHull(verts[mask])) < min_area:
        return None, n
    Hpi, inliers = cv2.findHomography(verts[mask], xy[mask].astype(np.float32), cv2.RANSAC, px_thresh)
    if Hpi is None or inliers is None:
        return None, n
    n_in = int(inliers.sum())
    if n_in < max(4, min_pts - 1):
        return None, n_in
    try:
        return np.linalg.inv(Hpi), n_in
    except np.linalg.LinAlgError:
        return None, n_in


def to_pitch(H, pts):
    if len(pts) == 0:
        return np.empty((0, 2))
    return cv2.perspectiveTransform(np.asarray(pts).reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def pitch_mask(frame, boxes):
    """Grass pixels with the people cut out: the only parts of the image worth tracking for camera motion."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    # Close first so the white pitch lines (not green, but the sharpest features on the grass) are kept;
    # the old erode-only version cut them out and left mostly featureless grass to track.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.erode(m, np.ones((9, 9), np.uint8))
    for x1, y1, x2, y2 in np.asarray(boxes).astype(int).reshape(-1, 4):
        m[max(y1 - 5, 0):y2 + 5, max(x1 - 5, 0):x2 + 5] = 0
    return m


def camera_motion(prev_gray, gray, mask, scale=MOTION_SCALE):
    """3x3 matrix M with x_current ~ M @ x_previous (image coordinates), or None if it can't be trusted
    (cut, close-up, motion blur)."""
    small = lambda im, interp: cv2.resize(im, None, fx=scale, fy=scale, interpolation=interp)
    pg, g, mk = small(prev_gray, cv2.INTER_AREA), small(gray, cv2.INTER_AREA), small(mask, cv2.INTER_NEAREST)
    pts = cv2.goodFeaturesToTrack(pg, 400, 0.01, 8, mask=mk)
    if pts is None or len(pts) < 20:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, pts, None)
    ok = st.ravel() == 1
    if ok.sum() < 20:
        return None
    Ms, inl = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 3.0)
    if Ms is None or inl is None or inl.sum() < 15:
        return None
    S = np.diag([scale, scale, 1.0])
    M = np.linalg.inv(S) @ Ms @ S
    if not np.isfinite(M).all() or abs(M[2, 2]) < 1e-9:
        return None
    M = M / M[2, 2]
    if not 0.5 < abs(np.linalg.det(M[:2, :2])) < 2.0:  # absurd zoom between two consecutive frames
        return None
    return M


def to_image(H, pitch_pts):
    """Pitch metres -> image pixels with the inverse of image->pitch H. Also returns a mask of points
    that are in front of the camera (w > 0)."""
    Hi = np.linalg.inv(H)
    with np.errstate(all="ignore"):
        P = np.c_[pitch_pts, np.ones(len(pitch_pts))] @ Hi.T
    front = np.isfinite(P).all(axis=1) & (P[:, 2] > 1e-9)
    return P[:, :2] / np.where(front, P[:, 2], 1.0)[:, None], front


def agree(Ha, Hb, shape, tol, L=105.0, W=68.0):
    """Do two calibrations put the same ground pixels within `tol` metres of each other?

    The old version compared fixed image points (e.g. the frame centre); from a low sideline camera
    those are often crowd or sky, beyond the horizon, where any homography gives garbage, so good
    calibrations were vetoed. Now: take a 5 m grid of pitch points that Hb says are visible, map them
    with Ha, and compare the bulk of them (75th percentile, so a far-corner extrapolation can't veto)."""
    h, w = shape[:2]
    gx, gy = np.meshgrid(np.arange(0, L + 0.1, 5.0), np.arange(0, W + 0.1, 5.0))
    grid = np.c_[gx.ravel(), gy.ravel()]
    try:
        img, front = to_image(Hb, grid)
    except np.linalg.LinAlgError:
        return False
    vis = front & (img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h)
    if vis.sum() < 4:
        return False
    back = to_pitch(Ha, img[vis])
    err = np.linalg.norm(back - grid[vis], axis=1)
    return bool(np.isfinite(err).all() and np.percentile(err, 75) < tol)


def torso_color(frame, box):
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w_img), min(y2, h_img)
    h, w = y2 - y1, x2 - x1
    crop = frame[y1 + int(0.2 * h): y1 + int(0.5 * h), x1 + int(0.25 * w): x2 - int(0.25 * w)]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 2:
        return None
    # Drop grass pixels: on thin or far players the torso crop is often half pitch.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 40) & (hsv[:, 2] >= 40)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if (~grass).sum() >= max(8, 0.2 * len(lab)):
        lab = lab[~grass]
    c = np.median(lab, axis=0)
    c[0] *= LIGHTNESS_WEIGHT  # half the weight on brightness: sun/shadow halves of the pitch, same shirt
    return c


def clip_halfplane(poly, a, b, c):
    """Sutherland-Hodgman: keep the part of polygon `poly` (Nx2) where a*x + b*y + c >= 0."""
    out = []
    n = len(poly)
    for j in range(n):
        p, q = poly[j], poly[(j + 1) % n]
        fp, fq = a * p[0] + b * p[1] + c, a * q[0] + b * q[1] + c
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            s = fp / (fp - fq)
            out.append(p + s * (q - p))
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def view_polygon(H, img_w, img_h, L, W, margin=5.0):
    """The part of the pitch the camera shows in this frame, as a polygon in pitch metres (corner origin).

    The image rectangle is first cut off just below the horizon (pixels at or above it show crowd/sky and
    land nowhere on the pitch), then mapped with H (a homography maps straight edges to straight edges,
    so the corners are enough), then clipped to the pitch plus `margin` metres. Returns [] if none of
    the pitch is in view."""
    H = np.asarray(H, dtype=np.float64)
    rect = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float64)
    h3 = H[2]
    w_ref = h3 @ np.array([img_w / 2, img_h, 1.0])  # bottom centre of the frame: on the ground
    if abs(w_ref) < 1e-12:
        return []
    sgn = np.sign(w_ref)
    # keep pixels clearly on the ground side of the horizon: w >= 0.2% of the bottom-centre value
    poly = clip_halfplane(rect, sgn * h3[0], sgn * h3[1], sgn * h3[2] - 0.002 * abs(w_ref))
    if len(poly) < 3:
        return []
    P = np.c_[poly, np.ones(len(poly))] @ H.T
    pts = P[:, :2] / P[:, 2:3]
    for a_, b_, c_ in ((1, 0, margin), (-1, 0, L + margin), (0, 1, margin), (0, -1, W + margin)):
        pts = clip_halfplane(pts, a_, b_, c_)
        if len(pts) < 3:
            return []
    return pts


def warp_boxes(xyxy, M):
    """Boxes through a 3x3 image transform (bounding box of the 4 warped corners)."""
    b = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    if not len(b):
        return b
    corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], axis=1).reshape(-1, 2)
    P = np.c_[corners, np.ones(len(corners))] @ np.asarray(M, dtype=np.float64).T
    P = (P[:, :2] / P[:, 2:3]).reshape(-1, 4, 2)
    return np.c_[P[:, :, 0].min(1), P[:, :, 1].min(1), P[:, :, 0].max(1), P[:, :, 1].max(1)]


def overlap_frac(boxes):
    """For each box, the largest fraction of its area covered by any other box."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(b) < 2:
        return np.zeros(len(b))
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    np.fill_diagonal(inter, 0)
    area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-6)
    return (inter / area[:, None]).max(axis=1)


def smooth_tracks(frames_out, window, jump_window=3, jump_m=1.0):
    """Offline, zero-lag smoothing of every track, per contiguous run of frames:
      1. jump filter: a position more than `jump_m` metres from the median of the surrounding
         `jump_window` frames is a glitch (overlapping boxes, a box briefly on the wrong person) and is
         replaced by that median;
      2. a Gaussian-weighted average over +-window/2 frames removes the remaining jitter.
    Both windows are given in frames but chosen from seconds by the caller, so the smoothing is the same
    in time at any stride (the old fixed 5 frames meant half the smoothing at 30 fps vs 15 fps)."""
    if window <= 1 and jump_window <= 1:
        return
    k = max(window // 2, 0)
    sigma = max(k / 2.0, 0.5)
    h = max(jump_window // 2, 1)
    by_track = defaultdict(list)
    for i, f in enumerate(frames_out):
        for pl in f["players"]:
            by_track[pl["id"]].append((i, pl))
    for obs in by_track.values():
        runs, cur = [], [obs[0]]
        for o in obs[1:]:
            if o[0] - cur[-1][0] > 2:
                runs.append(cur)
                cur = []
            cur.append(o)
        runs.append(cur)
        for run in runs:
            if len(run) < 3:
                continue
            idx = np.array([o[0] for o in run], dtype=float)
            xy = np.array([(o[1]["x"], o[1]["y"]) for o in run], dtype=float)
            fixed = xy.copy()
            for j in range(len(xy)):
                lo, hi = max(0, j - h), min(len(xy), j + h + 1)
                med = np.median(xy[lo:hi], axis=0)
                if np.linalg.norm(xy[j] - med) > jump_m:
                    fixed[j] = med
            if k >= 1:
                out = np.empty_like(fixed)
                for j in range(len(fixed)):
                    d = idx - idx[j]
                    wts = np.where(np.abs(d) <= k, np.exp(-0.5 * (d / sigma) ** 2), 0.0)
                    out[j] = (wts[:, None] * fixed).sum(axis=0) / wts.sum()
            else:
                out = fixed
            for (_, pl), p in zip(run, out):
                pl["x"], pl["y"] = float(p[0]), float(p[1])


def marking_samples(L, W, step=0.5, pb_w=40.32, pb_l=16.5, gb_w=18.32, gb_l=5.5, cc_r=9.15, spot=11.0):
    """Points every `step` metres along all pitch markings, including the whole centre circle and the
    penalty arcs (the keypoint model only has 4 points on the circle and none on the arcs).
    Returns (points, unit tangent of the marking at each point), both in pitch metres."""
    pts, tan = [], []

    def seg(a, b):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step))
        d = np.subtract(b, a, dtype=float)
        d /= np.linalg.norm(d)
        for s_ in np.linspace(0, 1, n):
            pts.append((a[0] + (b[0] - a[0]) * s_, a[1] + (b[1] - a[1]) * s_))
            tan.append(d)

    def rect(x0, y0, x1, y1):
        seg((x0, y0), (x1, y0)); seg((x1, y0), (x1, y1)); seg((x1, y1), (x0, y1)); seg((x0, y1), (x0, y0))

    def arc(cx, cy, r, t0, t1):
        n = max(4, int(r * abs(t1 - t0) / step))
        for t in np.linspace(t0, t1, n):
            pts.append((cx + r * np.cos(t), cy + r * np.sin(t)))
            tan.append((-np.sin(t), np.cos(t)))

    rect(0, 0, L, W)
    seg((L / 2, 0), (L / 2, W))
    arc(L / 2, W / 2, cc_r, 0, 2 * np.pi)
    t_arc = np.arccos((pb_l - spot) / cc_r)  # the part of the penalty arc outside the box
    for side in (0, 1):
        x = (lambda d: d) if side == 0 else (lambda d: L - d)
        rect(x(0), (W - pb_w) / 2, x(pb_l), (W + pb_w) / 2)
        rect(x(0), (W - gb_w) / 2, x(gb_l), (W + gb_w) / 2)
        if side == 0:
            arc(spot, W / 2, cc_r, -t_arc, t_arc)
        else:
            arc(L - spot, W / 2, cc_r, np.pi - t_arc, np.pi + t_arc)
    return np.array(pts, dtype=np.float64), np.array(tan, dtype=np.float64)


def line_mask(frame, boxes):
    """Pixels that look like painted pitch lines: thin, brighter than their surroundings, unsaturated,
    on the grass, and not on a person."""
    h, w = frame.shape[:2]
    k = max(7, int(round(15 * w / 1920)) | 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    field = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((k * 3, k * 3), np.uint8))
    field = cv2.erode(field, np.ones((5, 5), np.uint8))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    m = ((tophat > 18) & (hsv[..., 1] < 110) & (field > 0)).astype(np.uint8)
    for x1, y1, x2, y2 in np.asarray(boxes).astype(int).reshape(-1, 4):
        m[max(y1 - 4, 0):y2 + 4, max(x1 - 4, 0):x2 + 4] = 0
    return m


def _project(Hpi, P):
    """Pitch -> image. Points at or behind the camera's horizon (z <= 0) come back as NaN instead of
    dividing by zero; callers drop them."""
    with np.errstate(all="ignore"):
        Q = np.c_[P, np.ones(len(P))] @ Hpi.T
        z = Q[:, 2]
        img = Q[:, :2] / np.where(z > 1e-9, z, np.nan)[:, None]
    return img, np.nan_to_num(z, nan=-1.0, posinf=-1.0, neginf=-1.0)


def refine_with_lines(H, frame, boxes, samples, L, W):
    """Snap the calibration onto the white lines visible in the frame.

    Point-to-line fit: each projected marking sample is paired with the nearest line pixel, but only
    its distance PERPENDICULAR to the marking counts. (With plain point-to-point matching, points slide
    along a line, so the touchlines "vote" for no change and outvote the halfway line.) Robust loss,
    coarse-to-fine search radius. The result is kept only if more of the markings land on line pixels
    than before and it stays within a few metres of the start. Returns (H, fit, improved), where
    fit = fraction of visible marking samples within 3 px of a line pixel."""
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    pts, tans = samples
    h, w = frame.shape[:2]
    s = w / 1920.0
    try:
        Hpi0 = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return H, None, False
    Hpi0 = Hpi0 / Hpi0[2, 2]
    m = line_mask(frame, boxes)
    # Only pixels on the pitch (+3 m) as the current calibration sees it: cuts out the green ad
    # boards behind the touchline, whose white lettering otherwise looks like line paint.
    mg = 3.0
    edge = np.r_[[(x, -mg) for x in np.linspace(-mg, L + mg, 60)], [(L + mg, y) for y in np.linspace(-mg, W + mg, 40)],
                 [(x, W + mg) for x in np.linspace(L + mg, -mg, 60)], [(-mg, y) for y in np.linspace(W + mg, -mg, 40)]]
    poly, z = _project(Hpi0, edge)
    poly = poly[(z > 1e-9) & (np.abs(poly) < 20 * max(h, w)).all(axis=1)]
    if len(poly) >= 3:
        region = np.zeros_like(m)
        cv2.fillPoly(region, [np.round(poly).astype(np.int32)], 1)
        m &= region
    ys, xs = np.nonzero(m)
    if len(xs) < 200:
        return H, None, False
    line_px = np.c_[xs, ys].astype(np.float64)
    tree = cKDTree(line_px)

    def visible(Hpi):
        img, z = _project(Hpi, pts)
        vis = (z > 1e-9) & np.isfinite(img).all(axis=1) & (img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h)
        return img, vis

    def score(Hpi):
        img, vis = visible(Hpi)
        if vis.sum() < 30:
            return 0.0
        d, _ = tree.query(img[vis], distance_upper_bound=3.0 * s)
        return float(np.isfinite(d).mean())

    base = score(Hpi0)
    Hpi = Hpi0.copy()
    for r in LINE_RADII:
        for _ in range(2):
            img, vis = visible(Hpi)
            if vis.sum() < 30:
                break
            d, idx = tree.query(img[vis], distance_upper_bound=r * s)
            ok = np.isfinite(d)
            if ok.sum() < 30:
                break
            P, T, Q = pts[vis][ok], tans[vis][ok], line_px[idx[ok]]
            # image-space normal of the marking at each sample
            a, _ = _project(Hpi, P)
            b, _ = _project(Hpi, P + 0.25 * T)
            t_img = b - a
            good = np.isfinite(t_img).all(axis=1)
            if good.sum() < 30:
                break
            P, Q, t_img = P[good], Q[good], t_img[good]
            n = np.c_[-t_img[:, 1], t_img[:, 0]]
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            far = 10.0 * r * s  # residual for a point the trial calibration pushes past the horizon

            def resid(x):
                Hx = np.append(x, 1.0).reshape(3, 3)
                proj, _ = _project(Hx, P)
                res = ((proj - Q) * n).sum(axis=1)
                return np.where(np.isfinite(res), res, far)

            try:
                sol = least_squares(resid, Hpi.ravel()[:8], loss="soft_l1", f_scale=2.0 * s, max_nfev=50)
            except (ValueError, np.linalg.LinAlgError):
                break
            H_try = np.append(sol.x, 1.0).reshape(3, 3)
            if not np.isfinite(H_try).all():
                break  # the fit ran away: keep the last good estimate
            Hpi = H_try
    try:
        Hc = np.linalg.inv(Hpi)
    except np.linalg.LinAlgError:
        return H, base, False
    if not np.isfinite(Hc).all():
        return H, base, False
    new = score(Hpi)
    # Small corrections need only a better fit. Bigger ones (a calibration from a few noisy landmarks
    # can be 3-5 m out) need the lines to fit clearly well, and are capped at 10 m so the fit can't
    # snap onto the wrong set of lines (e.g. the goal area instead of the penalty area).
    small = agree(Hc, H, frame.shape, 4.0, L, W)
    big_ok = new >= 0.6 and new > base + 0.2 and agree(Hc, H, frame.shape, 10.0, L, W)
    if new > base + 0.02 and (small or big_ok):
        return Hc, new, True
    return H, base, False


def on_upper_body(xy, person_boxes):
    b = np.asarray(person_boxes).reshape(-1, 4)
    if not len(b):
        return False
    inside = (xy[0] >= b[:, 0]) & (xy[0] <= b[:, 2]) & (xy[1] >= b[:, 1]) & \
             (xy[1] <= b[:, 1] + UPPER_BODY * (b[:, 3] - b[:, 1]))
    return bool(inside.any())


def choose_ball_path(ball_cands, motions, views, frame_w):
    """Pick the ball over the whole clip instead of frame by frame.

    1. Link candidates into tracklets in IMAGE space, compensating for the camera pan. Image motion of
       a ball is smooth even when it's in the air, unlike its projected pitch position.
    2. Per shot (between no-view frames), choose the chain of non-overlapping tracklets with the best
       total confidence, minus a penalty for any physically impossible jump between tracklets.
       A false positive that flickers in for a frame or two can't compete with a consistent track.
    Returns {frame index: pitch position}."""
    s = frame_w / 1920.0
    tracklets, active = [], []
    for i, cands in enumerate(ball_cands):
        if not views[i]:
            active = []
            continue
        M = motions[i] if i < len(motions) else None
        # move every active tracklet's prediction into this frame's image coordinates
        for tr in active:  # last seen position, carried along with the camera pan
            if M is not None:
                p = M @ np.append(tr["pos"], 1.0)
                tr["pos"] = p[:2] / p[2]
        pairs = []
        for a_i, tr in enumerate(active):
            gap = i - tr["last"]
            pred = tr["pos"] + tr["vel"] * gap
            for c_i, (xy, conf, q) in enumerate(cands):
                d = np.linalg.norm(xy - pred)
                if d <= BALL_LINK_PX * s * gap:
                    pairs.append((d, a_i, c_i))
        used_t, used_c = set(), set()
        for d, a_i, c_i in sorted(pairs):
            if a_i in used_t or c_i in used_c:
                continue
            used_t.add(a_i); used_c.add(c_i)
            tr = active[a_i]
            xy, conf, q = cands[c_i]
            v = (xy - tr["pos"]) / (i - tr["last"])
            tr["vel"] = v if tr["n"] == 1 else 0.5 * tr["vel"] + 0.5 * v
            tr["pos"], tr["last"], tr["n"] = xy.copy(), i, tr["n"] + 1
            tr["obs"].append((i, q, conf))
        for c_i, (xy, conf, q) in enumerate(cands):
            if c_i not in used_c:
                tr = {"pos": xy.copy(), "vel": np.zeros(2), "last": i, "n": 1, "obs": [(i, q, conf)]}
                active.append(tr)
                tracklets.append(tr)
        active = [tr for tr in active if i - tr["last"] < BALL_LINK_GAP]

    # split into shots
    shot_of, shot = {}, 0
    for i, v in enumerate(views):
        if not v:
            shot += 1
        shot_of[i] = shot
    by_shot = defaultdict(list)
    for tr in tracklets:
        by_shot[shot_of[tr["obs"][0][0]]].append(tr)

    out = {}
    for trs in by_shot.values():
        trs.sort(key=lambda tr: tr["obs"][0][0])
        score = [sum(c for _, _, c in tr["obs"]) for tr in trs]
        best, prev = list(score), [-1] * len(trs)
        for j, tj in enumerate(trs):
            j0, q0 = tj["obs"][0][0], tj["obs"][0][1]
            for k in range(j):
                tk = trs[k]
                k1, q1 = tk["obs"][-1][0], tk["obs"][-1][1]
                if k1 >= j0:
                    continue
                jump = max(0.0, np.linalg.norm(q0 - q1) - MAX_BALL_STEP * (j0 - k1))
                cand = best[k] + score[j] - BALL_TELEPORT_COST * jump
                if cand > best[j]:
                    best[j], prev[j] = cand, k
        if not trs:
            continue
        j = int(np.argmax(best))
        while j >= 0:
            for i, q, _ in trs[j]["obs"]:
                out[i] = q
            j = prev[j]
    return out


def _frag_velocity(o, fps, at_end):
    """Velocity (metres per frame) over the last (or first) half second of a fragment, capped at 9 m/s."""
    if at_end:
        part = [p for p in o if p[0] >= o[-1][0] - int(fps * 0.5)]
    else:
        part = [p for p in o if p[0] <= o[0][0] + int(fps * 0.5)]
    if len(part) < 2 or part[-1][0] <= part[0][0]:
        return np.zeros(2)
    v = (np.array(part[-1][1:]) - np.array(part[0][1:])) / (part[-1][0] - part[0][0])
    sp = np.linalg.norm(v) * fps
    return v * (9.0 / sp) if sp > 9.0 else v


def stitch_tracks(frames_out, color_samples, role_votes, fps, max_gap_s, L, W):
    """Join track fragments that belong to the same player (ByteTrack hands out a new ID whenever a
    player is hidden for a while, leaves the frame, or the camera cuts).

    Every fragment end is matched to at most one later fragment start in one global assignment
    (Hungarian), so two players can never claim the same continuation. A link must:
      * start within max_gap_s seconds after the end,
      * start near where the player was heading (2 m + 7 m/s of gap, capped at 15 m),
      * be the same team (shirt colour) and the same role (keeper / outfield), where known.
    Repeated over a few passes, so A -> B -> C chains form. Rewrites the IDs in frames_out (renumbered
    1..N by first appearance) and merges the colour samples / role votes. Returns {old id: new id}."""
    from scipy.optimize import linear_sum_assignment
    max_gap = int(round(max_gap_s * fps))
    obs = defaultdict(list)
    for i, f in enumerate(frames_out):
        for pl in f["players"]:
            obs[pl["id"]].append((i, pl["x"], pl["y"]))
    if max_gap <= 0 or len(obs) < 2:
        return {t: t for t in obs}

    roles = {t: max(v, key=v.get) for t, v in role_votes.items()}
    teams, _ = assign_teams(color_samples, roles)
    is_keeper = {t: roles.get(t) == GOALKEEPER for t in obs}

    parent = {t: t for t in obs}

    def root(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    # Easy (short) gaps first, then longer ones; within each stage repeat so chains A -> B -> C form.
    stages = [g for g in (0.5, 1.5, max_gap_s) if g <= max_gap_s] or [max_gap_s]
    for stage_s in [st for st in stages for _ in range(2)]:
        stage_gap = int(round(stage_s * fps))
        groups = defaultdict(list)
        for t, o in obs.items():
            groups[root(t)].extend(o)
        frag = {}
        for g, o in groups.items():
            o.sort()
            frag[g] = o
        # team / role of a merged group = majority of its members
        g_team, g_keep = {}, {}
        for t in obs:
            g = root(t)
            if teams.get(t) is not None:
                g_team.setdefault(g, []).append(teams[t])
            g_keep.setdefault(g, []).append(is_keeper[t])
        g_team = {g: max(set(v), key=v.count) for g, v in g_team.items()}
        g_keep = {g: sum(v) * 2 >= len(v) for g, v in g_keep.items()}

        ids = list(frag)
        ends = []  # (group, last frame, predicted position function)
        for g in ids:
            o = frag[g]
            i1, x1, y1 = o[-1]
            ends.append((g, i1, np.array([x1, y1]), _frag_velocity(o, fps, at_end=True)))
        starts = [(g, frag[g][0][0], np.array(frag[g][0][1:]), _frag_velocity(frag[g], fps, at_end=False))
                  for g in ids]

        BIG = 1e6
        C = np.full((len(ends), len(starts)), BIG)
        for a_i, (ga, i1, p1, v) in enumerate(ends):
            for b_i, (gb, i0, p0, v0) in enumerate(starts):
                gap = i0 - i1
                if ga == gb or gap <= 0 or gap > stage_gap:
                    continue
                # never link two fragments that are on screen at the same time
                fa, fb = frag[ga], frag[gb]
                if fb[0][0] <= fa[-1][0] and fa[0][0] <= fb[-1][0]:
                    continue
                ta, tb = g_team.get(ga), g_team.get(gb)
                if ta is not None and tb is not None and ta != tb:
                    continue
                if g_keep.get(ga) != g_keep.get(gb):
                    continue
                gap_s = gap / fps
                if np.linalg.norm(p0 - p1) > 1.5 + SPRINT * gap_s:
                    continue  # nobody covers that distance in that time
                # Project the ending fragment forward and the new one backward to the middle of the gap
                # (each for at most 1 s); they should meet there, moving in a similar way.
                half = min(gap / 2.0, fps)
                fwd = np.clip(p1 + v * half, [-2, -2], [L + 2, W + 2])
                bwd = np.clip(p0 - v0 * half, [-2, -2], [L + 2, W + 2])
                d = np.linalg.norm(fwd - bwd)
                gate = min(2.0 + 7.0 * gap_s, 15.0)
                if d > gate:
                    continue
                dv = np.linalg.norm(v - v0) * fps  # m/s difference in velocity
                C[a_i, b_i] = d / gate + 0.5 * gap_s / max_gap_s + 0.05 * dv
        if (C < BIG).sum() == 0:
            continue
        rows, cols = linear_sum_assignment(C)
        for r_, c_ in zip(rows, cols):
            c = C[r_, c_]
            if c >= BIG:
                continue
            # Ambiguity check: skip the link if another candidate is nearly as good for this end or
            # for this start. A broken ID is better than one silently swapped onto a teammate.
            alt_r = np.delete(C[r_], c_)
            alt_c = np.delete(C[:, c_], r_)
            alt = min(alt_r.min() if len(alt_r) else BIG, alt_c.min() if len(alt_c) else BIG)
            if alt < BIG and c > AMBIGUITY_RATIO * alt:
                continue
            ga, gb = ends[r_][0], starts[c_][0]
            if root(ga) != root(gb):
                parent[root(gb)] = root(ga)

    # renumber 1..N by first appearance and rewrite everything
    first = {}
    for t, o in obs.items():
        g = root(t)
        first[g] = min(first.get(g, 10 ** 9), o[0][0])
    new_id = {g: k + 1 for k, g in enumerate(sorted(first, key=first.get))}
    mapping = {t: new_id[root(t)] for t in obs}
    for f in frames_out:
        for pl in f["players"]:
            pl["id"] = mapping[pl["id"]]
    merged_c, merged_r = defaultdict(list), defaultdict(lambda: defaultdict(int))
    for t, nid in mapping.items():
        merged_c[nid].extend(color_samples.get(t, []))
        for cls, n in role_votes.get(t, {}).items():
            merged_r[nid][cls] += n
    for nid in merged_c:  # keep the sample count bounded, evenly spread over the merged track
        if len(merged_c[nid]) > 2 * MAX_COLOR_SAMPLES:
            keep = np.linspace(0, len(merged_c[nid]) - 1, 2 * MAX_COLOR_SAMPLES).astype(int)
            merged_c[nid] = [merged_c[nid][k] for k in keep]
    color_samples.clear(); color_samples.update(merged_c)
    role_votes.clear(); role_votes.update(merged_r)
    return mapping


def _reassign_after(frames_out, color_seq, role_votes, tid, f_split, nid):
    """Frames >= f_split of track `tid` become track `nid` (entries, colour samples, role votes)."""
    for k in range(f_split, len(frames_out)):
        for pl in frames_out[k]["players"]:
            if pl["id"] == tid:
                pl["id"] = nid
    if tid in color_seq:
        color_seq[nid] = [x for x in color_seq[tid] if x[0] >= f_split]
        color_seq[tid] = [x for x in color_seq[tid] if x[0] < f_split]
    role_votes[nid] = defaultdict(int, role_votes.get(tid, {}))


def split_impossible_jumps(frames_out, color_seq, role_votes, fps, side=3):
    """Split a track where it jumps further than a player can run: the tracker put the ID on a different
    person (after losing the first one, or when two boxes merged). Compared robustly: the median of
    `side` sightings before vs after each step, so detection wobble doesn't trigger it. The parts get
    separate IDs; stitching then reconnects each to the right continuation. Returns the number of splits."""
    obs = defaultdict(list)
    for k, f in enumerate(frames_out):
        for pl in f["players"]:
            obs[pl["id"]].append((k, pl["x"], pl["y"]))
    next_id = max(list(obs) + [0]) + 1
    splits = 0
    cand = {}
    for tid, o in obs.items():
        if len(o) < 2 * side:
            continue
        cuts, j = [], side
        while j <= len(o) - side:
            before = np.median([(x, y) for _, x, y in o[j - side:j]], axis=0)
            after = np.median([(x, y) for _, x, y in o[j:j + side]], axis=0)
            dt = (np.mean([k for k, _, _ in o[j:j + side]]) - np.mean([k for k, _, _ in o[j - side:j]])) / fps
            if np.linalg.norm(after - before) > 1.5 + SPRINT * dt:
                cuts.append(o[j][0])
                j += side
                continue
            j += 1
        if cuts:
            cand[tid] = cuts
    # A calibration correction moves everyone at once; that's not an identity change. Ignore jumps that
    # at least 40% of the players on screen make within +-2 frames of each other.
    jump_frames = defaultdict(int)
    for cuts in cand.values():
        for f_split in cuts:
            for k in range(f_split - 2, f_split + 3):
                jump_frames[k] += 1
    on_screen = [len(f["players"]) for f in frames_out]
    for tid, cuts in cand.items():
        cuts = [c for c in cuts if jump_frames[c] < max(3, 0.4 * on_screen[c])]
        cur = tid
        for f_split in cuts:
            _reassign_after(frames_out, color_seq, role_votes, cur, f_split, next_id)
            cur = next_id
            next_id += 1
            splits += 1
    return splits


def remove_duplicate_tracks(frames_out, fps, near_m=1.0, share=0.6, min_shared=10):
    """A second box on the same player shows up as a track glued to another one. If two tracks sit within
    `near_m` of each other for `share` of their shared frames, drop the one with fewer frames. Fragments
    under a second that stay within `near_m` of some other player are dropped too (too short for the
    pairwise test, but the same phantom box)."""
    pos = defaultdict(dict)
    for k, f in enumerate(frames_out):
        for pl in f["players"]:
            pos[pl["id"]][k] = (pl["x"], pl["y"])
    ids, drop = sorted(pos, key=lambda t: -len(pos[t])), set()
    for i, a in enumerate(ids):
        if a in drop:
            continue
        for b in ids[i + 1:]:
            if b in drop:
                continue
            shared = set(pos[a]) & set(pos[b])
            if len(shared) < min_shared:
                continue
            close = np.mean([np.hypot(pos[a][k][0] - pos[b][k][0], pos[a][k][1] - pos[b][k][1]) < near_m
                             for k in shared])
            if close >= share:
                drop.add(b)  # b has fewer frames (ids sorted longest first)
    for t in ids:
        if t in drop or len(pos[t]) >= fps:
            continue
        dn = []
        for k, (x, y) in pos[t].items():
            others = [np.hypot(x - ox, y - oy) for o, (ox, oy) in
                      ((o, pos[o][k]) for o in ids if o != t and o not in drop and k in pos[o])]
            if others:
                dn.append(min(others))
        if dn and np.median(dn) < near_m:
            drop.add(t)
    if drop:
        for f in frames_out:
            f["players"] = [pl for pl in f["players"] if pl["id"] not in drop]
    return len(drop)


def cap_speed(frames_out, fps, vmax=MAX_SHOWN_SPEED):
    """Final safety net: no frame-to-frame move faster than vmax. A forward and a backward pass each
    limit the step, and are averaged, so the result doesn't lag; moves under vmax are left untouched."""
    step = vmax / fps
    by_track = defaultdict(list)
    for k, f in enumerate(frames_out):
        for pl in f["players"]:
            by_track[pl["id"]].append((k, pl))
    for o in by_track.values():
        runs, cur = [], [o[0]]
        for x in o[1:]:
            if x[0] - cur[-1][0] > 1:
                runs.append(cur)
                cur = []
            cur.append(x)
        runs.append(cur)
        for run in runs:
            if len(run) < 2:
                continue
            P = np.array([(pl["x"], pl["y"]) for _, pl in run], dtype=float)
            if (np.linalg.norm(np.diff(P, axis=0), axis=1) <= step).all():
                continue

            def limited(Q):
                out = Q.copy()
                for j in range(1, len(Q)):
                    d = Q[j] - out[j - 1]
                    n = np.linalg.norm(d)
                    out[j] = out[j - 1] + (d * step / n if n > step else d)
                return out

            P2 = 0.5 * (limited(P) + limited(P[::-1])[::-1])
            for (_, pl), p in zip(run, P2):
                pl["x"], pl["y"] = float(p[0]), float(p[1])


def fill_player_gaps(frames_out, max_gap, fps):
    """Linearly fill short gaps inside each track (missed detections, brief occlusions), only through
    frames that have a pitch view. Filled entries get "interp": true."""
    if max_gap <= 0:
        return 0
    obs = defaultdict(list)
    for i, f in enumerate(frames_out):
        for pl in f["players"]:
            obs[pl["id"]].append((i, pl))
    n = 0
    for tid, o in obs.items():
        for (i0, a), (i1, b) in zip(o[:-1], o[1:]):
            if not 1 < i1 - i0 <= max_gap + 1:
                continue
            if not all(frames_out[j]["view"] for j in range(i0 + 1, i1)):
                continue
            if np.hypot(b["x"] - a["x"], b["y"] - a["y"]) > 1.0 + SPRINT * (i1 - i0) / fps:
                continue  # can't be one person: don't draw a sprint between them
            for j in range(i0 + 1, i1):
                s = (j - i0) / (i1 - i0)
                frames_out[j]["players"].append({"id": tid, "x": a["x"] + s * (b["x"] - a["x"]),
                                                 "y": a["y"] + s * (b["y"] - a["y"]), "interp": True})
                n += 1
    return n


def interpolate_ball(frames_out, max_gap):
    """Fill short ball dropouts (motion blur, occlusion by a player) linearly. Only between two real
    detections, only through frames with a pitch view. Filled frames get "ball_interp": true."""
    last = None
    for i, f in enumerate(frames_out):
        if not f["view"]:
            last = None
            continue
        if f["ball"] is None:
            continue
        if last is not None and 1 < i - last <= max_gap + 1:
            a, b = np.array(frames_out[last]["ball"]), np.array(f["ball"])
            for j in range(last + 1, i):
                s = (j - last) / (i - last)
                p = a + s * (b - a)
                frames_out[j]["ball"] = [round(float(p[0]), 2), round(float(p[1]), 2)]
                frames_out[j]["ball_interp"] = True
        last = i


def fit_team_colors(color_samples, roles):
    """Two team shirt colours from the tracks' median colours (goalkeepers left out; colour outliers such
    as a keeper or referee the detector called a 'player' are removed and the fit repeated).
    Returns (KMeans, track ids, their median colours, outlier mask) or None."""
    tids = [t for t, c in color_samples.items() if len(c)]
    if len(tids) < 4:
        return None
    X = np.array([np.median(color_samples[t], axis=0) for t in tids])
    enough = np.array([len(color_samples[t]) >= 3 for t in tids])
    use = np.array([roles.get(t) != GOALKEEPER for t in tids])
    if use.sum() < 2:
        return None
    # With similar kits (white vs light blue) a referee's colour is so different that a plain 2-way
    # split puts him alone in one "team" and both real teams in the other. So: a cluster with only a
    # couple of members is set aside as outliers and the split is redone.
    outlier = np.zeros(len(tids), bool)
    for _ in range(4):
        keep = use & ~outlier
        if keep.sum() < 4:
            break
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X[keep])
        sizes = np.bincount(km.labels_, minlength=2)
        small = int(sizes.argmin())
        if sizes[small] >= max(3, 0.2 * keep.sum()):
            break
        idx = np.nonzero(keep)[0][km.labels_ == small]
        outlier[idx[enough[idx]]] = True
        if not enough[idx].any():
            break
    keep = use & ~outlier
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X[keep] if keep.sum() >= 2 else X[use])
    # plus anything far from both team colours
    d = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None], axis=2).min(axis=1)
    outlier |= enough & (d > max(OUTLIER_MIN_DIST, 3.0 * (np.median(d[use & ~outlier]) + 1e-6)))
    keep = use & ~outlier
    if keep.sum() >= 2:  # refit without the outliers so they can't drag a cluster centre
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X[keep])
    canonical_team_order(km)
    return km, tids, X, outlier


def canonical_team_order(km):
    """KMeans numbers its two clusters arbitrarily, so team 0/1 could swap between runs (or strides) of
    the same video. Fix the order by the kits themselves: team 0 = the lighter shirt; when both are about
    equally light, team 0 = the warmer one (higher LAB a + b: red/orange/yellow before blue/green)."""
    c = km.cluster_centers_
    L0, L1 = c[0, 0] / LIGHTNESS_WEIGHT, c[1, 0] / LIGHTNESS_WEIGHT  # undo the brightness down-weighting
    if abs(L0 - L1) >= TEAM_ORDER_MIN_DL:
        swap = L1 > L0
    else:
        swap = (c[1, 1] + c[1, 2]) > (c[0, 1] + c[0, 2])
    if swap:
        km.cluster_centers_ = c[::-1].copy()
        if hasattr(km, "labels_"):
            km.labels_ = 1 - km.labels_
    return km


def kit_rgb(center):
    """Cluster centre (OpenCV LAB with down-weighted L) -> [r, g, b] 0-255 and '#rrggbb'."""
    lab = np.array(center, dtype=np.float64).copy()
    lab[0] /= LIGHTNESS_WEIGHT
    px = np.clip(np.round(lab), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    r, g, b = (int(v) for v in cv2.cvtColor(px, cv2.COLOR_LAB2RGB)[0, 0])
    return [r, g, b], f"#{r:02x}{g:02x}{b:02x}"


def sample_votes(km, colors):
    """Team vote per colour sample; None where the sample is about equally close to both teams."""
    C = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    d = np.linalg.norm(C[:, None, :] - km.cluster_centers_[None], axis=2)
    lab = d.argmin(axis=1)
    clear = np.abs(d[:, 0] - d[:, 1]) > 0.15 * d.max(axis=1)
    return [int(l) if c else None for l, c in zip(lab, clear)]


def assign_teams(color_samples, roles):
    """Team per track = majority vote of its individual shirt-colour samples (a few bad samples, e.g. a
    half-hidden player, no longer decide it), falling back to the median colour.
    Returns (team per track id, set of colour outliers)."""
    fit = fit_team_colors(color_samples, roles)
    if fit is None:
        return {}, set()
    km, tids, X, outlier = fit
    teams = {}
    median_lab = km.predict(X)
    for t, ml in zip(tids, median_lab):
        v = [x for x in sample_votes(km, color_samples[t]) if x is not None]
        teams[t] = max((0, 1), key=v.count) if len(v) >= 3 else int(ml)
    return teams, {t for t, o in zip(tids, outlier) if o}


def split_team_switches(frames_out, color_seq, role_votes, fps):
    """Split tracks whose shirt colour switches team partway through and stays switched: the tracker
    swapped two players of opposite teams (typically in a tackle). The part after the switch gets a new
    ID, so track stitching can join each part to the right player. Returns the number of splits."""
    color_samples = {t: [c for _, c in seq] for t, seq in color_seq.items()}
    roles = {t: max(v, key=v.get) for t, v in role_votes.items()}
    fit = fit_team_colors(color_samples, roles)
    if fit is None:
        return 0
    km, tids, X, outlier = fit
    skip = {t for t, o in zip(tids, outlier) if o} | {t for t in roles if roles[t] == GOALKEEPER}
    w = max(5, int(round(0.7 * fps)))  # samples each side of a switch
    next_id = max([pl["id"] for f in frames_out for pl in f["players"]] + [0]) + 1
    new_seqs, splits = {}, 0
    for tid, seq in list(color_seq.items()):
        if tid in skip or len(seq) < 2 * w:
            continue
        seq = sorted(seq, key=lambda x: x[0])
        votes = sample_votes(km, [c for _, c in seq])
        cut_points, j = [], w
        while j <= len(seq) - w:
            left = [v for v in votes[j - w:j] if v is not None]
            right = [v for v in votes[j:j + w] if v is not None]
            if len(left) >= 0.6 * w and len(right) >= 0.6 * w:
                la, ra = max((0, 1), key=left.count), max((0, 1), key=right.count)
                if la != ra and left.count(la) >= 0.8 * len(left) and right.count(ra) >= 0.8 * len(right):
                    cut_points.append(j)
                    j += w
                    continue
            j += 1
        if not cut_points:
            continue
        bounds = [0] + cut_points + [len(seq)]
        for a_, b_ in zip(bounds[1:-1], bounds[2:]):
            # this part runs from its first sample up to (not including) the next part's first sample,
            # covering the frames in between that had no colour sample (player partly hidden)
            f0 = seq[a_][0]
            f_end = seq[b_][0] if b_ < len(seq) else len(frames_out)
            nid = next_id
            next_id += 1
            for k in range(f0, f_end):
                for pl in frames_out[k]["players"]:
                    if pl["id"] == tid:
                        pl["id"] = nid
            new_seqs[nid] = seq[a_:b_]
            role_votes[nid] = defaultdict(int, role_votes[tid])
            splits += 1
        color_seq[tid] = seq[:cut_points[0]]
    color_seq.update(new_seqs)
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--players", required=True)
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--ball")
    ap.add_argument("--out", default="match.json")
    ap.add_argument("--length", type=float, default=105.0, help="pitch length in m")
    ap.add_argument("--width", type=float, default=68.0, help="pitch width in m")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--imgsz", type=int, default=1280, help="detector input size (bigger = better ball)")
    ap.add_argument("--device", help="e.g. mps (Apple GPU), cuda, cpu; default = Ultralytics' choice")
    # calibration
    ap.add_argument("--kp-conf", type=float, default=0.25, help="pitch model detection confidence")
    ap.add_argument("--kp-min-conf", type=float, default=0.3, help="min confidence per landmark")
    ap.add_argument("--kp-imgsz", type=int, default=None, help="pitch model input size (default: model's own)")
    ap.add_argument("--kp-every", type=int, default=3,
                    help="run the pitch keypoint model every Nth processed frame (always when the line fit is "
                         "weak or there is no calibration); 1 = every frame")
    ap.add_argument("--player-imgsz", type=int, default=None,
                    help="player detector input size (default: --imgsz). 960 is usually enough for players and "
                         "~40%% faster; keep --imgsz 1280 for the ball")
    ap.add_argument("--min-kp", type=int, default=6, help="min landmarks for a fresh calibration")
    ap.add_argument("--coast", type=float, default=3.0,
                    help="max seconds to follow the camera without a fresh calibration")
    ap.add_argument("--gate-tol", type=float, default=4.0,
                    help="metres: a new calibration further than this from the prediction is ignored")
    # detection / tracking
    ap.add_argument("--det-conf", type=float, default=0.15, help="player detector confidence")
    ap.add_argument("--ball-conf", type=float, default=0.2)
    ap.add_argument("--track-thresh", type=float, default=0.25, help="ByteTrack activation threshold")
    ap.add_argument("--margin", type=float, default=1.5, help="metres outside the lines where people are kept")
    ap.add_argument("--smooth-seconds", type=float, default=0.35,
                    help="length of the offline (zero-lag) position smoothing window in seconds")
    ap.add_argument("--smooth-window", type=int, default=None,
                    help="smoothing window in frames instead of seconds (0/1 = off)")
    ap.add_argument("--ball-tiles", type=int, default=1,
                    help="also run the ball detector on an NxN grid of crops (2 = much better recall on "
                         "small/far balls, ~5x slower ball pass)")
    ap.add_argument("--player-tiles", type=int, default=1,
                    help="also run the player detector on an NxN grid of crops (finds small far-side players)")
    ap.add_argument("--no-line-refine", dest="line_refine", action="store_false",
                    help="don't snap the calibration onto the detected white pitch lines")
    ap.add_argument("--no-cmc", dest="cmc", action="store_false",
                    help="don't compensate camera pans when matching player boxes between frames")
    ap.add_argument("--stitch-gap", type=float, default=3.0,
                    help="seconds: join a track that ends to one that starts this soon after, near where the "
                         "player was heading, same team/role (0 = off)")
    ap.add_argument("--fill-gap", type=float, default=1.0,
                    help="seconds: fill gaps inside a track up to this long by interpolation (0 = off)")
    ap.add_argument("--ball-gap", type=int, default=BALL_MEMORY,
                    help="fill ball dropouts up to this many frames by interpolation (0 = off)")
    ap.add_argument("--save-h", action="store_true", help="store the image->pitch matrix in every frame")
    ap.add_argument("--debug-teams", action="store_true",
                    help="store each player's raw tracker ID and that frame's own shirt-colour team vote, so "
                         "viz_match.py can flag frames where the colour disagrees with the assigned team")
    ap.add_argument("--origin", choices=["center", "corner"], default="center",
                    help="output coordinates: 'center' = (0,0) at the kickoff spot, x in [-L/2, L/2] along the "
                         "touchline, y in [-W/2, W/2] across; 'corner' = the old 0..L, 0..W")
    a = ap.parse_args()

    L, W = a.length, a.width
    verts = pitch_vertices(L, W)

    def on_pitch(p, m):
        return -m <= p[0] <= L + m and -m <= p[1] <= W + m

    players_model = Model(a.players, a.device)
    pitch_model = Model(a.pitch, a.device)
    ball_model = Model(a.ball, a.device) if a.ball else None

    info = sv.VideoInfo.from_video_path(a.video)
    frame_w = info.width
    fps = info.fps / a.stride
    tracker = Tracker(fps, a.track_thresh)
    max_coast = int(a.coast * fps)

    frames_out = []
    color_seq = defaultdict(list)                   # track id -> [(frame, shirt colour)]
    color_by_entry = {}                             # (frame, raw track id) -> shirt colour, for --debug-teams
    role_votes = defaultdict(lambda: defaultdict(int))
    stats = defaultdict(int)

    last_H, drift, had_view = None, 0, False       # calibration state
    prev_frame = prev_gray = None                    # camera-motion state (rotated at the top of the loop)
    prev_boxes = np.empty((0, 4))
    cur_frame = cur_gray = None
    cur_boxes = np.empty((0, 4))
    id_offset, max_tid = 0, 0                       # ByteTrack restarts its IDs after reset(); keep ours unique
    ball_cands, motions = [], []                    # per processed frame, for the offline ball choice
    stab = np.eye(3)                                # current image -> pan-stabilised image (for the tracker)
    timing = defaultdict(float)                     # seconds per stage, printed at the end
    last_fit = None                                 # line fit of the previous frame (decides if keypoints run)
    line_samples = marking_samples(L, W) if a.line_refine else None
    t0 = time.time()

    for i, frame in enumerate(sv.get_video_frames_generator(a.video, stride=a.stride)):
        t = round(i / fps, 3)
        print(f"frame {i * a.stride}/{info.total_frames}  ({i / max(time.time() - t0, 1e-6):.1f} processed fps)")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame, prev_gray, prev_boxes = cur_frame, cur_gray, cur_boxes

        # 0) Detect people first: their boxes are cut out of the line mask used to refine the calibration
        #    (white shirts and socks would otherwise look like pitch lines).
        _t = time.perf_counter()
        det = players_model.detections(frame, conf=a.det_conf, imgsz=a.player_imgsz or a.imgsz,
                                       tiles=a.player_tiles)
        timing["player detector"] += time.perf_counter() - _t
        names = det.data.get("class_name", np.array([], dtype=str))
        boxes_now = det.xyxy[names != BALL] if len(det) else np.empty((0, 4))
        cur_frame, cur_gray, cur_boxes = frame, gray, boxes_now

        # 1) Calibration: where is the camera? Predict from the last frame, then check against landmarks.
        H_pred, M = None, None
        if prev_gray is not None:
            _t = time.perf_counter()
            M = camera_motion(prev_gray, gray, pitch_mask(prev_frame, prev_boxes))
            timing["camera motion"] += time.perf_counter() - _t
        motions.append(M)
        if M is not None and a.cmc:
            try:
                stab = stab @ np.linalg.inv(M)  # undo this frame's camera motion
            except np.linalg.LinAlgError:
                pass
        if last_H is not None and prev_gray is not None:
            if M is not None:
                H_pred = last_H @ np.linalg.inv(M)
            elif drift < 3:
                H_pred = last_H  # one blurry frame: assume the camera held still

        # The keypoint model is only needed every few frames: in between, camera-motion tracking plus the
        # line fit keep the calibration. It always runs when there is no calibration or the fit is weak.
        need_kp = (i % max(1, a.kp_every) == 0 or H_pred is None or last_fit is None or last_fit < LINE_CONFIRM)
        H_new, n_kp = None, 0
        if need_kp:
            _t = time.perf_counter()
            H_new, n_kp = estimate_homography(
                pitch_model.keypoints(frame, conf=a.kp_conf, imgsz=a.kp_imgsz), verts, a.kp_min_conf, a.min_kp)
            timing["pitch keypoints"] += time.perf_counter() - _t
        else:
            stats["kp_not_run"] += 1
        if H_pred is not None and drift >= max_coast:
            H_pred = None

        # 1b) Snap each candidate calibration onto every visible white line (whole centre circle, penalty
        #     arcs, touchlines, box edges), not just the handful of landmark points, then keep whichever
        #     candidate fits the lines best. A landmark calibration from 5-6 noisy points can be metres
        #     out; a camera-motion prediction that still sits on the lines is better than it.
        fit = None
        if line_samples is not None and (H_new is not None or H_pred is not None):
            options = []
            for Hc, label in ((H_pred, "coast"), (H_new, "kp")):
                if Hc is not None and label == "kp" and options and options[0][0] >= LINE_SKIP_KP:
                    stats["kp_skipped"] += 1
                    continue  # the prediction already fits the lines very well: no need to try landmarks
                if Hc is not None:
                    _t = time.perf_counter()
                    Hr, f_r, imp = refine_with_lines(Hc, frame, boxes_now, line_samples, L, W)
                    timing["line fit"] += time.perf_counter() - _t
                    options.append((f_r if f_r is not None else -1.0, label, Hr, imp))
            best_fit, calib, H, improved = max(options, key=lambda o: o[0])
            if best_fit >= 0:
                fit = best_fit
                stats["line_refined"] += improved
                if len(options) == 2 and calib == "coast":
                    stats["gated"] += 1  # landmarks fitted the lines worse than the camera prediction
                # fresh landmarks, or a prediction confirmed by the lines, resets the coasting clock
                drift = 0 if (calib == "kp" or best_fit >= LINE_CONFIRM) else drift + 1
            else:  # no usable lines in view: fall back to the landmark-vs-prediction gate below
                fit = None
        if fit is None:
            if H_new is not None and H_pred is not None and drift < GATE_FRAMES \
                    and not agree(H_new, H_pred, frame.shape, a.gate_tol, L, W):
                stats["gated"] += 1  # glitch or left/right flip: keep following the camera instead
                H_new = None
            if H_new is not None:
                H, drift, calib = H_new, 0, "kp"
            elif H_pred is not None:
                H, drift, calib = H_pred, drift + 1, "coast"
            else:
                H, calib = None, None
        last_H = H
        last_fit = fit if H is not None else None

        if H is None:
            # Close-up / replay / cutaway. Tracks won't survive the cut anyway.
            if had_view:
                tracker.reset()
                stab = np.eye(3)
                id_offset, had_view = max_tid + 1, False
            frames_out.append({"t": t, "view": False, "n_kp": n_kp, "players": [], "ball": None})
            ball_cands.append([])
            stats["noview"] += 1
            continue
        had_view = True
        stats[calib] += 1

        # 2) Track people in image space.
        person = det[np.isin(names, [PLAYER, GOALKEEPER])].with_nms(threshold=0.6, class_agnostic=True)
        # Camera-motion compensation: ByteTrack matches boxes by overlap with where they were last frame.
        # In a fast pan every box jumps further than its own width, so all tracks were lost at once
        # (15 tracked players -> 4 in 6-10 m/s pans). Feed the tracker boxes in pan-stabilised image
        # coordinates instead, and put the real boxes back afterwards.
        person_boxes = person.xyxy.copy() if len(person) else np.empty((0, 4))  # real image boxes
        if a.cmc and len(person):
            person.data["orig_xyxy"] = person.xyxy.copy()
            person.xyxy = warp_boxes(person.xyxy, stab).astype(person.xyxy.dtype)
        _t = time.perf_counter()
        people = tracker.update(person, frame)
        timing["tracker"] += time.perf_counter() - _t
        if a.cmc and len(people):
            if "orig_xyxy" in people.data:
                people.xyxy = np.asarray(people.data["orig_xyxy"]).astype(people.xyxy.dtype)
            else:  # tracker didn't carry the extra field: map back with the inverse warp
                people.xyxy = warp_boxes(people.xyxy, np.linalg.inv(stab)).astype(people.xyxy.dtype)

        players = []
        if len(people):
            feet = to_pitch(H, people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
            covered = overlap_frac(people.xyxy)
            fh = frame.shape[0]
            classes = people.data.get("class_name", np.full(len(people), PLAYER))
            for box, tid, cls, p, cov in zip(people.xyxy, people.tracker_id, classes, feet, covered):
                if box[3] >= fh - 3:
                    stats["feet_cut"] += 1  # feet below the frame edge: the projected spot would be wrong
                    continue
                if not np.isfinite(p).all() or not on_pitch(p, a.margin):
                    continue
                tid = int(tid) + id_offset
                max_tid = max(max_tid, tid)
                role_votes[tid][str(cls)] += 1
                # Colour samples only from unoccluded players (two overlapping shirts = a mixed colour),
                # and spread over the whole track rather than just its first 30 frames.
                # Colour sample from every frame where the player is unoccluded (two overlapping shirts give
                # a mixed colour) and big enough to see; kept with the frame number so a mid-track switch
                # of team (an ID swap) can be found.
                if cov < MAX_COLOR_OVERLAP and box[3] - box[1] >= 20:
                    c = torso_color(frame, box)
                    if c is not None:
                        color_seq[tid].append((i, c))
                        color_by_entry[(i, tid)] = c
                players.append({"id": tid, "x": float(p[0]), "y": float(p[1]), "_raw": tid})

        # 3) Ball candidates. The choice between them is made after the whole clip is processed
        #    (choose_ball_path), so one frame's false positive can't pull the track away.
        def balls_of(d, single_class_ok):
            if not len(d):
                return d
            nm = np.char.lower(d.data.get("class_name", np.full(len(d), BALL)).astype(str))
            # A dedicated ball model may call its class "football"/"sports ball"/...
            keep = np.isin(nm, list(BALL_NAMES))
            if single_class_ok and not keep.any() and len(set(nm)) == 1:
                keep[:] = True
            return d[keep]

        cand_det = [balls_of(det, False)]
        if ball_model:
            _t = time.perf_counter()
            cand_det.append(balls_of(ball_model.detections(frame, conf=a.ball_conf, imgsz=a.imgsz,
                                                           tiles=a.ball_tiles), True))
            timing["ball detector"] += time.perf_counter() - _t
        cands = []
        for d in cand_det:
            if not len(d):
                continue
            ctr = d.get_anchors_coordinates(sv.Position.CENTER)
            foot = d.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pts = to_pitch(H, foot)
            conf = d.confidence if d.confidence is not None else np.ones(len(d))
            for c, q, xy in zip(conf, pts, ctr):
                if not (np.isfinite(q).all() and on_pitch(q, BALL_MARGIN)):
                    continue
                if on_upper_body(xy, person_boxes):
                    stats["ball_on_body"] += 1  # white shirt / number / sock, not the ball
                    continue
                cands.append((xy.astype(float), float(c), q.astype(float)))
        ball_cands.append(cands)
        ball = None
        rec = {"t": t, "view": True, "calib": calib, "n_kp": n_kp, "players": players, "ball": ball}
        if fit is not None:
            rec["line_fit"] = round(float(fit), 3)
        vp = view_polygon(H, frame.shape[1], frame.shape[0], L, W)
        rec["view_polygon"] = [[round(float(x), 2), round(float(y), 2)] for x, y in vp]
        if a.save_h:
            rec["H"] = H.tolist()
        frames_out.append(rec)

    # 3b) At the start of a shot the first calibration can be metres off until the line fit has locked on
    #     (players then all "jump" into place). Blank those leading frames: usually a tenth of a second.
    n_unsettled, k = 0, 0
    while k < len(frames_out):
        if frames_out[k]["view"] and (k == 0 or not frames_out[k - 1]["view"]):  # a shot starts here
            j = k
            while j < len(frames_out) and frames_out[j]["view"] and frames_out[j].get("line_fit") is not None \
                    and frames_out[j]["line_fit"] < UNSETTLED_FIT and j - k < int(fps):
                j += 1
            settled = j < len(frames_out) and frames_out[j]["view"] and \
                (frames_out[j].get("line_fit") or 0) >= UNSETTLED_FIT
            if settled and j > k:
                for jj in range(k, j):
                    frames_out[jj].update({"view": False, "players": [], "ball": None, "unsettled": True})
                    frames_out[jj].pop("view_polygon", None)
                    ball_cands[jj] = []
                n_unsettled += j - k
            k = max(j, k + 1)
        else:
            k += 1
    stats["unsettled"] = n_unsettled

    # 4) Offline clean-up now that the whole clip is known: the ball path, zero-lag smoothing, short ball gaps.
    for k, p in choose_ball_path(ball_cands, motions, [f["view"] for f in frames_out], frame_w).items():
        frames_out[k]["ball"] = [round(float(p[0]), 2), round(float(p[1]), 2)]
    n_raw_ids = len({pl["id"] for f in frames_out for pl in f["players"]})
    n_jump_splits = split_impossible_jumps(frames_out, color_seq, role_votes, fps)
    n_splits = split_team_switches(frames_out, color_seq, role_votes, fps)
    color_samples = defaultdict(list)
    for t, seq in color_seq.items():  # evenly spread subset per track for clustering and stitching
        cs = [c for _, c in sorted(seq, key=lambda x: x[0])]
        if len(cs) > 2 * MAX_COLOR_SAMPLES:
            cs = [cs[k] for k in np.linspace(0, len(cs) - 1, 2 * MAX_COLOR_SAMPLES).astype(int)]
        color_samples[t] = cs
    stitch_tracks(frames_out, color_samples, role_votes, fps, a.stitch_gap, L, W)
    n_ids = len({pl["id"] for f in frames_out for pl in f["players"]})
    n_dups = remove_duplicate_tracks(frames_out, fps)
    n_filled = fill_player_gaps(frames_out, int(round(a.fill_gap * fps)), fps)
    win = a.smooth_window if a.smooth_window is not None else int(round(a.smooth_seconds * fps)) | 1
    smooth_tracks(frames_out, win, jump_window=max(3, int(round(0.25 * fps)) | 1))
    cap_speed(frames_out, fps)
    if a.ball_gap > 0:
        interpolate_ball(frames_out, a.ball_gap)
    for f in frames_out:
        for pl in f["players"]:
            pl["x"], pl["y"] = round(pl["x"], 2), round(pl["y"], 2)

    # 5) Roles and teams.
    roles = {tid: max(v, key=v.get) for tid, v in role_votes.items()}
    pos_sum = defaultdict(lambda: np.zeros(2))
    pos_n = defaultdict(int)
    for f in frames_out:
        for pl in f["players"]:
            pos_sum[pl["id"]] += (pl["x"], pl["y"])
            pos_n[pl["id"]] += 1

    teams, outliers = assign_teams(color_samples, roles)
    team_fit = fit_team_colors(color_samples, roles)  # same model as assign_teams (kit colours, --debug-teams)
    team_info = {}
    if team_fit is not None:
        for k in (0, 1):
            rgb, hx = kit_rgb(team_fit[0].cluster_centers_[k])
            team_info[str(k)] = {"kit_rgb": rgb, "kit_hex": hx}

    # Goalkeeper = the detector said so, or a shirt-colour outlier that spends its time near a goal.
    keepers = set()
    for tid in roles:
        near_goal = pos_n[tid] > 0 and not (25 <= pos_sum[tid][0] / pos_n[tid] <= L - 25)
        if roles[tid] == GOALKEEPER or (tid in outliers and near_goal):
            keepers.add(tid)
            roles[tid] = GOALKEEPER

    # A "keeper" seen for under 2 s is almost always an outfield player the detector mislabelled once.
    for tid in list(keepers):
        if pos_n[tid] < 2 * fps:
            keepers.discard(tid)
            votes_no_gk = {c: n for c, n in role_votes[tid].items() if c != GOALKEEPER}
            roles[tid] = max(votes_no_gk, key=votes_no_gk.get) if votes_no_gk else PLAYER

    # Other colour outliers (not keepers): the referee or an assistant the detector called a 'player'.
    # Seen for 2 s+, they get role "referee" and no team, so they don't count for either side.
    referees = set()
    for tid in outliers:
        if tid not in keepers and pos_n[tid] >= 2 * fps:
            referees.add(tid)
            roles[tid] = "referee"
            teams[tid] = None

    # A keeper's team = whichever team's outfield players are, frame by frame, nearer his end of the pitch.
    # (Voting per frame keeps this right after teams swap ends at half time.)
    for tid in keepers:
        votes = []
        for f in frames_out:
            me = next((pl for pl in f["players"] if pl["id"] == tid), None)
            if me is None:
                continue
            xs = {0: [], 1: []}
            for pl in f["players"]:
                tm = teams.get(pl["id"])
                if pl["id"] not in keepers and tm in xs:
                    xs[tm].append(pl["x"])
            if xs[0] and xs[1]:
                votes.append(min((0, 1), key=lambda tm: abs(float(np.mean(xs[tm])) - me["x"])))
        teams[tid] = max(set(votes), key=votes.count) if votes else None

    # One keeper per team: join that team's keeper fragments that never appear at the same time (he
    # leaves the shot whenever play is at the other end, so stitching alone rarely joins them).
    keeper_frames = defaultdict(set)
    for k, f in enumerate(frames_out):
        for pl in f["players"]:
            if pl["id"] in keepers:
                keeper_frames[pl["id"]].add(k)
    rename = {}
    for tm in (0, 1):
        frags = sorted((t for t in keepers if teams.get(t) == tm), key=lambda t: min(keeper_frames[t]))
        merged = []  # [(keeper id, frames)]
        for t in frags:
            for j, (kid, kf) in enumerate(merged):
                if not (kf & keeper_frames[t]):
                    rename[t] = kid
                    merged[j] = (kid, kf | keeper_frames[t])
                    break
            else:
                merged.append((t, set(keeper_frames[t])))
    if rename:
        for f in frames_out:
            for pl in f["players"]:
                pl["id"] = rename.get(pl["id"], pl["id"])
        keepers -= set(rename)

    for k_f, f in enumerate(frames_out):
        for pl in f["players"]:
            pl["team"] = teams.get(pl["id"])
            pl["role"] = roles.get(pl["id"])
            raw = pl.pop("_raw", None)
            if a.debug_teams:
                # raw_id: the tracker's own ID before splitting/joining; team_vote: this frame's shirt
                # colour on its own (null = no usable sample that frame)
                pl["raw_id"] = raw
                c = color_by_entry.get((k_f, raw)) if raw is not None else None
                pl["team_vote"] = sample_votes(team_fit[0], [c])[0] if (c is not None and team_fit) else None

    # 6) Output coordinates. Everything above works with the origin at a corner (that's how the pitch
    #    landmarks are defined); shift only what is written out.
    if a.origin == "center":
        ox, oy = L / 2, W / 2
        T = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]])  # corner metres -> centre metres
        for f in frames_out:
            for pl in f["players"]:
                pl["x"], pl["y"] = round(pl["x"] - ox, 2), round(pl["y"] - oy, 2)
            if f["ball"]:
                f["ball"] = [round(f["ball"][0] - ox, 2), round(f["ball"][1] - oy, 2)]
            if "H" in f:
                f["H"] = (T @ np.array(f["H"])).tolist()  # still maps image pixels -> output coordinates
            if f.get("view_polygon"):
                f["view_polygon"] = [[round(x - ox, 2), round(y - oy, 2)] for x, y in f["view_polygon"]]

    pitch_info = {"length": L, "width": W, "origin": a.origin,
                  "x_range": [-L / 2, L / 2] if a.origin == "center" else [0, L],
                  "y_range": [-W / 2, W / 2] if a.origin == "center" else [0, W]}
    with open(a.out, "w") as fh:
        json.dump({"pitch": pitch_info, "fps": fps, "video": {"width": info.width, "height": info.height},
                   "teams": team_info, "stats": dict(stats), "frames": frames_out}, fh)
    n, usable = len(frames_out), sum(f["view"] for f in frames_out)
    print(f"wrote {n} frames -> {a.out}")
    if team_info:
        print(f"  teams: 0 = kit {team_info['0']['kit_hex']}, 1 = kit {team_info['1']['kit_hex']}  "
              f"(team 0 is always the lighter kit, or the warmer one when equally light)")
    print(f"  usable pitch view: {usable}/{n}  (fresh calibration {stats['kp']}, coasting {stats['coast']}, "
          f"no view {stats['noview']}, landmark calibrations rejected {stats['gated']}, snapped to lines {stats['line_refined']}, "
          f"keypoint model skipped on {stats['kp_not_run']} frames, {stats['unsettled']} frames dropped while the "
          f"calibration settled)")
    print(f"  ball in {sum(1 for f in frames_out if f['ball'])}/{usable} usable frames "
          f"({sum(1 for f in frames_out if f.get('ball_interp'))} filled by interpolation); "
          f"{stats['feet_cut']} detections dropped with feet out of frame")
    total = time.time() - t0
    if timing and n:
        parts = sorted(timing.items(), key=lambda kv: -kv[1])
        other = max(total - sum(timing.values()), 0.0)
        print(f"  time per processed frame: {1000 * total / n:.0f} ms = " + ", ".join(
            f"{name} {1000 * sec / n:.0f}" for name, sec in parts) + f", other {1000 * other / n:.0f}")
    print(f"  tracks: {n_raw_ids} raw ({n_jump_splits} split at an impossible jump, {n_splits} at a team switch, "
          f"{n_dups} duplicates removed) -> {n_ids} after stitching, {n_filled} player positions filled in; "
          f"referees found: {len(referees)}, goalkeepers found: {len(keepers)}{f' ({len(rename)} keeper fragments joined)' if rename else ''}")


if __name__ == "__main__":
    main()