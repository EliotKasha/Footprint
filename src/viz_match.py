"""
Visual check for track_match.py output: draws match.json as a top-down pitch video.

  python src/viz_match.py match.json --out viz.mp4
  python src/viz_match.py match.json --video test1.mp4 --out viz.mp4   # side by side with the source
  python src/viz_match.py match.json --video test1.mp4 --show          # live window, q = quit, space = pause

What to look for:
  * With --video and a match.json made with --save-h, the pitch lines are drawn back onto the source
    video (cyan = fresh calibration, magenta = coasting). They should sit on the real lines. This is
    the most direct check of the calibration: if the lines are off, every position is off.
    "fit NN%" in the header = share of the visible markings that land on a detected white line.
  * Players should glide smoothly. A red ring means that player moved faster than --max-speed since
    he was last seen; a red banner means the MEDIAN player did (bad calibration, not a bad track).
  * The ball (white) should follow the real ball. Its trail makes flicker between two candidates obvious.
    A hollow ball / hollow player = position filled in by interpolation across a short dropout.
  * With match_events.json (from match_events.py): a white ring marks the player in possession, a
    lofted ball is drawn larger with its height and a shadow, and passes flash as arrows
    (white = completed, red = turnover, yellow = shot / clearance / ball in the air with no receiver).
  * With tracking run with --debug-teams: labels read "ID/raw tracker ID", and a small triangle in the
    other team's colour marks frames where the player's own shirt colour disagrees with his team.
  * Player numbers should stay the same for the whole clip (track stitching); a number that keeps
    changing on one player means the stitching missed a link.
  * "NO PITCH VIEW" frames are close-ups / cuts where the homography failed.
  * The lighter area on the minimap is what the camera shows in that frame. Players leaving it walked
    out of shot; players vanishing in the middle of it were missed by the detector.
"""
import argparse
import json
import time
from collections import deque

import cv2
import numpy as np

SCALE = 10   # pixels per metre
PAD = 30
TEAM_COLORS = {0: (60, 60, 230), 1: (230, 130, 40), None: (160, 160, 160)}  # BGR
GREEN, WHITE, YELLOW, RED = (60, 110, 50), (255, 255, 255), (0, 220, 255), (0, 0, 255)
CYAN, MAGENTA = (255, 255, 0), (255, 0, 255)
NOISE_FLOOR = 0.3  # metres of per-frame calibration jitter that is never flagged as a jump

PB_W, PB_L, GB_W, GB_L, CC_R, SPOT = 40.32, 16.5, 18.32, 5.5, 9.15, 11.0  # as pitch_vertices() in track_match.py


def draw_pitch(L, W):
    """Pitch background using the same regulation dimensions as pitch_vertices() in track_match.py."""
    def P(x, y):
        return int(round(PAD + x * SCALE)), int(round(PAD + y * SCALE))

    img = np.full((int(W * SCALE) + 2 * PAD, int(L * SCALE) + 2 * PAD, 3), GREEN, np.uint8)
    for poly in pitch_polylines(L, W):
        pts = np.array([P(x, y) for x, y in poly], np.int32)
        cv2.polylines(img, [pts], False, WHITE, 2, cv2.LINE_AA)
    for sx in (SPOT, L - SPOT):
        cv2.circle(img, P(sx, W / 2), 3, WHITE, -1)
    return img, P


def pitch_polylines(L, W):
    """All pitch markings as polylines in metres (densely sampled so they bend correctly under perspective)."""
    def seg(a, b, n=40):
        return [(a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s) for s in np.linspace(0, 1, n)]

    def rect(x0, y0, x1, y1):
        return seg((x0, y0), (x1, y0)) + seg((x1, y0), (x1, y1)) + seg((x1, y1), (x0, y1)) + seg((x0, y1), (x0, y0))

    lines = [rect(0, 0, L, W), seg((L / 2, 0), (L / 2, W))]
    th = np.linspace(0, 2 * np.pi, 80)
    lines.append([(L / 2 + CC_R * np.cos(t), W / 2 + CC_R * np.sin(t)) for t in th])
    for x_out, x_in_pb, x_in_gb in ((0, PB_L, GB_L), (L, L - PB_L, L - GB_L)):
        lines.append(rect(x_out, (W - PB_W) / 2, x_in_pb, (W + PB_W) / 2))
        lines.append(rect(x_out, (W - GB_W) / 2, x_in_gb, (W + GB_W) / 2))
    t = np.arccos((PB_L - SPOT) / CC_R)  # penalty arcs: the part of the spot's circle outside the box
    lines.append([(SPOT + CC_R * np.cos(a), W / 2 + CC_R * np.sin(a)) for a in np.linspace(-t, t, 30)])
    lines.append([(L - SPOT + CC_R * np.cos(a), W / 2 + CC_R * np.sin(a)) for a in np.linspace(np.pi - t, np.pi + t, 30)])
    return lines


def overlay_pitch(frame, H, L, W, color):
    """Draw the pitch markings onto the source frame with the inverse of the stored image->pitch H."""
    try:
        Hi = np.linalg.inv(np.asarray(H, dtype=np.float64))
    except np.linalg.LinAlgError:
        return
    h, w = frame.shape[:2]
    for poly in pitch_polylines(L, W):
        P = np.c_[np.asarray(poly), np.ones(len(poly))] @ Hi.T
        front = P[:, 2] > 1e-9
        xy = P[:, :2] / np.where(front, P[:, 2], 1.0)[:, None]
        ok = front & (np.abs(xy) < 4 * max(h, w)).all(axis=1)  # drop points beyond the horizon
        # draw only runs of consecutive valid points, so a line never jumps across the horizon
        run = []
        for p, good in zip(xy, ok):
            if good:
                run.append(p)
            elif run:
                if len(run) > 1:
                    cv2.polylines(frame, [np.round(run).astype(np.int32)], False, color, 2, cv2.LINE_AA)
                run = []
        if len(run) > 1:
            cv2.polylines(frame, [np.round(run).astype(np.int32)], False, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--video", help="source video, shown next to the minimap")
    ap.add_argument("--out", default="viz.mp4")
    ap.add_argument("--show", action="store_true", help="show a live window instead of / as well as writing")
    ap.add_argument("--max-speed", type=float, default=12.0,
                    help="m/s above which a player's move is flagged (top sprint speed is ~10 m/s). "
                         "Replaces --max-jump, which ignored frame rate: 2 m/frame at 25 fps is 50 m/s.")
    a = ap.parse_args()

    with open(a.json) as fh:
        data = json.load(fh)
    L, W, fps = data["pitch"]["length"], data["pitch"]["width"], data["fps"]
    # Drawing works in corner coordinates; files written with --origin center are shifted back here.
    centered = data["pitch"].get("origin", "corner") == "center"
    ox, oy = (L / 2, W / 2) if centered else (0.0, 0.0)
    T_back = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]])
    for f in data["frames"]:
        for pl in f["players"]:
            pl["x"], pl["y"] = pl["x"] + ox, pl["y"] + oy
        if f.get("ball"):
            f["ball"] = [f["ball"][0] + ox, f["ball"][1] + oy] + list(f["ball"][2:])  # keep height if present
        if "H" in f:
            f["H"] = (T_back @ np.asarray(f["H"])).tolist()
        if f.get("view_polygon"):
            f["view_polygon"] = [[x + ox, y + oy] for x, y in f["view_polygon"]]
    dt = 1.0 / fps
    base, P = draw_pitch(L, W)
    h, w = base.shape[:2]

    cap, step, vw = None, 1, 0
    if a.video:
        cap = cv2.VideoCapture(a.video)
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) <= 0:
            raise SystemExit(f"could not open video: {a.video}")
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(1, round(src_fps / fps))  # = the --stride used in track_match.py
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * h / cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    has_h = any("H" in f for f in data["frames"])
    if a.video and not has_h:
        print("note: run track_match.py with --save-h to see the calibration drawn on the video")

    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w + vw, h))
    last_pos = {}                 # pid -> (frame index, position): compare against where he was last seen
    trail = deque(maxlen=15)      # (frame index, ball position)
    passes = [e for e in data.get("events", []) if e["type"] in ("pass", "aerial_ball")]
    for e in passes:  # same coordinate shift as the frames
        e["start"] = [e["start"][0] + ox, e["start"][1] + oy]
        e["end"] = [e["end"][0] + ox, e["end"][1] + oy]
    n_nov = n_bad = n_ball = n_interp = n_kp = n_coast = 0

    t_start, n_frames = time.time(), len(data["frames"])
    for k, f in enumerate(data["frames"]):
        if k % 5 == 0:  # progress, in the same format as track_match.py (run_pipeline.sh draws a bar from it)
            print(f"frame {k}/{n_frames}  ({k / max(time.time() - t_start, 1e-6):.1f} processed fps)", flush=True)
        img = base.copy()
        vframe = None
        if cap:
            ok, vf = cap.read()
            for _ in range(step - 1):
                cap.grab()
            if ok:
                if f.get("view") and "H" in f:
                    overlay_pitch(vf, f["H"], L, W, CYAN if f.get("calib") == "kp" else MAGENTA)
                vframe = cv2.resize(vf, (vw, h))
            else:
                vframe = np.zeros((h, vw, 3), np.uint8)

        jumps = []
        if f.get("view_polygon"):  # what the camera shows: light fill + outline
            vp = np.array([P(x, y) for x, y in f["view_polygon"]], np.int32)
            shade = img.copy()
            cv2.fillPoly(shade, [vp], (95, 150, 85))
            img = cv2.addWeighted(shade, 0.45, img, 0.55, 0)
            cv2.polylines(img, [vp], True, (170, 220, 160), 1, cv2.LINE_AA)
        if not f["view"]:
            n_nov += 1
            last_pos.clear()   # track IDs restart after a cut
            trail.clear()      # and don't draw a trail line across it
            cv2.putText(img, "NO PITCH VIEW", (w // 2 - 160, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, YELLOW, 3)
        else:
            n_kp += f.get("calib") == "kp"
            n_coast += f.get("calib") == "coast"
            for p in f["players"]:
                pid, pos = p["id"], (p["x"], p["y"])
                flagged = False
                if pid in last_pos:
                    k0, pos0 = last_pos[pid]
                    gap = k - k0
                    d = float(np.hypot(pos[0] - pos0[0], pos[1] - pos0[1]))
                    if gap == 1:
                        jumps.append(d)
                    flagged = d > a.max_speed * dt * gap + NOISE_FLOOR
                last_pos[pid] = (k, pos)
                c = TEAM_COLORS.get(p.get("team"), TEAM_COLORS[None])
                cv2.circle(img, P(*pos), 8, c, 3 if p.get("interp") else -1)  # hollow = filled-in gap
                cv2.circle(img, P(*pos), 8, YELLOW if p.get("role") == "goalkeeper" else (0, 0, 0), 2)
                if flagged:
                    cv2.circle(img, P(*pos), 14, RED, 2)
                tv = p.get("team_vote")  # --debug-teams: this frame's shirt colour disagrees with the team
                if tv is not None and p.get("team") is not None and tv != p["team"] and p.get("role") == "player":
                    cv2.drawMarker(img, (P(*pos)[0] - 13, P(*pos)[1] - 13), TEAM_COLORS[tv], cv2.MARKER_TRIANGLE_UP, 9, 2)
                if f.get("owner") == pid:  # match_events.json: player in possession
                    cv2.circle(img, P(*pos), 12, WHITE, 2)
                label = str(pid) if p.get("raw_id") is None else f"{pid}/{p['raw_id']}"
                cv2.putText(img, label, (P(*pos)[0] + 9, P(*pos)[1] - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
            if f["ball"]:
                n_ball += 1
                n_interp += bool(f.get("ball_interp"))
                trail.append((k, tuple(f["ball"][:2])))
            for (ka, pa), (kb, pb) in zip(list(trail)[:-1], list(trail)[1:]):
                if kb - ka <= 3:  # don't connect ball positions across a long gap
                    cv2.line(img, P(*pa), P(*pb), WHITE, 1, cv2.LINE_AA)
            if f["ball"]:
                bxy = f["ball"][:2]
                z = f["ball"][2] if len(f["ball"]) > 2 else 0.0
                if z > 0.05:  # in the air: shadow on the grass, ball drawn bigger, height label
                    cv2.circle(img, P(*bxy), 4, (30, 60, 25), -1)
                    r = int(6 + min(z, 15) * 0.6)
                    cv2.circle(img, P(*bxy), r, WHITE, 2)
                    cv2.putText(img, f"{z:.1f} m", (P(*bxy)[0] + r + 3, P(*bxy)[1] + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
                elif f.get("ball_interp"):
                    cv2.circle(img, P(*bxy), 6, WHITE, 2)
                else:
                    cv2.circle(img, P(*bxy), 6, WHITE, -1)
                    cv2.circle(img, P(*bxy), 6, (0, 0, 0), 2)

        for e in passes:  # show each pass from kick until 1 s after it arrives
            if e["t_start"] <= f["t"] <= e["t_end"] + 1.0:
                col = YELLOW if e["type"] == "aerial_ball" else WHITE if e["completed"] else RED
                cv2.arrowedLine(img, P(*e["start"]), P(*e["end"]), col, 2, cv2.LINE_AA, tipLength=0.06)

        med = float(np.median(jumps)) if jumps else 0.0
        if med > a.max_speed * dt + NOISE_FLOOR:
            n_bad += 1
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), RED, 6)
            cv2.putText(img, "CALIBRATION JUMP", (PAD, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
        calib = f.get("calib") or "-"
        if "line_fit" in f:
            calib += f" fit {100 * f['line_fit']:.0f}%"
        cv2.putText(img, f"t={f['t']:.2f}s  frame {k}  calib {calib} ({f.get('n_kp', 0)} kp)  "
                         f"players {len(f['players'])}  median move {med / dt:.1f} m/s",
                    (PAD, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

        out = np.hstack([img, vframe]) if vframe is not None else img
        writer.write(out)
        if a.show:
            cv2.imshow("match", out)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)

    writer.release()
    if cap:
        cap.release()
    if a.show:
        cv2.destroyAllWindows()
    n = len(data["frames"])
    usable = n - n_nov
    print(f"{n} frames: {n_nov} without pitch view, {n_kp} fresh calibration, {n_coast} coasting, "
          f"{n_bad} with a calibration jump")
    print(f"ball present in {n_ball}/{usable} usable frames ({n_interp} interpolated) -> {a.out}")


if __name__ == "__main__":
    main()