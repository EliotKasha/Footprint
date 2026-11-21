"""
Match events from track_match.py output: possession, touches, passes, ball height, and stats.

  python src/match_events.py match.json --out match_events.json

Reads match.json, writes a copy with:
  * per frame  "owner": player id in possession, or null
               "ball": [x, y, z]  (z = height in metres, 0 on the ground)
               "ball_air": true while the ball is in a lofted flight
  * top level  "events": touches and passes, in time order
               "stats":  per team (possession %, passes, accuracy, shape) and per player
                         (touches, passes, distance covered, top speed, average position)
Everything else in match.json is kept unchanged, so Unity can load this file instead.

How it works
  Possession: the nearest player within --poss-radius metres of the ball, with the ball moving roughly
    with him (relative speed below --rel-speed), for a few frames in a row. An airborne ball's projected
    position races across the pitch, so it isn't "owned" by players it flies over.
  Touch: each new spell of possession.
  Pass: possession goes from player A to player B within --pass-gap seconds, in the same shot.
    Completed if B is A's teammate, otherwise a turnover.
  Ball height, with --save-h calibration (preferred): the camera (position, angle, zoom) is recovered
    from each frame's calibration, every ball detection becomes a line of sight, and a gravity curve
    is fitted through them after every possession. It counts as a flight only if it explains what the
    camera saw clearly better than a ball rolling on the grass. Works for shots, clearances and
    headers too (no receiver needed); those become "aerial_ball" events.
  Ball height without calibration (fallback): between two possessions the ball either rolls (its ground track is close to a straight
    line at steady speed) or flies. A flying ball is projected onto the grass far beyond where it really
    is, so its track bulges away from the straight line. For a flight, the takeoff and landing points are
    on the grass and therefore correct; the ground track in between is a straight line, and the height
    follows from gravity and flight time: z(t) = g/2 * (t - t0) * (t_land - t), peak g*T^2/8.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

G = 9.81


def segments(frames):
    """Index ranges of consecutive frames with a pitch view (one camera shot)."""
    segs, cur = [], []
    for k, f in enumerate(frames):
        if f["view"]:
            cur.append(k)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return segs


def ball_xy(f):
    b = f.get("ball")
    return None if b is None else np.array(b[:2], dtype=float)


def velocities(frames, fps):
    """Central-difference velocities (m/s) for the ball and every player, per frame (None if unknown)."""
    n = len(frames)
    ball_v = [None] * n
    # referees aren't candidates for possession (the ball bouncing off one isn't a touch)
    pl_pos = [{p["id"]: np.array([p["x"], p["y"]]) for p in f["players"] if p.get("role") != "referee"}
              for f in frames]
    pl_v = [dict() for _ in range(n)]
    for k in range(n):
        if not frames[k]["view"]:
            continue
        a, b = max(k - 1, 0), min(k + 1, n - 1)
        if a == b or not (frames[a]["view"] and frames[b]["view"]):
            continue
        ba, bb = ball_xy(frames[a]), ball_xy(frames[b])
        if ba is not None and bb is not None:
            ball_v[k] = (bb - ba) * fps / (b - a)
        for pid in pl_pos[k]:
            if pid in pl_pos[a] and pid in pl_pos[b]:
                pl_v[k][pid] = (pl_pos[b][pid] - pl_pos[a][pid]) * fps / (b - a)
    return ball_v, pl_pos, pl_v


def find_possession(frames, fps, a):
    """Per-frame owner (player id or None) and the list of possession spells."""
    ball_v, pl_pos, pl_v = velocities(frames, fps)
    team = {p["id"]: p.get("team") for f in frames for p in f["players"]}
    owner = [None] * len(frames)
    spells = []  # dicts: id, team, start, end (frame indices, inclusive)
    min_frames = max(2, int(round(a.min_poss * fps)))
    hold = max(1, int(round(0.2 * fps)))  # keep the owner through a short ball dropout

    for seg in segments(frames):
        cur, cand, cand_start, cand_n, missing = None, None, None, 0, 0
        for k in seg:
            b = ball_xy(frames[k])
            if b is None:
                missing += 1
                if cur is not None and missing <= hold:
                    owner[k] = cur
                else:
                    cur, cand, cand_n = None, None, 0
                continue
            missing = 0
            near = []
            for pid, p in pl_pos[k].items():
                d = float(np.linalg.norm(p - b))
                if d > a.keep_radius:
                    continue
                bv, pv = ball_v[k], pl_v[k].get(pid)
                rel = float(np.linalg.norm(bv - (pv if pv is not None else 0))) if bv is not None else 0.0
                near.append((d, rel, pid))
            near.sort()
            eligible = [(d, pid) for d, rel, pid in near if d <= a.poss_radius and rel <= a.rel_speed]
            d_cur = next((d for d, rel, pid in near if pid == cur), None)
            if cur is not None and d_cur is not None and not (eligible and eligible[0][0] < d_cur - 0.7
                                                              and eligible[0][1] != cur):
                owner[k] = cur  # current owner keeps the ball (hysteresis: a looser radius to keep it)
                cand, cand_n = None, 0
                continue
            cur = None
            if eligible:
                best = eligible[0][1]
                if best == cand:
                    cand_n += 1
                else:
                    cand, cand_start, cand_n = best, k, 1
                if cand_n >= min_frames:
                    cur = cand
                    for j in range(cand_start, k + 1):  # the spell began when he first had the ball
                        if frames[j]["view"]:
                            owner[j] = cur
                    cand, cand_n = None, 0
            else:
                cand, cand_n = None, 0

    # spells from the per-frame owners
    prev = None
    for k, o in enumerate(owner):
        if o is not None and (prev is None or prev["id"] != o or prev["end"] != k - 1):
            prev = {"id": o, "team": team.get(o), "start": k, "end": k}
            spells.append(prev)
        elif o is not None:
            prev["end"] = k

    # The owner keeps the ball until it is --keep-radius away, which dates every kick a few frames
    # late (shorter flights, lower heights, pass start points metres downfield). Move each spell's end
    # back to the last frame the ball was still at its usual distance from his feet.
    for s in spells:
        d = []
        for k in range(s["start"], s["end"] + 1):
            b, p = ball_xy(frames[k]), pl_pos[k].get(s["id"])
            d.append(None if b is None or p is None else float(np.linalg.norm(b - p)))
        known = [x for x in d if x is not None]
        if len(known) < 3:
            continue
        typical = float(np.median(known))
        close = [s["start"] + j for j, x in enumerate(d) if x is not None and x <= typical + 0.75]
        if close and close[-1] < s["end"]:
            for k in range(close[-1] + 1, s["end"] + 1):
                owner[k] = None
            s["end"] = close[-1]
    return owner, spells


def same_segment(frames, k0, k1):
    return all(frames[j]["view"] for j in range(k0, k1 + 1))


def find_flights(frames, fps, spells, a, skip=()):
    """Classify each transit between two possession spells as rolling or airborne; for airborne ones,
    rewrite the ball as [x, y, z]. Returns {spell index: flight info} keyed by the spell the ball left."""
    flights = {}
    for si in range(len(spells) - 1):
        if si in skip:
            continue
        s1, s2 = spells[si], spells[si + 1]
        k0, k1 = s1["end"], s2["start"]
        if k1 - k0 < 3 or (k1 - k0) / fps > a.pass_gap or not same_segment(frames, k0, k1):
            continue
        p0, p1 = ball_xy(frames[k0]), ball_xy(frames[k1])
        if p0 is None or p1 is None:
            continue
        dist = float(np.linalg.norm(p1 - p0))
        devs = []
        for k in range(k0 + 1, k1):
            b = ball_xy(frames[k])
            if b is None or frames[k].get("ball_interp"):
                continue
            u = p0 + (p1 - p0) * (k - k0) / (k1 - k0)  # where a steadily rolling ball would be
            devs.append((k, float(np.linalg.norm(b - u))))
        if len(devs) < 2:
            continue
        max_dev = max(d for _, d in devs)
        threshold = a.air_dev + 0.1 * dist
        if max_dev <= threshold or sum(d > threshold for _, d in devs) < 2:
            continue
        # Landing: after the last frame that is still clearly off the straight line. From there the
        # ball is on the grass again (rolling on towards the receiver along the same line).
        tol = max(2.0, 0.25 * max_dev)
        k_land = min(max(k for k, d in devs if d > tol) + 1, k1)
        T = (k_land - k0) / fps
        T = min(T, 4.9)  # a 30 m high ball is already absurd for football
        flight_dist = float(np.linalg.norm(p1 - p0)) * T / max((k1 - k0) / fps, 1e-6)
        # A lofted ball travels fast (>= 7 m/s across the ground) and isn't launched steeper than ~55
        # degrees (peak <= 0.36 x distance). A slow "flight" is a rolling ball with detection gaps.
        if flight_dist / max(T, 1e-6) < 7.0 or G * T * T / 8 > 0.36 * max(flight_dist, 1e-6):
            continue
        t0 = k0 / fps
        peak = 0.0
        for k in range(k0, k1 + 1):
            xy = p0 + (p1 - p0) * (k - k0) / (k1 - k0)
            t = k / fps - t0
            z = 0.5 * G * t * (T - t) if 0 <= t <= T else 0.0
            peak = max(peak, z)
            frames[k]["ball"] = [round(float(xy[0]), 2), round(float(xy[1]), 2), round(float(max(z, 0.0)), 2)]
            if z > 0:
                frames[k]["ball_air"] = True
            frames[k].pop("ball_interp", None)
        flights[si] = {"t_takeoff": round(t0, 3), "t_land": round(t0 + T, 3), "max_height": round(peak, 2),
                       "method": "flight_time"}
    return flights


# ----------------------------------------------------------------------------------------------------
# 3D ball flights (needs match.json made with --save-h)
# ----------------------------------------------------------------------------------------------------

def camera_from_H(H, w, h, f=None):
    """Camera (K, R, t, centre C, up sign) from an image->pitch homography, assuming square pixels and
    the principal point at the image centre (true for broadcast cameras to within a few pixels).
    Pitch->image is K [r1 r2 t]; the focal length follows from r1 and r2 being orthonormal."""
    try:
        Hpi = np.linalg.inv(np.asarray(H, dtype=float))
    except np.linalg.LinAlgError:
        return None
    cx, cy = w / 2.0, h / 2.0
    A = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]]) @ Hpi  # = diag(f, f, 1) [r1 r2 t] * scale
    h1, h2 = A[:, 0], A[:, 1]
    if f is None:
        ests = []
        d1 = h1[2] * h2[2]
        if abs(d1) > 1e-12:
            f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / d1
            if f2 > 0:
                ests.append(np.sqrt(f2))
        d2 = h2[2] ** 2 - h1[2] ** 2
        if abs(d2) > 1e-12:
            f2 = (h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / d2
            if f2 > 0:
                ests.append(np.sqrt(f2))
        ests = [e for e in ests if 0.3 * w < e < 30 * w]
        if not ests:
            return None
        f = float(np.exp(np.mean(np.log(ests))))
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Ki = np.linalg.inv(K)
    b1, b2, b3 = Ki @ Hpi[:, 0], Ki @ Hpi[:, 1], Ki @ Hpi[:, 2]
    lam = 2.0 / (np.linalg.norm(b1) + np.linalg.norm(b2))
    r1, r2, t = b1 * lam, b2 * lam, b3 * lam
    # the pitch point seen at the image centre must be in front of the camera
    pc = np.asarray(H, dtype=float) @ np.array([cx, cy, 1.0])
    if abs(pc[2]) < 1e-12:
        return None
    pc = pc[:2] / pc[2]
    if (r1 * pc[0] + r2 * pc[1] + t)[2] < 0:
        r1, r2, t = -r1, -r2, -t
    R = np.c_[r1, r2, np.cross(r1, r2)]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    C = -R.T @ t
    if abs(C[2]) < 1.0:  # camera at grass level: geometry can't be trusted
        return None
    return {"K": K, "R": R, "t": t, "C": C, "up": float(np.sign(C[2])), "f": f}


def frame_cameras(frames, w, h):
    """Camera per frame. The focal length estimate is noisy frame to frame, so it is median-filtered
    over +-0.5 s within each shot and the camera recomputed with the smoothed focal length."""
    cams = [None] * len(frames)
    raw = [camera_from_H(f["H"], w, h) if f["view"] and "H" in f else None for f in frames]
    for seg in segments(frames):
        fs = {k: raw[k]["f"] for k in seg if raw[k] is not None}
        for k in seg:
            if "H" not in frames[k]:
                continue
            near = [fs[j] for j in range(k - 7, k + 8) if j in fs]
            if near:
                cams[k] = camera_from_H(frames[k]["H"], w, h, f=float(np.median(near)))
    return cams


def project_cam(cam, X):
    x = cam["K"] @ (cam["R"] @ X + cam["t"])
    return x[:2] / x[2]


def ball_pixel(f):
    """The ball's pixel in the image, recovered from its projected pitch position and the frame's H."""
    b = ball_xy(f)
    if b is None or "H" not in f:
        return None
    p = np.linalg.inv(np.asarray(f["H"], dtype=float)) @ np.array([b[0], b[1], 1.0])
    return p[:2] / p[2] if abs(p[2]) > 1e-12 else None


def _fit_flight(obs, up, takeoff_on_ground):
    """Least-squares gravity trajectory through the ball's pixels. obs: [(dt, pixel, cam)].
    Returns (p0, v, rms_px). Unknowns: start point and velocity (6); gravity is fixed."""
    from scipy.optimize import least_squares
    acc = np.array([0.0, 0.0, -up * G])
    dts = np.array([o[0] for o in obs])
    T = max(dts[-1], 1e-3)

    def ground_hit(cam, px):
        d = cam["R"].T @ (np.linalg.inv(cam["K"]) @ np.array([px[0], px[1], 1.0]))
        lam = -cam["C"][2] / d[2] if abs(d[2]) > 1e-12 else np.nan
        return cam["C"] + lam * d

    g0, g1 = ground_hit(obs[0][2], obs[0][1]), ground_hit(obs[-1][2], obs[-1][1])
    if not (np.isfinite(g0).all() and np.isfinite(g1).all()):
        return None
    x0 = np.r_[g0[:2], 0.0, (g1[:2] - g0[:2]) / T, up * G * T / 2]

    def resid(x):
        p0, v = x[:3], x[3:]
        r = []
        for dt, px, cam in obs:
            X = p0 + v * dt + 0.5 * acc * dt * dt
            z_cam = (cam["R"] @ X + cam["t"])[2]
            r.extend(project_cam(cam, X) - px if z_cam > 0 else [1e3, 1e3])
        if takeoff_on_ground:
            r.append(20.0 * (up * p0[2]))  # kicked off the grass: start height ~0 (20 px per metre)
        return np.asarray(r)

    # Keep the search physical: ball within 30 m of the pitch area it was seen in, at most 25 m up,
    # at most 45 m/s. Junk detections then fail fast instead of fitting absurd 1000 m/s "flights".
    zlo, zhi = (-1.0, 25.0) if up > 0 else (-25.0, 1.0)
    lo = np.r_[x0[:2] - 30, zlo, -45, -45, -45]
    hi = np.r_[x0[:2] + 30, zhi, 45, 45, 45]
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    try:
        sol = least_squares(resid, x0, loss="soft_l1", f_scale=3.0, max_nfev=200, bounds=(lo, hi))
    except ValueError:
        return None
    rr = resid(sol.x)[: 2 * len(obs)].reshape(-1, 2)
    return sol.x[:3], sol.x[3:], float(np.sqrt((rr ** 2).sum(axis=1).mean()))


def _fit_ground(obs):
    """Rolling ball on the grass: straight line with constant deceleration (5 unknowns). rms in px."""
    from scipy.optimize import least_squares
    dts = np.array([o[0] for o in obs])
    T = max(dts[-1], 1e-3)

    def hit(cam, px):
        d = cam["R"].T @ (np.linalg.inv(cam["K"]) @ np.array([px[0], px[1], 1.0]))
        return cam["C"] + (-cam["C"][2] / d[2]) * d

    g0, g1 = hit(obs[0][2], obs[0][1]), hit(obs[-1][2], obs[-1][1])
    x0 = np.r_[g0[:2], (g1[:2] - g0[:2]) / T, 0.0]

    def resid(x):
        p0, u, dec = x[:2], x[2:4], x[4]
        sp = np.linalg.norm(u) + 1e-9
        r = []
        for dt, px, cam in obs:
            P = p0 + u * dt - 0.5 * dec * dt * dt * u / sp
            r.extend(project_cam(cam, np.r_[P, 0.0]) - px)
        return np.asarray(r)

    try:
        sol = least_squares(resid, x0, loss="soft_l1", f_scale=3.0, max_nfev=200)
    except ValueError:
        return np.inf
    rr = resid(sol.x).reshape(-1, 2)
    return float(np.sqrt((rr ** 2).sum(axis=1).mean()))


def find_flights_3d(frames, fps, spells, cams, w, a):
    """After every possession spell, look at the ball until the next possession (or the ball is lost,
    the shot ends, or 5 s pass). Fit both a rolling and a flying model to what the camera saw; call it
    a flight only if flying fits clearly better and goes at least 0.6 m up. If the ball lands and rolls
    on inside the window, the flight is fitted on the airborne part only.
    Returns ({spell index: info} for flights that end at the next possession,
             [info] for flights with no receiver, e.g. shots and clearances)."""
    thr = max(3.0, 0.0025 * w)  # px: a good fit, given detection noise
    linked, loose = {}, []
    for si, s in enumerate(spells):
        k0 = s["end"]
        if not frames[k0]["view"] or cams[k0] is None:
            continue
        nxt = spells[si + 1] if si + 1 < len(spells) else None
        k_end, missing = k0, 0
        while k_end + 1 < len(frames) and frames[k_end + 1]["view"] and (k_end + 1 - k0) / fps <= 5.0:
            k_end += 1
            if nxt is not None and k_end >= nxt["start"]:
                break
            missing = missing + 1 if ball_xy(frames[k_end]) is None else 0
            if missing > int(0.6 * fps):
                k_end -= missing
                break
        obs = []
        for k in range(k0, k_end + 1):
            if frames[k].get("ball_interp") or cams[k] is None:
                continue
            px = ball_pixel(frames[k])
            if px is not None:
                obs.append(((k - k0) / fps, px, cams[k], k))
        if len(obs) < 6:
            continue
        up = cams[k0]["up"]
        # longest airborne prefix that a gravity curve explains
        best = None
        n = len(obs)
        step = max(1, n // 12)
        for m in range(n, 5, -step):
            part = [(dt, px, cam) for dt, px, cam, _ in obs[:m]]
            fit = _fit_flight(part, up, takeoff_on_ground=True)
            if fit is None:
                continue
            p0, v, rms = fit
            if rms <= thr:
                best = (m, p0, v, rms)
                break
        if best is None:
            continue
        m, p0, v, rms = best
        part = [(dt, px, cam) for dt, px, cam, _ in obs[:m]]
        rms_g = _fit_ground(part)
        vz = up * v[2]
        t_land = 2 * vz / G + 0.0 if vz > 0 else 0.0   # height returns to 0 (takeoff at ~0)
        peak = vz * vz / (2 * G) if vz > 0 else 0.0
        if peak < 0.6 or rms_g <= 1.5 * rms + 1.0:
            continue  # rolling explains it just as well
        last_obs_t = obs[m - 1][0]
        t_land = min(max(t_land, last_obs_t), (k_end - k0) / fps)
        k_land = k0 + int(round(t_land * fps))
        acc = np.array([0.0, 0.0, -up * G])
        for k in range(k0, k_land + 1):
            dt = (k - k0) / fps
            X = p0 + v * dt + 0.5 * acc * dt * dt
            z = max(0.0, float(up * X[2]))
            frames[k]["ball"] = [round(float(X[0]), 2), round(float(X[1]), 2), round(z, 2)]
            if z > 0.05:
                frames[k]["ball_air"] = True
            else:
                frames[k].pop("ball_air", None)
            frames[k].pop("ball_interp", None)
        X_land = p0 + v * t_land + 0.5 * acc * t_land ** 2
        info = {"t_takeoff": frames[k0]["t"], "t_land": round(frames[k0]["t"] + t_land, 3),
                "max_height": round(peak, 2), "from": s["id"], "team": s["team"],
                "start": [round(float(p0[0]), 2), round(float(p0[1]), 2)],
                "land": [round(float(X_land[0]), 2), round(float(X_land[1]), 2)],
                "speed_ms": round(float(np.linalg.norm(v)), 1), "fit_px": round(rms, 2), "method": "3d"}
        if nxt is not None and k_end >= nxt["start"] and nxt["id"] != s["id"] \
                and (nxt["start"] - k0) / fps <= a.pass_gap:
            linked[si] = info
        else:
            loose.append(info)
    return linked, loose


def build_events(frames, fps, spells, flights, a, loose=()):
    events = []
    for si, s in enumerate(spells):
        f = frames[s["start"]]
        b = ball_xy(f)
        events.append({"type": "touch", "t": f["t"], "player": s["id"], "team": s["team"],
                       "x": None if b is None else round(float(b[0]), 2),
                       "y": None if b is None else round(float(b[1]), 2),
                       "duration": round((s["end"] - s["start"] + 1) / fps, 2)})
        if si + 1 >= len(spells):
            continue
        n = spells[si + 1]
        k0, k1 = s["end"], n["start"]
        if n["id"] == s["id"] or (k1 - k0) / fps > a.pass_gap or not same_segment(frames, k0, k1):
            continue
        p0, p1 = ball_xy(frames[k0]), ball_xy(frames[k1])
        if p0 is None or p1 is None:
            continue
        if np.linalg.norm(p1 - p0) < a.min_pass:
            continue  # ball changes feet within a couple of metres: a duel or a scramble, not a pass
        known = s["team"] is not None and n["team"] is not None
        ev = {"type": "pass", "t_start": frames[k0]["t"], "t_end": frames[k1]["t"],
              "from": s["id"], "to": n["id"], "team": s["team"],
              "start": [round(float(p0[0]), 2), round(float(p0[1]), 2)],
              "end": [round(float(p1[0]), 2), round(float(p1[1]), 2)],
              "length": round(float(np.linalg.norm(p1 - p0)), 2),
              "completed": (s["team"] == n["team"]) if known else None,
              "aerial": si in flights}
        if si in flights:
            ev["max_height"] = flights[si]["max_height"]
        events.append(ev)
    for fl in loose:  # shots, clearances, headed-on balls: in the air, but nobody took it straight after
        events.append({"type": "aerial_ball", "t_start": fl["t_takeoff"], "t_end": fl["t_land"],
                       "player": fl["from"], "team": fl["team"], "start": fl["start"], "end": fl["land"],
                       "max_height": fl["max_height"], "speed_ms": fl["speed_ms"]})
    events.sort(key=lambda e: e.get("t", e.get("t_start")))
    return events


def build_stats(frames, fps, owner, spells, events):
    team_of = {p["id"]: p.get("team") for f in frames for p in f["players"]}
    role_of = {p["id"]: p.get("role") for f in frames for p in f["players"]}
    passes = [e for e in events if e["type"] == "pass"]

    # possession by time: frames owned, plus the travel time of completed passes, go to that team
    team_frames = defaultdict(int)
    for k, o in enumerate(owner):
        if o is not None and team_of.get(o) is not None:
            team_frames[team_of[o]] += 1
    for p in passes:
        if p["completed"]:
            team_frames[p["team"]] += int(round((p["t_end"] - p["t_start"]) * fps))
    total = sum(team_frames.values())

    teams = {}
    for tm in (0, 1):
        tp = [p for p in passes if p["team"] == tm]
        done = [p for p in tp if p["completed"]]
        # shape: outfield players only, per frame; 10th-90th percentile spread so one straggler
        # doesn't define it
        width, depth, cx = [], [], []
        for f in frames:
            if not f["view"]:
                continue
            pts = np.array([(p["x"], p["y"]) for p in f["players"]
                            if p.get("team") == tm and p.get("role") != "goalkeeper"])
            if len(pts) >= 4:
                width.append(np.percentile(pts[:, 1], 90) - np.percentile(pts[:, 1], 10))
                depth.append(np.percentile(pts[:, 0], 90) - np.percentile(pts[:, 0], 10))
                cx.append(pts[:, 0].mean())
        teams[str(tm)] = {
            "possession_pct": round(100 * team_frames[tm] / total, 1) if total else None,
            "passes": len(tp), "passes_completed": len(done),
            "pass_accuracy_pct": round(100 * len(done) / len(tp), 1) if tp else None,
            "aerial_passes": sum(p["aerial"] for p in tp),
            "touches": sum(1 for s in spells if s["team"] == tm),
            "avg_width_m": round(float(np.mean(width)), 1) if width else None,
            "avg_depth_m": round(float(np.mean(depth)), 1) if depth else None,
            "avg_line_x": round(float(np.mean(cx)), 1) if cx else None,
        }

    # per player
    track = defaultdict(list)
    for k, f in enumerate(frames):
        for p in f["players"]:
            track[p["id"]].append((k, p["x"], p["y"]))
    players = {}
    for pid, o in track.items():
        dist, top = 0.0, 0.0
        for (k0, x0, y0), (k1, x1, y1) in zip(o[:-1], o[1:]):
            if k1 - k0 == 1:
                d = float(np.hypot(x1 - x0, y1 - y0))
                if d * fps < 12.0:  # ignore calibration jumps
                    dist += d
        # top speed over 1 s windows (per-frame speeds are too noisy)
        w = int(round(fps))
        for j in range(len(o) - w):
            if o[j + w][0] - o[j][0] == w:
                top = max(top, float(np.hypot(o[j + w][1] - o[j][1], o[j + w][2] - o[j][2])) * fps / w)
        mine = [p for p in passes if p["from"] == pid]
        players[str(pid)] = {
            "team": team_of.get(pid), "role": role_of.get(pid),
            "seconds_tracked": round(len(o) / fps, 1),
            "avg_x": round(float(np.mean([q[1] for q in o])), 2),
            "avg_y": round(float(np.mean([q[2] for q in o])), 2),
            "touches": sum(1 for s in spells if s["id"] == pid),
            "passes": len(mine), "passes_completed": sum(1 for p in mine if p["completed"]),
            "passes_received": sum(1 for p in passes if p["to"] == pid and p["completed"]),
            "distance_m": round(dist, 1),
            "top_speed_ms": round(min(top, 11.0), 2),
        }
    return {"teams": teams, "players": players,
            "note": "distance and time only count while the player is on screen"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="match.json from track_match.py")
    ap.add_argument("--out", default="match_events.json")
    ap.add_argument("--poss-radius", type=float, default=2.0, help="m: ball this close to a player can be his")
    ap.add_argument("--keep-radius", type=float, default=3.0, help="m: the owner keeps the ball within this")
    ap.add_argument("--rel-speed", type=float, default=7.0,
                    help="m/s: the ball must move roughly with the player (filters balls flying past)")
    ap.add_argument("--min-poss", type=float, default=0.15, help="s: shortest spell counted as a touch")
    ap.add_argument("--pass-gap", type=float, default=6.0, help="s: longest ball travel counted as a pass")
    ap.add_argument("--min-pass", type=float, default=3.0,
                    help="m: possession changing over a shorter distance is a duel / scramble, not a pass")
    ap.add_argument("--image-size", default="1920x1080",
                    help="video size, only used if match.json doesn't record it (older files)")
    ap.add_argument("--air-dev", type=float, default=4.0,
                    help="m: bulge away from a straight rolling path (+10%% of pass length) that means the "
                         "ball was in the air")
    a = ap.parse_args()

    with open(a.json) as fh:
        data = json.load(fh)
    frames, fps = data["frames"], data["fps"]

    owner, spells = find_possession(frames, fps, a)
    flights, loose, method = {}, [], "flight time"
    if any("H" in f for f in frames):
        vid = data.get("video") or {}
        w, h = vid.get("width"), vid.get("height")
        if not (w and h):
            w, h = (int(v) for v in a.image_size.lower().split("x"))
            print(f"note: no video size in {a.json}; assuming {w}x{h} (--image-size)")
        cams = frame_cameras(frames, w, h)
        print(f"cameras recovered for {sum(c is not None for c in cams)}/{sum(f['view'] for f in frames)} frames")
        flights, loose = find_flights_3d(frames, fps, spells, cams, w, a)
        method = "3D fit"
    else:
        print("note: no calibration in the file (run track_match.py with --save-h) -> flight-time heights only")
    # Transits the 3D fit could not judge get the flight-time method, but only where there was no camera.
    # Where the 3D fit had a camera and found no flight, the flight-time method used to "find" 10-12 m
    # lobs in slow 4-second passes (rolling balls with detection gaps), so it is not used there.
    judged = set(flights) | (set(range(len(spells))) if method == "3D fit" else set())
    if method == "3D fit":
        judged = {si for si in judged if si in flights or cams[spells[si]["end"]] is not None}
    fallback = find_flights(frames, fps, spells, a, skip=judged)
    flights.update(fallback)
    events = build_events(frames, fps, spells, flights, a, loose)
    stats = build_stats(frames, fps, owner, spells, events)

    for k, f in enumerate(frames):
        f["owner"] = owner[k]
        if f.get("ball") is not None and len(f["ball"]) == 2:
            f["ball"] = [f["ball"][0], f["ball"][1], 0.0]
    data["events"] = events
    data["stats"] = {**data.get("stats", {}), "match": stats}  # keep track_match's own stats
    data["ball_format"] = "[x, y, z]: z = height in metres"
    with open(a.out, "w") as fh:
        json.dump(data, fh)

    passes = [e for e in events if e["type"] == "pass"]
    print(f"{len(spells)} touches, {len(passes)} passes "
          f"({sum(1 for p in passes if p['completed'])} completed, {sum(p['aerial'] for p in passes)} aerial), "
          f"{len(flights) + len(loose)} lofted flights ({len(loose)} without a receiver; {method}"
          f"{f', {len(fallback)} by flight time' if method == '3D fit' and fallback else ''}) -> {a.out}")
    for tm, s in stats["teams"].items():
        print(f"  team {tm}: possession {s['possession_pct']}%, passes {s['passes_completed']}/{s['passes']} "
              f"({s['pass_accuracy_pct']}%), width {s['avg_width_m']} m, depth {s['avg_depth_m']} m")


if __name__ == "__main__":
    main()