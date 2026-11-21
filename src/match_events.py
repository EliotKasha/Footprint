"""
Match events from track_match.py output: possession, touches, passes, ball height, and stats.

  python src/match_events.py match.json --out match_events.json

Reads match.json, writes a copy with:
  * top level  "ball_flights": every flight (takeoff/landing time and point, peak height), for animation
  * per frame  "owner": player id in possession, or null
               "ball": [x, y, z]  (z = height in metres, 0 on the ground)
               "ball_air": true while the ball is in a lofted flight ("ball_est": true if the arc is an
               estimate from the flight time because the ball wasn't visible enough to fit it)
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
    # Where a neighbour frame had no ball, measure the ball's speed against the nearest real sighting
    # within 0.3 s instead of leaving it unknown (unknown used to count as "slow", which let a flying
    # ball's projection be "owned" by players it passed over).
    reach = max(2, int(round(0.3 * fps)))
    seen = [k for k in range(n) if frames[k]["view"] and ball_xy(frames[k]) is not None]
    pos = {k: ball_xy(frames[k]) for k in seen}
    for idx, k in enumerate(seen):
        if ball_v[k] is not None:
            continue
        nb = [j for j in (seen[idx - 1] if idx > 0 else None, seen[idx + 1] if idx + 1 < len(seen) else None)
              if j is not None and abs(j - k) <= reach and same_segment(frames, min(j, k), max(j, k))]
        if nb:
            j = min(nb, key=lambda j: abs(j - k))
            ball_v[k] = (pos[max(j, k)] - pos[min(j, k)]) * fps / abs(j - k)
    return ball_v, pl_pos, pl_v


def hidden_possession(frames, fps, pl_pos, pl_v, ball_v, a, max_gap_s=3.0):
    """Possession while the ball can't be seen. A player holding or shielding the ball (a keeper with it
    in his hands, a dribbler with his back to the camera) hides it from the detector, sometimes for
    seconds. If the ball disappears at a player's feet and reappears at the same player's feet, he had
    it all along; if it reappears moving away from him, he had it until he kicked it, dated from how far
    and how fast the ball then travelled. Returns {frame: player id}."""
    def real(k):
        f = frames[k]
        return None if f.get("ball") is None or f.get("ball_interp") else np.array(f["ball"][:2], dtype=float)

    hidden = {}
    max_gap = int(round(max_gap_s * fps))
    for seg in segments(frames):
        seen = [k for k in seg if real(k) is not None]
        for k0, k1 in zip(seen[:-1], seen[1:]):
            if k1 - k0 < 2 or k1 - k0 > max_gap:
                continue
            b0 = real(k0)
            near = sorted((float(np.linalg.norm(p - b0)), pid) for pid, p in pl_pos[k0].items())
            if not near or near[0][0] > a.keep_radius:
                continue
            d0, p0 = near[0]
            bv, pv = ball_v[k0], pl_v[k0].get(p0)
            if d0 > a.poss_radius or bv is None or np.linalg.norm(bv - (pv if pv is not None else 0)) > a.rel_speed:
                continue  # not clearly at his feet: the ball may have been flying past him
            if p0 not in pl_pos[k1]:
                continue
            b1 = real(k1)
            d1 = float(np.linalg.norm(pl_pos[k1][p0] - b1))
            if d1 <= a.keep_radius:
                k_end = k1
            else:  # he played it during the gap
                nxt = next((j for j in seen if k1 < j <= k1 + int(round(0.3 * fps))), None)
                v = float(np.linalg.norm(real(nxt) - b1)) * fps / (nxt - k1) if nxt else 10.0
                travel = max(d1 - 0.5 * a.poss_radius, 0.0) / max(v, 5.0)
                k_end = max(k0, k1 - int(round(travel * fps)) - 1)
            for j in range(k0, k_end + 1):
                if p0 in pl_pos[j]:
                    hidden[j] = p0
    return hidden


def find_possession(frames, fps, a):
    """Per-frame owner (player id or None) and the list of possession spells."""
    ball_v, pl_pos, pl_v = velocities(frames, fps)
    team = {p["id"]: p.get("team") for f in frames for p in f["players"]}
    owner = [None] * len(frames)
    spells = []  # dicts: id, team, start, end (frame indices, inclusive)
    min_frames = max(2, int(round(a.min_poss * fps)))
    hold = max(1, int(round(0.2 * fps)))  # keep the owner through a short ball dropout

    hidden = hidden_possession(frames, fps, pl_pos, pl_v, ball_v, a)

    for seg in segments(frames):
        cur, cand, cand_start, cand_n, missing = None, None, None, 0, 0
        for k in seg:
            if k in hidden:  # ball hidden by the player holding / shielding it
                owner[k] = cur = hidden[k]
                cand, cand_n, missing = None, 0, 0
                continue
            b = ball_xy(frames[k])
            if b is None:
                missing += 1
                if missing <= hold:
                    # brief dropout: the owner keeps it, and a player about to win the ball keeps his
                    # count (resetting it on every missed frame meant possession rarely formed at 30 fps)
                    if cur is not None:
                        owner[k] = cur
                    continue
                cur, cand, cand_n = None, None, 0
                continue
            missing = 0
            near = []
            for pid, p in pl_pos[k].items():
                d = float(np.linalg.norm(p - b))
                if d > a.keep_radius:
                    continue
                bv, pv = ball_v[k], pl_v[k].get(pid)
                rel = float(np.linalg.norm(bv - (pv if pv is not None else 0))) if bv is not None else np.inf
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


def _fit_flight(obs, up, takeoff_on_ground, z0_max=None, anchor=None, end_anchor=None):
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
        if anchor is not None:  # continues from where the previous hop ended (8 px per metre of mismatch)
            r.extend(8.0 * (p0 - anchor))
        if end_anchor is not None:  # (t, point): must arrive where the next fitted piece starts
            te, Xe = end_anchor
            r.extend(8.0 * (p0 + v * te + 0.5 * acc * te * te - Xe))
        return np.asarray(r)

    # Keep the search physical: ball within 30 m of the pitch area it was seen in, at most 25 m up,
    # at most 45 m/s. Junk detections then fail fast instead of fitting absurd 1000 m/s "flights".
    zmax = 25.0 if z0_max is None else z0_max
    zlo, zhi = (-1.0, zmax) if up > 0 else (-zmax, 1.0)
    if anchor is not None:  # start the search at the anchor
        x0[:3] = anchor
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


def spell_point(frames, s, from_end):
    """Ball position at the end (or start) of a possession spell: the nearest frame inside the spell where
    the ball was actually seen, else the owner's position at that end."""
    ks = range(s["end"], s["start"] - 1, -1) if from_end else range(s["start"], s["end"] + 1)
    for k in ks:
        if frames[k].get("ball") is not None and not frames[k].get("ball_interp"):
            return ball_xy(frames[k])
    k = s["end"] if from_end else s["start"]
    me = next((p for p in frames[k]["players"] if p["id"] == s["id"]), None)
    return None if me is None else np.array([me["x"], me["y"]], dtype=float)


GROUND_BALL_MAX = 25.0   # m/s: a rolling ball moving faster than this between sightings is suspect
AIR_EVIDENCE = 30.0      # m/s: projected speed no ground ball reaches -> it was in the air
HOP_END_MAX = 2.5        # m: a free-ball hop starts and ends at ground or head height
MIN_FLIGHT_S = 0.65      # s: shorter "flights" (peak under ~0.5 m) are detection blips, kept on the ground


BRIDGE_MAX_S = 5.0        # s: longest high ball searched for (kick to first touch / landing)
FOCAL_SCALES = (0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3)  # zoom corrections tried per high ball


def _arc_at(p0, v, up, t):
    return p0 + v * t + 0.5 * np.array([0.0, 0.0, -up * G]) * t * t


def find_bridges(frames, fps, spells, cams, w, h, skip_t=()):
    """High balls that go out of the detector's sight (over the crowd, against the stands).

    The ball is seen rising off the kicker, lost for a second or more, and seen again coming down. Over
    the gap the detector often fires on other things (ad boards, socks, heads), and those false sightings
    used to decide the flight. Here the rise is joined to the first later run of sightings that lies on
    the SAME gravity arc; sightings off that arc (typically hundreds of px) are ignored and overwritten.
    The camera's zoom estimate drifts while it pans, which mostly shows up as wrong heights, so each arc
    also picks the zoom correction that fits best and keeps the ball above the grass. Changing the zoom
    never moves positions on the pitch (H is unchanged), only heights. Returns ({spell: info}, [info])."""
    n = len(frames)
    thr = max(6.0, 0.004 * w)
    seg_of = {}
    for si, sg in enumerate(segments(frames)):
        for k in sg:
            seg_of[k] = si
    linked, loose = {}, []
    real = lambda k: (k in seg_of and cams[k] is not None and frames[k].get("ball") is not None
                      and not frames[k].get("ball_interp") and not frames[k].get("ball_air"))
    for si, s in enumerate(spells):
        k0 = s["end"]
        if k0 not in seg_of or cams[k0] is None or round(frames[k0]["t"], 3) in skip_t:
            continue
        up = cams[k0]["up"]
        k_lim = min(n - 1, k0 + int(BRIDGE_MAX_S * fps))
        # rise: unbroken sightings right after the kick (gaps <= 0.3 s)
        b0 = ball_xy(frames[k0])
        if b0 is None:
            continue
        kick = np.r_[b0, 0.0]  # the arc starts at the kicker's foot
        rise, last = [], k0
        for k in range(k0 + 1, min(k_lim, k0 + int(1.0 * fps)) + 1):
            if seg_of.get(k) != seg_of[k0]:
                break
            if real(k):
                if k - last > int(0.3 * fps) and rise:
                    break
                rise.append(k)
                last = k
        if len(rise) < 5:
            continue
        pr = [np.array(ball_xy(frames[k])) for k in rise]
        fast = max(np.linalg.norm(pr[j + 1] - pr[j]) * fps / (rise[j + 1] - rise[j]) for j in range(len(rise) - 1))
        if fast < AIR_EVIDENCE:
            continue  # not flying off the kicker
        # later runs of sightings (gaps <= 3 frames), in time order
        later = [k for k in range(rise[-1] + int(0.25 * fps), k_lim + 1)
                 if seg_of.get(k) == seg_of[k0] and real(k)]
        runs, cur = [], []
        for k in later:
            if cur and k - cur[-1] > 3:
                runs.append(cur)
                cur = []
            cur.append(k)
        if cur:
            runs.append(cur)
        found = None
        for run in runs:
            if len(run) < 4:
                continue

            def fit_with(ks, scale, kt):
                """Arc taking off at frame kt through sightings ks; None unless it is a valid fit."""
                ks = [k for k in ks if k >= kt]
                cs = {k: (cams[k] if scale == 1.0 else
                          camera_from_H(frames[k]["H"], w, h, f=cams[k]["f"] * scale)) for k in ks}
                if any(c is None for c in cs.values()):
                    return None
                obs = [((k - kt) / fps, ball_pixel(frames[k]), cs[k]) for k in ks]
                if any(o[1] is None for o in obs):
                    return None
                fit = _fit_flight(obs, up, takeoff_on_ground=True, anchor=kick)
                if fit is None:
                    return None
                p0, v, rms = fit
                zs = np.array([up * _arc_at(p0, v, up, o[0])[2] for o in obs])
                if rms > thr or zs.min() < -0.6 or np.linalg.norm(p0[:2] - kick[:2]) > 3.0:
                    return None  # off the sightings, under the grass, or not starting at the kicker
                score = rms + 5.0 * max(0.0, -zs.min() - 0.2)
                return score, rms, p0, v, zs, kt
            # The foot meets the ball a few frames after possession ends: try takeoff 0..7 frames later.
            # Cheap test first (rise + the run's first 4 sightings); only takeoffs that pass it get the
            # search for the longest prefix of the run on the same arc (zoom as estimated).
            def longest(kt, sc):
                if fit_with(rise + run[:4], sc, kt) is None:
                    return None
                lo_m, hi_m, res = 4, len(run), None
                while lo_m <= hi_m:  # sightings past the landing or a header break the arc: bisect
                    mid = (lo_m + hi_m + 1) // 2
                    r = fit_with(rise + run[:mid], sc, kt)
                    if r is not None:
                        res, lo_m = (mid, sc, r), mid + 1
                    else:
                        hi_m = mid - 1
                return res
            best = None  # (m, scale, result)
            tried = {}
            for off in (0, 2, 4, 6):
                tried[off] = longest(k0 + off, 1.0)
            good = {o: r for o, r in tried.items() if r is not None}
            if good:
                o_best = max(good, key=lambda o: (good[o][0], -good[o][2][0]))
                for o in (o_best - 1, o_best + 1):
                    if 0 <= o <= 7 and o not in tried:
                        r = longest(k0 + o, 1.0)
                        if r is not None:
                            good[o] = r
                best = max(good.values(), key=lambda r: (r[0], -r[2][0]))
            else:  # maybe only the zoom is off
                for off in (0, 3, 6):
                    for sc in (0.7, 0.85, 1.15):
                        r = fit_with(rise + run[:4], sc, k0 + off)
                        if r is not None and (best is None or r[0] < best[2][0]):
                            best = (4, sc, r)
                if best is None:
                    continue
            m, _, r0 = best
            kt = r0[5]
            for sc in FOCAL_SCALES:  # best zoom correction for that arc, then extend it
                r = fit_with(rise + run[:m], sc, kt)
                if r is not None and r[0] < best[2][0]:
                    best = (m, sc, r)
            sc = best[1]
            if sc != 1.0:
                ext = longest(kt, sc)
                if ext is not None and ext[0] > best[0] and ext[2][0] <= best[2][0] + 1.0:
                    best = ext
            m, sc, (score, rms, p0, v, zs, kt) = best
            vz = up * v[2]
            peak = up * p0[2] + vz * vz / (2 * G) if vz > 0 else 0.0
            if peak < 2.0 or np.linalg.norm(v) > 40.0:
                continue
            found = (run[:m], sc, rms, p0, v, peak, kt)
            break
        if found is None:
            continue
        used, sc, rms, p0, v, peak, kt = found
        k_last = used[-1]
        t_last = (k_last - kt) / fps
        # landing: where the arc reaches the grass, if within 0.3 s after the last sighting on it
        disc = (up * v[2]) ** 2 + 2 * G * (up * p0[2])
        t_land = ((up * v[2]) + np.sqrt(max(disc, 0.0))) / G
        t_end = min(max(t_land, t_last), t_last + 0.3) if t_land > t_last else t_last
        k_end = min(k_lim, kt + int(round(t_end * fps)))
        # the fitted takeoff point can sit a metre or so from where the ball was last drawn at the
        # kicker's foot: fade that offset out over the first 0.2 s so the ball doesn't jump
        prev = ball_xy(frames[kt - 1]) if kt > 0 else None
        off0 = (np.array(prev) - p0[:2]) if prev is not None and np.linalg.norm(np.array(prev) - p0[:2]) < 3.0 \
            else np.zeros(2)
        n_blend = max(1, int(round(0.2 * fps)))
        for k in range(kt, k_end + 1):
            X = _arc_at(p0, v, up, (k - kt) / fps)
            X[:2] += off0 * max(0.0, 1.0 - (k - kt + 1) / n_blend)
            z = max(0.0, float(up * X[2]))
            frames[k]["ball"] = [round(float(X[0]), 2), round(float(X[1]), 2), round(z, 2)]
            if z > 0.05:
                frames[k]["ball_air"] = True
            else:
                frames[k].pop("ball_air", None)
            frames[k].pop("ball_interp", None)
            frames[k].pop("ball_est", None)
        for k in range(kt, k_end + 1):
            frames[k]["ball_fit"] = True
        X_end = _arc_at(p0, v, up, (k_end - kt) / fps)
        info = {"t_takeoff": frames[kt]["t"], "t_land": frames[k_end]["t"], "max_height": round(float(peak), 2),
                "from": s["id"], "team": s["team"],
                "start": [round(float(p0[0]), 2), round(float(p0[1]), 2)],
                "land": [round(float(X_end[0]), 2), round(float(X_end[1]), 2)],
                "end_height": round(max(0.0, float(up * X_end[2])), 2),
                "speed_ms": round(float(np.linalg.norm(v)), 1), "fit_px": round(rms, 2),
                "zoom_fix": sc, "method": "3d bridge", "_end": (k_end, X_end),
                "_fit": {"kt": kt, "ks": [k for k in rise + used if k >= kt], "scale": sc, "up": up,
                         "kick": kick, "rms": rms}}
        nxt = spells[si + 1] if si + 1 < len(spells) else None
        if nxt is not None and nxt["id"] != s["id"] and nxt["start"] <= k_end + int(1.0 * fps):
            linked[si] = info
        else:
            loose.append(info)
    return linked, loose


def join_bridges(frames, fps, bridges, hops, cams, w, h):
    """A high ball's landing and the header/bounce that follows are fitted separately, and along the
    camera's line of sight a few metres of depth trade off against a metre of height, so the two can
    disagree by several metres at the hand-over. Refit the high ball so that it arrives exactly where
    the next fitted hop starts (if that keeps it on its sightings), and draw it up to that frame."""
    thr = max(6.0, 0.004 * w)
    for fl in bridges:
        info = fl.pop("_fit", None)
        k_end, X_end = fl["_end"]
        if info is None:
            continue
        nxt = [hp for hp in hops if 0 <= hp["_start"][0] - k_end <= int(0.5 * fps)]
        if not nxt:
            continue
        k_h, X_h = min(nxt, key=lambda hp: hp["_start"][0])["_start"]
        if np.linalg.norm(X_h - X_end) < 1.0:
            continue
        kt, sc, up = info["kt"], info["scale"], info["up"]
        cs = {k: (cams[k] if sc == 1.0 else camera_from_H(frames[k]["H"], w, h, f=cams[k]["f"] * sc))
              for k in info["ks"]}
        obs = [((k - kt) / fps, ball_pixel(frames[k]), cs[k]) for k in info["ks"]]
        fit = _fit_flight(obs, up, takeoff_on_ground=True, anchor=info["kick"],
                          end_anchor=((k_h - kt) / fps, X_h))
        if fit is None:
            continue
        p0, v, rms = fit
        X_hit = _arc_at(p0, v, up, (k_h - kt) / fps)
        if rms > 2 * thr or np.linalg.norm(X_hit - X_h) > 1.0:
            continue  # can't meet it without leaving the sightings: keep the separate fits
        peak = 0.0
        for k in range(kt, k_h + 1):
            X = _arc_at(p0, v, up, (k - kt) / fps)
            z = max(0.0, float(up * X[2]))
            peak = max(peak, z)
            if k < k_h:
                frames[k]["ball"] = [round(float(X[0]), 2), round(float(X[1]), 2), round(z, 2)]
                frames[k]["ball_fit"] = True
                if z > 0.05:
                    frames[k]["ball_air"] = True
                frames[k].pop("ball_interp", None)
                frames[k].pop("ball_est", None)
        fl.update({"t_land": frames[k_h]["t"], "max_height": round(peak, 2), "fit_px": round(rms, 2),
                   "land": [round(float(X_h[0]), 2), round(float(X_h[1]), 2)],
                   "end_height": round(max(0.0, float(up * X_h[2])), 2),
                   "speed_ms": round(float(np.linalg.norm(v)), 1), "joined": True})


def find_hops_3d(frames, fps, cams, owner, w, max_hop_s=3.0, min_obs=6, step=3, anchors=None):
    """Aerial play nobody controls (headers, bounces, scrambles) has no possession spell to start a flight
    fit from, so it was shown as a ball sliding back and forth along the ground. Here every stretch with
    no owner and a fast-moving projected ball is split, greedily, into the longest pieces that fit a
    gravity arc through the ball's lines of sight. An arc is kept only if it explains the camera clearly
    better than a rolling ball and is physically sensible. Returns the list of hops."""
    thr = max(3.0, 0.0025 * w)
    n = len(frames)
    hops = []
    seg_of = {}
    for si, sg in enumerate(segments(frames)):
        for k in sg:
            seg_of[k] = si
    # stretches of free ball (no owner, not already in a fitted flight)
    free, cur = [], []
    for k in range(n):
        ok = k in seg_of and owner[k] is None and not frames[k].get("ball_air") and cams[k] is not None
        if ok and cur and seg_of[k] == seg_of[cur[-1]]:
            cur.append(k)
        else:
            if len(cur) >= int(0.4 * fps):
                free.append(cur)
            cur = [k] if ok else []
    if len(cur) >= int(0.4 * fps):
        free.append(cur)
    anchors = anchors or {}
    for stretch in free:
        last_end = None  # (frame, 3D point) where the previous accepted hop (or a high ball) landed
        for kk in range(stretch[0] - 1, stretch[0] - 4, -1):
            if kk in anchors:
                last_end = (kk, anchors[kk])
                break
        obs = []
        for k in stretch:
            if frames[k].get("ball") is not None and not frames[k].get("ball_interp"):
                px = ball_pixel(frames[k])
                if px is not None:
                    obs.append((k, px, cams[k]))
        if len(obs) < min_obs:
            continue
        pos = [np.array(frames[k]["ball"][:2], float) for k, _, _ in obs]
        fastest = max((np.linalg.norm(pos[j + 1] - pos[j]) * fps / (obs[j + 1][0] - obs[j][0])
                       for j in range(len(obs) - 1)), default=0.0)
        if fastest < 12.0:
            continue  # a slow rolling ball: nothing to look for
        i = 0
        while i <= len(obs) - min_obs:
            best, j = None, i + min_obs
            while j <= len(obs) and (obs[j - 1][0] - obs[i][0]) / fps <= max_hop_s:
                part = [((k - obs[i][0]) / fps, px, cam) for k, px, cam in obs[i:j]]
                anchor = last_end[1] if last_end is not None and obs[i][0] - last_end[0] <= 3 else None
                # a hop starts at ground or head height (a bounce or a header), never metres up
                fit = _fit_flight(part, cams[obs[i][0]]["up"], takeoff_on_ground=False,
                                  z0_max=HOP_END_MAX, anchor=anchor)
                if fit is None or fit[2] > thr:
                    if best is not None:
                        break
                else:
                    best = (j, fit)
                j += step
            accepted = False
            if best is not None:
                j, (p0, v, rms) = best
                part = [((k - obs[i][0]) / fps, px, cam) for k, px, cam in obs[i:j]]
                up = cams[obs[i][0]]["up"]
                T = (obs[j - 1][0] - obs[i][0]) / fps
                ts = np.linspace(0, T, 25)
                z = up * p0[2] + up * v[2] * ts - 0.5 * G * ts * ts
                if T >= 0.3 and z.max() >= 0.5 and z.min() >= -1.0 and np.linalg.norm(v[:2]) <= 40.0 \
                        and z[0] <= HOP_END_MAX + 0.1 and z[-1] <= HOP_END_MAX + 0.5 \
                        and _fit_ground(part) > 1.5 * rms + 1.0:
                    acc = np.array([0.0, 0.0, -up * G])
                    k_a, k_b = obs[i][0], obs[j - 1][0]
                    for k in range(k_a, k_b + 1):
                        t = (k - k_a) / fps
                        X = p0 + v * t + 0.5 * acc * t * t
                        zz = max(0.0, float(up * X[2]))
                        frames[k]["ball"] = [round(float(X[0]), 2), round(float(X[1]), 2), round(zz, 2)]
                        if zz > 0.05:
                            frames[k]["ball_air"] = True
                        frames[k].pop("ball_interp", None)
                        frames[k].pop("ball_est", None)
                        frames[k]["ball_fit"] = True
                    X0, X1 = p0, p0 + v * T + 0.5 * acc * T * T
                    last_end = (k_b, X1)
                    hops.append({"t_takeoff": frames[k_a]["t"], "t_land": frames[k_b]["t"],
                                 "max_height": round(float(max(z.max(), 0)), 2),
                                 "start": [round(float(X0[0]), 2), round(float(X0[1]), 2)],
                                 "land": [round(float(X1[0]), 2), round(float(X1[1]), 2)],
                                 "fit_px": round(rms, 2), "method": "3d", "_start": (k_a, X0)})
                    i = j - 1
                    accepted = True
            if not accepted:
                i += 2
    return hops


def clean_ball(frames, fps, max_flight_s=4.5):
    """Make the ball track safe to animate, after the fitted flights are in.

    1. Reliable ground sightings: real detections, not already in a fitted flight, moving at a believable
       ground speed relative to the sightings around them.
    2. Between two reliable sightings, if the projected ball in between moved faster than any ground
       ball can (AIR_EVIDENCE), it flew: the ground track is a straight line between the two points and
       the height follows from gravity and the time in the air (peak g*T^2/8). These frames get
       "ball_air": true and "ball_est": true.
    3. Otherwise a short gap is filled along the ground; anything still impossible is set to null rather
       than shown as a teleport.
    Returns the list of estimated flights."""
    n = len(frames)
    seg_of = {}
    for si, seg in enumerate(segments(frames)):
        for k in seg:
            seg_of[k] = si

    def in_fit(k):
        b = frames[k].get("ball")
        return frames[k].get("ball_air") or frames[k].get("ball_fit") or (b is not None and len(b) > 2 and b[2] > 0)

    real = [k for k in range(n) if k in seg_of and frames[k].get("ball") is not None
            and not frames[k].get("ball_interp") and not in_fit(k)]
    pos = {k: ball_xy(frames[k]) for k in real}
    win = max(2, int(round(0.2 * fps)))
    support = max(3, int(round(0.3 * fps)))
    # A sighting is a reliable ground anchor only if at least two other sightings within 0.3 s agree with
    # it at a ground ball's speed. One agreeing neighbour isn't enough: a false detection on the
    # advertising boards lasting two frames used to anchor a high ball's landing at the touchline.
    reliable = set()
    real_set = sorted(real)
    for i, k in enumerate(real_set):
        agree = 0
        lo = i
        while lo > 0 and k - real_set[lo - 1] <= support:
            lo -= 1
        hi = i
        while hi + 1 < len(real_set) and real_set[hi + 1] - k <= support:
            hi += 1
        for j in real_set[lo:hi + 1]:
            if j != k and seg_of[j] == seg_of[k] and \
                    np.linalg.norm(pos[j] - pos[k]) * fps / abs(j - k) <= GROUND_BALL_MAX:
                agree += 1
        if agree >= 2:
            reliable.add(k)
    # fitted flights' end points count as reliable anchors too (the ball is on the grass there)
    rel = sorted(reliable)
    flights = []
    for k0, k1 in zip(rel[:-1], rel[1:]):
        if k1 - k0 < 2 or seg_of[k0] != seg_of[k1] or any(in_fit(j) for j in range(k0 + 1, k1)):
            continue
        T = (k1 - k0) / fps
        inner = [j for j in range(k0 + 1, k1) if j in pos]
        fast = 0.0
        seq = [k0] + inner + [k1]
        for a_, b_ in zip(seq[:-1], seq[1:]):
            fast = max(fast, np.linalg.norm(pos[b_] - pos[a_]) * fps / (b_ - a_))
        p0, p1 = pos[k0], pos[k1]
        across = np.linalg.norm(p1 - p0) / T
        if fast > AIR_EVIDENCE and MIN_FLIGHT_S <= T <= max_flight_s and across <= 40.0:
            peak = 0.0
            for j in range(k0 + 1, k1):
                t = (j - k0) / fps
                xy = p0 + (p1 - p0) * t / T
                z = 0.5 * G * t * (T - t)
                peak = max(peak, z)
                frames[j]["ball"] = [round(float(xy[0]), 2), round(float(xy[1]), 2), round(float(z), 2)]
                frames[j]["ball_air"] = True
                frames[j]["ball_est"] = True
                frames[j].pop("ball_interp", None)
            flights.append({"t_takeoff": frames[k0]["t"], "t_land": frames[k1]["t"], "max_height": round(peak, 2),
                            "start": [round(float(p0[0]), 2), round(float(p0[1]), 2)],
                            "land": [round(float(p1[0]), 2), round(float(p1[1]), 2)], "method": "estimated"})
        elif T <= 1.0 and across <= GROUND_BALL_MAX * 1.4:
            for j in range(k0 + 1, k1):  # short gap / wobble on the ground: straight line
                xy = p0 + (p1 - p0) * (j - k0) / (k1 - k0)
                frames[j]["ball"] = [round(float(xy[0]), 2), round(float(xy[1]), 2), 0.0]
                frames[j]["ball_interp"] = True
        else:
            for j in range(k0 + 1, k1):  # can't tell where it was: unknown rather than a teleport
                if not in_fit(j):
                    frames[j]["ball"] = None
                    frames[j].pop("ball_interp", None)
    # isolated sightings that are neither reliable nor inside a gap handled above: drop the impossible ones
    for k in real:
        if k not in reliable and frames[k].get("ball") is not None and not frames[k].get("ball_air"):
            nb = [j for j in rel if abs(j - k) <= win and seg_of[j] == seg_of[k]]
            if not nb or min(np.linalg.norm(pos[j] - pos[k]) * fps / abs(j - k) for j in nb) > AIR_EVIDENCE:
                frames[k]["ball"] = None
    # Final safety net: no jump between consecutive shown positions faster than a struck ball (40 m/s).
    # The less supported of the two is blanked, until none are left.
    gap_max = max(2, int(round(0.3 * fps)))
    for _ in range(50):
        shown = [k for k in range(n) if frames[k].get("ball") is not None]
        bad = False
        for a_, b_ in zip(shown[:-1], shown[1:]):
            if b_ - a_ > gap_max or seg_of.get(a_) != seg_of.get(b_) or \
                    frames[a_].get("ball") is None or frames[b_].get("ball") is None:
                continue  # (one of them was just blanked in this pass)
            pa, pb = (np.array((list(frames[j]["ball"]) + [0.0])[:3], float) for j in (a_, b_))
            if np.linalg.norm(pb - pa) * fps / (b_ - a_) > 40.0:
                rank = lambda k: (k in reliable) * 2 + bool(frames[k].get("ball_air"))
                victim = b_ if rank(b_) <= rank(a_) else a_
                frames[victim]["ball"] = None
                frames[victim].pop("ball_air", None)
                frames[victim].pop("ball_est", None)
                frames[victim].pop("ball_interp", None)
                bad = True
        if not bad:
            break
    return flights


def smooth_ball(frames, fps, seconds=0.15, max_fill=3):
    """Remove the detector's frame-to-frame wobble from the ball on the ground (it was ~10x jitterier than
    the smoothed players: 5-14 cm per frame, a visibly vibrating ball at 30 fps), and fill blinks of up to
    `max_fill` frames. Zero-lag: a 3-frame median then a Gaussian over +-`seconds`, per run of consecutive
    ground frames. Flight frames are left as they are (their arcs are already smooth)."""
    n = len(frames)
    seg = {}
    for si, sg in enumerate(segments(frames)):
        for k in sg:
            seg[k] = si
    # 1) fill short blinks on the ground when the motion across them is plausible
    shown = [k for k in range(n) if frames[k].get("ball") is not None]
    for a_, b_ in zip(shown[:-1], shown[1:]):
        if 1 < b_ - a_ <= max_fill + 1 and seg.get(a_) == seg.get(b_) is not None \
                and not frames[a_].get("ball_air") and not frames[b_].get("ball_air"):
            pa, pb = np.array(frames[a_]["ball"][:2], float), np.array(frames[b_]["ball"][:2], float)
            if np.linalg.norm(pb - pa) * fps / (b_ - a_) <= AIR_EVIDENCE:
                for j in range(a_ + 1, b_):
                    xy = pa + (pb - pa) * (j - a_) / (b_ - a_)
                    frames[j]["ball"] = [round(float(xy[0]), 2), round(float(xy[1]), 2), 0.0]
                    frames[j]["ball_interp"] = True
    # 2) smooth each run of consecutive ground frames
    k_half = max(1, int(round(seconds * fps)))
    sigma = max(k_half / 2.0, 0.5)
    runs, cur = [], []
    for k in range(n):
        b = frames[k].get("ball")
        ground = b is not None and not frames[k].get("ball_air")
        if ground and cur and (k - cur[-1] == 1 and seg.get(k) == seg.get(cur[-1])):
            cur.append(k)
        else:
            if len(cur) >= 3:
                runs.append(cur)
            cur = [k] if ground else []
    if len(cur) >= 3:
        runs.append(cur)
    for run in runs:
        xy = np.array([frames[k]["ball"][:2] for k in run], float)
        med = xy.copy()
        for j in range(1, len(xy) - 1):
            med[j] = np.median(xy[j - 1:j + 2], axis=0)
        out = np.empty_like(med)
        idx = np.arange(len(med))
        for j in range(len(med)):
            d = idx - j
            w = np.where(np.abs(d) <= k_half, np.exp(-0.5 * (d / sigma) ** 2), 0.0)
            out[j] = (w[:, None] * med).sum(axis=0) / w.sum()
        for k, p in zip(run, out):
            z = frames[k]["ball"][2] if len(frames[k]["ball"]) > 2 else 0.0
            frames[k]["ball"] = [round(float(p[0]), 2), round(float(p[1]), 2), z]


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
        # where the ball left and arrived: the last / first sighting inside each spell (the ball is often
        # hidden by the player at the exact moment), else the player's own position
        p0 = spell_point(frames, s, from_end=True)
        p1 = spell_point(frames, n, from_end=False)
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
    flights, loose, method, bridges = {}, [], "flight time", []
    if any("H" in f for f in frames):
        vid = data.get("video") or {}
        w, h = vid.get("width"), vid.get("height")
        if not (w and h):
            w, h = (int(v) for v in a.image_size.lower().split("x"))
            print(f"note: no video size in {a.json}; assuming {w}x{h} (--image-size)")
        cams = frame_cameras(frames, w, h)
        print(f"cameras recovered for {sum(c is not None for c in cams)}/{sum(f['view'] for f in frames)} frames")
        flights, loose = find_flights_3d(frames, fps, spells, cams, w, a)
        done_t = {round(fl["t_takeoff"], 3) for fl in list(flights.values()) + loose}
        b_linked, b_loose = find_bridges(frames, fps, spells, cams, w, h, skip_t=done_t)
        flights.update({si: v for si, v in b_linked.items() if si not in flights})
        loose += b_loose
        bridges = list(b_linked.values()) + b_loose
        if bridges:
            print(f"high balls joined across the detector's blind spot: {len(bridges)}")
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
    anchors = {}
    if method == "3D fit":
        for fl in bridges:
            k_e, X_e = fl["_end"]
            anchors[k_e] = X_e
    hops = find_hops_3d(frames, fps, cams, owner, w, anchors=anchors) if method == "3D fit" else []
    if bridges:
        join_bridges(frames, fps, bridges, hops, cams, w, h)
    for fl in bridges:
        fl.pop("_end", None)
        fl.pop("_fit", None)
    for hp in hops:
        hp.pop("_start", None)
    estimated = clean_ball(frames, fps)
    smooth_ball(frames, fps)
    events = build_events(frames, fps, spells, flights, a, loose)
    # passes whose ball was in the air only by the estimate: mark them aerial too
    for e in events:
        if e["type"] == "pass" and not e["aerial"]:
            ks = [k for k, f in enumerate(frames) if e["t_start"] <= f["t"] <= e["t_end"] and f.get("ball_air")]
            if ks:
                e["aerial"] = True
                e["max_height"] = max(frames[k]["ball"][2] for k in ks)
    passes_t = [(e["t_start"], e["t_end"]) for e in events if e["type"] == "pass"]
    for fl in estimated:  # flights no pass covers: shots, clearances, headers
        if not any(t0 - 0.2 <= fl["t_takeoff"] <= t1 for t0, t1 in passes_t):
            events.append({"type": "aerial_ball", "t_start": fl["t_takeoff"], "t_end": fl["t_land"], "player": None,
                           "team": None, "start": fl["start"], "end": fl["land"], "max_height": fl["max_height"],
                           "estimated": True})
    events.sort(key=lambda e: e.get("t", e.get("t_start")))
    stats = build_stats(frames, fps, owner, spells, events)

    for k, f in enumerate(frames):
        f.pop("ball_fit", None)
        f["owner"] = owner[k]
        if f.get("ball") is not None and len(f["ball"]) == 2:
            f["ball"] = [f["ball"][0], f["ball"][1], 0.0]
    data["events"] = events
    # every flight in one list, for animating arcs / bounces in Unity
    all_flights = [dict(v, kind="pass") for v in flights.values()] + [dict(v, kind="no receiver") for v in loose] + \
                  [dict(v, kind="estimated") for v in estimated] + [dict(v, kind="free ball") for v in hops]
    data["ball_flights"] = sorted(all_flights, key=lambda f: f["t_takeoff"])
    data["stats"] = {**data.get("stats", {}), "match": stats}  # keep track_match's own stats
    data["ball_format"] = "[x, y, z]: z = height in metres"
    with open(a.out, "w") as fh:
        json.dump(data, fh)

    passes = [e for e in events if e["type"] == "pass"]
    print(f"{len(spells)} touches, {len(passes)} passes "
          f"({sum(1 for p in passes if p['completed'])} completed, {sum(p['aerial'] for p in passes)} aerial), "
          f"{len(flights) + len(loose) + len(estimated) + len(hops)} lofted flights ({len(loose)} without a receiver, "
          f"{len(hops)} free-ball hops/headers, {len(estimated)} estimated from impossible ground speed; {method}"
          f"{f', {len(fallback)} by flight time' if method == '3D fit' and fallback else ''}) -> {a.out}")
    for tm, s in stats["teams"].items():
        print(f"  team {tm}: possession {s['possession_pct']}%, passes {s['passes_completed']}/{s['passes']} "
              f"({s['pass_accuracy_pct']}%), width {s['avg_width_m']} m, depth {s['avg_depth_m']} m")


if __name__ == "__main__":
    main()