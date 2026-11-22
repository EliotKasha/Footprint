# FootPrint - Football Optical Observation and Position Reconstruction Intelligence Network
*FootPrint* takes a video of a soccer match and outputs (for every frame) where every player and the ball is located on the pitch in meters. It also tracks possession, passes, ball height and other match stats. A Unity project reads in that JSON and creates a fully animated replay of the match in 3D, like a Video Game recreation.

---

## Demo

![Demo](demo.gif "Alt")

---

## Quick start

```bash
./go.sh test1.mp4 --stride [n] --show  # Stride of n frames, default 2. Show output
./go.sh test1.mp4 --skip-track         # Reuse match.json, re-run events + viz only (faster)
./go.sh test1.mp4 --no-viz             # Skip check video
./go.sh test1.mp4 -- --debug-teams     # Debug mode (everything after -- passed directly to track_match.py)
```

Results will go to `out/<video name>/` as follows:
* `match.json` — Raw tracking: players, ball (2D), pitch calibration
* `match_events.json` — Everything in `match.json` plus ball height (z in json maps later to y in Unity), possession, events, and other stats
* `viz.mp4` — Video of 2d minimap adjacent to original footage
* `pipeline.log` — Console log of run w/ timestamps

### Model download

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q" -O football-player-detection.pt
gdown "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V" -O football-ball-detection.pt
gdown "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf" -O football-pitch-detection.pt
```

On a Mac, Core ML exports of the two detectors are much faster. The pitch model is faster as `.pt`:

```bash
yolo export model=football-player-detection.pt format=coreml "imgsz=[736,1280]"
yolo export model=football-ball-detection.pt format=coreml "imgsz=[736,1280]"
```

Optimal setting are encoded in bash script by default, simply run:

```bash
./go.sh #...
```

Which for me yielded best performance on MacOS (M4). This could very well differ by machine.

---

## Tools

* **Ultralytics YOLO:** Player detector (ball / goalkeeper / player / referee), dedicated ball detector, pitch keypoint model w/ 32 landmarks (Roboflow sports pitch config).
* **Supervision / trackers:** ByteTrack multi-object tracking.
* **OpenCV:** Video I/O, homographies, optical flow (camera motion), color conversion, drawing.
* **NumPy / SciPy:** Least-squares fits (line refinement, 3D ball flights), Hungarian matching (track stitching), smoothing.
* **Scikit-learn:** KMeans for team split by jersey color.
* **PyTorch:** MPS ("GPU") binding for MacOS. Alternate binding needed for equivalent performance on other OS.

---

## Stage 1: `track_match.py` (video --> match.json)

Most expensive step. It is broken into two parts:
* Per-frame loop
* Offline cleanup (Whole clip at once)

### 1a. Per-frame loop

For each processed frame (every n-th frame for `--stride n`):

1. **Detect players and ball:** The player model finds players, keepers and referees, while the ball model locates the ball (detections in top 70% of a player's box discarded for high false positive rate).

2. **Calibrate the camera: image → pitch homography `H`.** This is the core of everything. A homography is a 3×3 matrix mapping a pixel on the grass to (x, y) meters on the pitch.
   - **Base:** The pitch model finds key landmarks (corners, boxes, center circle, etc...). With enough of them (`--min-kp` = 6) we solve for `H`. It runs every `--kp-every` 3 frames to save time.
   - **Line refinement:** White pitch lines are extracted (top-hat filter) and `H` is nudged such that the projected pitch lines sit on them (point-to-line least squares, coarse-to-fine). This fixes wobble from keypoints alone and is scored as `line_fit` (0–1).
   - **Gating:** New `H` notably different with the previous one is rejected unless it is a *notably* better fit to the lines (otherwise it is likely a blip).
   - **Coasting:** Between keypoint frames, or when keypoints fail, we measure camera motion via optical flow on the background and the last good `H` is carried forward. Works up to `--coast` 3 s. After that, the frame is marked `view: false`.

3. **Track people (ByteTrack) with camera-motion compensation:** Boxes warped to undo pan before matching such that a fast pan doesn't make everyone "jump" and lose their IDs.

4. **Project to pitch:** Each player's foot position (bottom-center of bounding-box) goes through `H` to give pitch meters. The ball's pixel goes through `H` in a similar way, though this is only correct when the ball is on the grass (detail in Stage 2).

5. **Sample jersey colors**: Needed for team identification. Torso crop w/o grass pixels --> median color. Brightness is down-weighted so sun/shadow doesn't split a team. Occluded players skipped.

6. **View polygon:** Which part of the pitch is currently in-view. Useful for Unity to fade out-of-frame players.

### 1b. Offline cleanup

1. **Drop unsettled start frames:** First frames before calibration locks in are junk.
2. **`choose_ball_path`:** DP over ball candidates per frame. Pick the most consistent path.
3. **`split_impossible_jumps`:** A track (one player's positions over time) moving faster than a sprint (8 m/s) between frames is two people sharing an ID: split it (ignored when 40%+ of players jump together, which is a calibration glitch, not ID swaps).
4. **`split_team_switches`:** A track whose jersey color changes team partway through is split.
5. **`stitch_tracks`:** Joins fragments of the same player across gaps (up to `--stitch-gap` 3 s), using Hungarian matching on where each fragment was heading, whether a person could physically cover that distance, same team, same role. Ambiguous matches are left alone.
6. **`remove_duplicate_tracks`:** Two tracks on the same person, or short phantom fragments are removed.
7. **`fill_player_gaps`:** Short gaps (<= `--fill-gap` 1 s) filled by interpolation if plausible.
8. **`smooth_tracks`:** Removes jitter by applying Gaussian smoothing over `--smooth-seconds` 0.35 s.
9. **`cap_speed`:** Players cannot move faster than 11 m/s.
10. **`interpolate_ball`:** Short ball gaps filled (on ground only).
11. **Teams / roles:** KMeans (k=2) on each track's jersey color. Outliers are goalies if near goal, refs otherwise. 
12. **Team order:** Team 0 = lighter kit (or warmer color if equally light) so labels don't swap between runs.
13. **Origin:** Coordinates shifted so (0, 0) is the kickoff spot / center of central circle

---

## Stage 2: `match_events.py` (match.json --> match_events.json)

Works purely on JSON (quick to iterate via `--skip-track`). Mostly 3D fitting.

1. **Possession (`find_possession`):** Ball belongs to nearest player within 2m (keeps it up to 3 m) if ball is roughly moving with him (relative speed ≤ 7 m/s) for at least 0.15 s. Also means this handles a ball being hidden behind the player in possession (he just keeps it).

2. **Recover 3D camera per frame (`frame_cameras`):** From `H` alone we can recover focal length, rotation, and camera position: Homography of a plane is `K·[r1 r2 t]` with r1/r2 orthonormal. This gives each frame's camera in 3D s.t. a pixel may become a *line of sight*.

3. **Ball flights:** Fitted as gravity arcs through the lines of sight. *An arc is only accepted if it explains the pixels notably better than a rolling ball*:
   - `find_flights_3d`: Passes and kicks right after possession spell ends.
   - `find_bridges`: High balls which leave detector's sight (namely going into/over the crowd). The rise off the kicker is joined to the first later run of sightings on the same arc. False detections within this gap are ignored. Also searches exact kick moment and a per-flight zoom correction (the zoom estimate drifts while panning, which mostly shows up as incorrect heights).
   - `find_hops_3d`: Janky movements, namely headers, bounces, and scrambles. Free-ball stretches are split into the longest possible pieces which fit an arc, with each starting where the previous one landed.
   - `clean_ball` (fallback): When the projected ball moves faster than any ground ball can (30 m/s) but no 3D fit worked, an arc is instead estimated from flight time (`ball_est: true`).
4. **`clean_ball` / `smooth_ball`:** Final safety net. No shown ball moves faster than 40 m/s between frames. If we can't tell where the ball is, it's `null` rather than a teleport (safer). Short blinks are filled, jitter is smoothed.
5. **Events and stats:** Touches, Passes (≥ 3 m, possession changes player within 6 s; completed if the receiver is a teammate), aerial passes, aerial balls w/o a receiver. Then per-team and per-player stats.

---

## Stage 3: `viz_match.py` (check video)

Minimap next to the source video.

- Pitch lines drawn from `H` over the video (cyan = fresh keypoint calibration, magenta = coasting).
- Shaded current camera view polygon.
- Players in team colors.
- Active player in possession highlight.
- Ball w/ height label and shadow (hollow = interpolated)
- Pass arrows (white = completed, red = lost, yellow = aerial ball)

---

## Data formatting

### Coords
- Meters. Pitch is 105m x 68m by default (`--length/--width`).
- Origin (0, 0) = kickoff spot. x runs along the length (−52.5 ... 52.5), y across the width (−34 ... 34). On minimap, positive y drawn downward.
- Ball height (m). 0 implies on grass.
- Time `t` in seconds from video begin.

### `match_events.json` (T/L)

```jsonc
{
  "pitch":  {"length": 105, "width": 68, "origin": "center", "x_range": [-52.5, 52.5], "y_range": [-34, 34]},
  "fps": 29.99,
  "video": {"width": 1920, "height": 1080},
  "teams":  {"0": {"kit_rgb": [235,235,235], "kit_hex": "#ebebeb"}, "1": {...}},
  "frames": [ ... one per processed frame ... ],
  "events": [ ... ],
  "ball_flights": [ ... ],
  "stats": {"match": {"teams": {...}, "players": {...}}, ...tracking counters...},
  "ball_format": "[x, y, z]: z = height in metres"
}
```

### A frame

```jsonc
{
  "t": 30.007,
  "view": true,            // false = no trustworthy calibration: don't trust positions this frame
  "calib": "coast",        // "kp" = fresh keypoint calibration, "coast" = carried by camera motion
  "n_kp": 5, "line_fit": 0.92,          // calibration quality (line_fit 0-1, higher is better)
  "players": [
    {"id": 33, "x": -33.85, "y": 20.29, "team": 1, "role": "player"}   // role: player | goalkeeper | referee
  ],
  "ball": [-33.98, -9.93, 7.83],        // or null when unknown
  "ball_air": true,        // in a flight / hop
  "ball_est": true,        // height estimated from flight time, not fitted (less reliable)
  "ball_interp": true,     // filled in, not seen
  "owner": null,           // player id in possession, or null
  "view_polygon": [[x, y], ...],        // visible part of the pitch
  "H": [[...],[...],[...]]              // image --> pitch matrix
}
```

Flags only present when true. Player IDs are stable within one run. Player IDs not guaranteed
across runs, though team 0/1 is.

### Events

```jsonc
{"type": "touch", "t": 0.967, "player": 14, "team": 1, "x": -17.67, "y": -18.38, "duration": 1.07}
{"type": "pass", "t_start": 2.0, "t_end": 4.4, "from": 14, "to": 16, "team": 1,
 "start": [-15.7, -14.9], "end": [-1.6, -30.9], "length": 21.3, "completed": true, "aerial": true, "max_height": 0.6}
{"type": "aerial_ball", "t_start": 5.3, "t_end": 6.1, "player": null, "team": null,
 "start": [...], "end": [...], "max_height": 0.79, "estimated": true}
```

### Ball flights (for drawing smooth arcs in Unity)

Each airborne stretch, with `kind`: `pass`, `no receiver` (shot, clearance, high ball), `free ball`
(header/bounce) or `estimated`. Each has `t_takeoff`, `t_land`, `start`, `land`, `max_height`. Fitted
ones also have `fit_px` (how far the arc is from the detections, in pixels) and `method` (`3d` or
`3d bridge`).

---

## Credits

* Eliot Kasha — Detection pipeline
* Ariyan Nagratha — 3d Unity game replay and interpolation
* Zain Gareau — Player animation and smoothing
* Gambol Guignard — Advanced game analytics