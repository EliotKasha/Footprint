# Soccer Match 3D Replay (Unity)

Part of the **Footprint** code jam project: soccer match video → computer vision tracking (`src/`, by the CV team) → **3D replay in Unity** (this folder) → formation and player-rating analysis.

The CV pipeline outputs `match_events.json` (players, ball, possession, passes and fitted ball flights). This Unity project replays that file in a 3D stadium, cleaning up tracking errors as it loads.
## Running it
1. Install **Unity 6.6 (6000.6.10f1)** via Unity Hub and open **this folder** (`unity/`) as a project.
2. Open `Assets/Scenes/Replay.unity`.
3. Press **Play**. (Or use the menu **Soccer Replay → Create Replay Scene and Play** to rebuild the scene.)

**Controls:** Space play/pause · ←/→ skip 1 s · `[` `]` speed (0.25×–4×) · `1` `2` `3` broadcast / top-down / follow-ball camera · `R` restart.
The bar at the bottom has the same controls plus a timeline slider. The top-left panel is a live event feed.

**Loading a new CV output:** copy the JSON into `Assets/StreamingAssets/matches/`, select the `Replay` object and set **Match File**. The current one is `matches/match_events_v5.json`.

## What's been built

### Replay
- Loads the CV team's JSON (any frame rate). Positions are in meters, and center or corner origin is detected (or read from `pitch.origin`).
- **Flip Y** mirrors the pitch top to bottom. It's on for the current data, because the CV y axis points the other way.
- Players wear the **real kit colors** from the file's `teams` block (falls back to red and blue), with a chest stripe that contrasts with each kit. They're labeled with their tracking ID and smoothly interpolated. **Referees** are drawn in black ("REF").
- **Goalkeepers:** solid when tracked. When off camera they're semi-transparent ("GK (est.)"), stay near where they were last or next seen, and are capped at walking speed. A keeper the camera never sees is estimated from the ball's position.
- **Keeper validation:** each team's goal is decided from the keeper with the most data. A track labeled goalkeeper that averages more than 25 m from its own goal is treated as a misclassified outfield player.

### Off-camera players
Broadcast footage only shows part of the pitch, so players drop in and out of the data. The replay keeps them on the pitch:
- **Tracks are joined into people:** a same-team track that starts within 8 s of another ending, close enough to have run there unseen, is treated as the same person. On the current clip, 60 tracks become 42 people.
- **Estimated players are drawn see-through with no number.** Between two sightings they glide from the last-seen to the next-seen position. Before the first or after the last sighting they stay near that spot and move with their team's overall shape, for up to 8 s.
- At most **10 outfield players per team** are shown. When there are more estimates, the ones closest in time to a real sighting win, and changes fade instead of popping.
- Tunable under **Off-camera players** on the `Replay` object (see-through amount, hold time, team follow, join gap and speed). This is a first version and still needs tuning.

### Players' body motion
- Each player has a chest stripe (shows facing) and a shorts band.
- Velocity is measured over a window centered on the current moment, then smoothed. Players face their direction of travel when moving and turn to the ball when standing.
- They lean into runs and bob with a speed-scaled running rhythm. All of this follows pause, scrubbing and playback speed.

### Ball and events
- Ball with height and a ground shadow, and a **possession ring** under the player on the ball (`owner`).
- **Fitted ball flights** (`ball_flights`): during a flight the ball follows the smooth arc (a straight ground track plus a parabola for height) instead of noisy per-frame points, so high balls never jitter. It starts on the grass and ends at `end_height`: 0 for a landing, above 0 for a header or volley.
- **Pass arcs:** white = completed, red = failed, amber = loose lofted ball (`aerial_ball`).
- **Event feed** of passes, touches and balls in the air.

### Tracking cleanup on load
The CV output still has errors, so the loader fixes them before playback:
- **Camera calibration jumps** (many players shifting together) are cancelled and eased out over about 2 s.
- **One-frame glitches** are removed, and **ID swaps** (one ID jumping to a different person) are split into separate tracks instead of drawing someone zooming across the pitch.
- Frame-rate-independent smoothing (±0.2 s), fade in and out when tracks start or stop, and short dropouts are interpolated.
- On the latest data, 99% of displayed movement is under 6.6 m/s and under 0.03% exceeds 12 m/s (raw data had about 33 m/s spikes).

### Stadium
- A two-tier bowl with closed corners, concourse, aisles and roof, built in code to fit the 105 × 68 m pitch.
- About 13k instanced crowd figures in team colors with an away section, and a gentle bob.
- **Ad boards** with the **Sheffield Thursday** crest and name. The crest is cut out of `Assets/Replay/Textures/SheffieldThursdayLogo.png`.
- Sunny-afternoon lighting with the stand shadow across the pitch, procedural sky, haze, post-processing (ACES, bloom, vignette) and MSAA.
- The broadcast camera sits on a gantry at the front of the main stand.

Almost everything is tunable on the `Replay` object's Inspector: Data cleanup, Body motion, Goalkeepers, Stadium Builder (colors, crowd, boards, sun, fog), and Use Fitted Flights.

## Code layout
| Path | What |
|---|---|
| `Assets/Replay/MatchData.cs` | JSON format classes, loading, coordinate handling, tracking cleanup, tracks, ball flights |
| `Assets/Replay/ReplayController.cs` | Playback, cameras, players, keepers, ball, possession, pass arcs, UI |
| `Assets/Replay/PlayerBody.cs` | Player body visuals and motion (facing, lean, bob) |
| `Assets/Replay/PitchBuilder.cs` | Pitch, markings and goals |
| `Assets/Replay/StadiumBuilder.cs` | Stands, roof, crowd, ad boards, lighting and post-processing |
| `Assets/Replay/Editor/ReplaySceneSetup.cs` | **Soccer Replay** menu that builds the scene |
| `Assets/StreamingAssets/matches/` | CV outputs used for testing (`match_events_v5.json` is the latest) |
| `docs/plan.md` | Original weekend plan |

## Data format (CV output → Unity)
```jsonc
{
  "pitch": { "length": 105, "width": 68, "origin": "center" },
  "teams": { "0": { "kit_hex": "#a0adba" }, "1": { "kit_hex": "#303f56" } },   // optional kit colors
  "fps": 14.99,
  "frames": [
    { "t": 0.0,
      "players": [ { "id": 7, "team": 0, "role": "player", "x": -12.5, "y": 4.2, "interp": false } ],
      "ball": [x, y, z],          // z = height in meters; may be null
      "owner": 11 }               // player on the ball; may be null
  ],
  "events": [
    { "type": "pass", "t_start": 4.27, "t_end": 8.14, "from": 14, "to": 11, "team": 0,
      "start": [x, y], "end": [x, y], "length": 16.5, "completed": true, "aerial": true, "max_height": 9.6 },
    { "type": "touch", "t": 8.14, "player": 11, "team": 0, "x": -35.4, "y": -11.9, "duration": 4.4 },
    { "type": "aerial_ball", "t_start": 28.6, "t_end": 30.5, "player": null, "start": [x, y], "end": [x, y], "max_height": 4.1 }
  ],
  "ball_flights": [
    { "t_takeoff": 28.67, "t_land": 31.14, "max_height": 7.24, "end_height": 0.0, "start": [x, y], "land": [x, y],
      "speed_ms": 14.3, "from": 37, "team": 1, "method": "3d bridge", "kind": "pass" }
  ]
}
```
- `role`: `player`, `goalkeeper` or `referee`. `team`: 0 or 1 (`null` for referees). Nulls in ID fields are handled.
- Unknown extra fields (`H`, `view_polygon`, `calib`, `stats`, …) are ignored safely.

## Notes for the CV side
- **Kit colors** (`teams.kit_hex`) are used directly, so colors now follow the real kits. Please keep exporting them.
- **Role classifier:** in `match_events_v5`, track 58 is labeled goalkeeper while playing about 25–35 m from the wrong goal. The replay treats it as an outfield player.
- **Confirm the y-axis direction.** The replay currently needs Flip Y.
- **`ball_flights`:** a few fits are too short for their height (e.g. 2.95 m in 0.67 s implies about 5× gravity). Adding takeoff and landing heights (`start_z`, `land_z`) or a launch velocity would let headers and volleys be drawn exactly instead of snapping to the ground.
- ID stability and camera calibration (still "coasting" most frames) are the biggest remaining accuracy factors.

## Next up
- **Tier 1:** animated humanoid players (Mixamo character and locomotion blend tree) driven by the same motion data.
- **Tier 2:** pass, kick and receive animations triggered from events.
- Side panel for formations and player ratings from the analysis model, a standalone build, and a backup demo video.
