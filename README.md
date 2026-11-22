# Soccer Match 3D Replay (Unity)

Part of the **Footprint** code jam project: soccer match video → computer vision tracking (`src/`, by the CV team) → **3D replay in Unity** (this folder) → formation and player-rating analysis.

The CV pipeline outputs `match_events.json` (players, ball, possession, passes and fitted ball flights). This Unity project replays that file in a 3D stadium, cleaning up tracking errors as it loads.
## Running it
1. Install **Unity 6.6 (6000.6.10f1)** via Unity Hub and open **this folder** (`unity/`) as a project.
2. Open `Assets/Scenes/Replay.unity`.
3. Press **Play**. The empty stadium appears with a **No match loaded** panel.
4. Click **Load match…** (or press `L`) and pick the pipeline's `match_events.json` from anywhere on your computer. The replay builds and plays it.

**Standalone build (Windows):** File → Build Profiles, scene list = `Replay.unity` only, Windows x64, Build. Run the `.exe` with all the files next to it. Already set up for this project: DirectX 11 (DirectX 12 showed a grey screen on hybrid-graphics laptops), .NET Framework API level (for the file dialog), and `Soccer Replay → Prepare Build Shaders`, which runs automatically before every build and saves one material per shader variant the game creates in code into `Assets/Replay/Resources/BuildShaderVariants/` (without it the build strips them and crashes or draws things wrong).

> Note: the menu **Soccer Replay → Create Replay Scene and Play** rebuilds `Replay.unity` from scratch with default settings (no animated players, Flip Y off). Don't use it on the existing scene.

**Controls:** Space play/pause · ←/→ skip 1 s · `[` `]` speed (0.25×–4×) · `1` `2` `3` broadcast / top-down / follow-ball camera · `4` free camera · `5` behind the goal (press again for the other end) · `L` load match · `Tab` stats panel · `F` formation lines · `R` restart.

**Free camera** (`4` or the **Free** button): hold the right mouse button and move the mouse to look, WASD to fly, E/Q up/down, Shift for 3× speed, mouse wheel to change fly speed (shown in the bottom bar). Works while paused.
The bar at the bottom has the same controls plus a timeline slider. The top-left panel is a live event feed.

**Stats overlay** (from the analysis teammates' `PitchOverlay` scripts): once a match is loaded, a scoreboard bar sits across the top over the stands: team names, score (`– : –` until set), clock, possession, and each team's measured shape chip (`DEFENDING 4-5-1`, or `–` when not measured). **STATS** on the bar, **Stats** in the bottom bar, or `Tab` opens the full panel: goal threat (placeholder estimate), passes, and team shape (width, depth, line height). **Formation lines** link neighbouring tracked players in each team's line and thicken with passes between them, coloured by line: **blue = defence, yellow = midfield, red = attack** (a key appears above the bottom bar; with fewer than 3 lines on camera a line's role comes from how far it is from its own goal: under 40 m defence, under 65 m midfield; colours and zones are on the `FormationLinks` object, untick **Color By Line** for the old single pink); turn them on or off with the **Lines** button in the bottom bar or `F` (default: **Show Formation Lines** on the `Replay` object).
- The overlay reads the same file you load and follows the replay clock (pause, scrub, speed, loop).
- **Shape chips are measured live in Unity** from the match file alone (`LiveShapes.cs`, a C# port of formation-ml's `shape.py`): about 3 moments a second, a moment counts when all 10 outfield players are on camera (or 9 when the camera view shows which line the missing one is in), players are split into the 3 tightest lines by depth, and the moments of the **last 20 s** vote. Attacking (team has the ball) and defending are measured separately; the chip shows the one for the current phase, fainter at lower confidence. Checked: with the whole clip and 20 moments it gives exactly what `export_for_unity.py` writes (clips 1-3, v5, v6), and the 20 s windows match a Python version too. Settings on the `MatchOverlay` object: **Shape Window** (20 s; 0 = whole clip so far), **Shape Min Moments** (9 = 3 s of usable footage; formation-ml uses 20, which on our 30 s clips often shows `–`), **Prefer Analysis File** (use a matching `<match file>.analysis.json` instead; it's only used if its `source_clip` is that same file). Team names come from that file if present; otherwise White and Black.
- The overlay reads the raw JSON, not the replay's cleaned data, so a phantom pass the replay dropped can still count in the panel.

**Choosing your own match files** (UnityStandaloneFileBrowser):
- **Load match…** (bottom bar, start panel, or `L`) opens a native file dialog. Pick `match_events.json`: it contains everything in `match.json` plus ball height, possession, passes and ball flights, so it's the only file the replay needs.
- If you pick `match.json` by mistake, the replay looks for `match_events.json` in the same folder and uses that; without it, it plays the tracking only (no ball in the air, no kick animations).
- You can load a different match at any time. The loaded file is shown in the bottom bar, and the dialog reopens in the last folder used.
- The picked file lasts until you stop Play; settings (Flip Y, sponsor boards, etc.) come from the open scene.
- **Load Match On Start** on the `Replay` object (off by default) plays the scene's **Match File** straight away instead, e.g. for a demo. Files for that go in `Assets/StreamingAssets/matches/`.

**Scenes** (same scripts, different clips; settings like Flip Y and the sponsor boards are saved per scene):
Each scene's **Match File** (used when Load Match On Start is ticked):

| Scene | Data | Clip |
|---|---|---|
| `Assets/Scenes/Replay.unity` | `matches/match_events_v5.json` | 60 s, play in the west half |
| `Assets/Scenes/Replay_Match2.unity` | `matches/match_events_v6.json` | 30 s, play around the east penalty area |

## What's been built

### Replay
- Loads the CV team's JSON (any frame rate). Positions are in meters, and center or corner origin is detected (or read from `pitch.origin`).
- **Flip Y** mirrors the pitch top to bottom. It's on for the current data, because the CV y axis points the other way.
- Kits: currently **white (team 0) vs black (team 1)** (Home / Away Color). Tick **Use Kit Colors** to use the file's `teams` colors instead. Players are labeled with their tracking ID and smoothly interpolated. **Referees** are drawn in yellow ("REF").
- **Goalkeepers:** solid when tracked. When off camera they're semi-transparent ("GK (est.)"), stay near where they were last or next seen, and are capped at walking speed. A keeper the camera never sees is estimated from the ball's position.
- **Keeper validation:** each team's goal is decided from the keeper with the most data. A track labeled goalkeeper that averages more than 25 m from its own goal is treated as a misclassified outfield player.

### Off-camera players
Broadcast footage only shows part of the pitch, so players drop in and out of the data. The replay keeps them on the pitch:
- **Tracks are joined into people:** a same-team track that starts within 8 s of another ending, close enough to have run there unseen, is treated as the same person. On the current clip, 60 tracks become 42 people.
- **Estimated players are drawn see-through with no number.** Between two sightings they glide from the last-seen to the next-seen position. Before the first or after the last sighting they stay near that spot and move with their team's overall shape, for up to 8 s.
- At most **10 outfield players per team** are shown. When there are more estimates, the ones closest in time to a real sighting win, and changes fade instead of popping.
- Tunable under **Off-camera players** on the `Replay` object (see-through amount, hold time, team follow, join gap and speed). This is a first version and still needs tuning.

### Animated players
- Players are a **textured Mixamo human footballer** (`Assets/Characters/Ch38/`, set as **Player Model**). The shirt and socks are tinted with the team kit color and the shorts with the contrast color; skin, hair and boots keep their textures. Keepers and the referee get their own colors, and estimated off-camera players use a see-through version. The **Y Bot** mannequin (`Assets/Characters/SoccerGamePack/`) still works: set it back as Player Model. All animations work on either, since they share the humanoid rig.
- **Locomotion blend tree** (`Assets/Characters/PlayerLocomotion.controller`): a 2D freeform-directional tree on body-relative velocity. `offensive idle` is in the center, **Walking** at 1.68 m/s, and the **8-direction jog set** at about 2.4 m/s, each placed at its measured speed and direction. Goalkeepers use the same tree with `goalkeeper idle` (`KeeperLocomotion.overrideController`).
- Animation speed follows ground speed (above jog speed the jog plays up to 1.6× faster) and replay time (it freezes on pause and scales with playback speed). Players' steps are offset so they don't move in sync. Root motion is off, so players stay exactly on their tracked positions.
- **Soccer Replay → Setup Animated Players** rebuilds everything (clip import settings, blend tree, keeper variant, wiring). Drop `Running` / `Fast Run` / `Sprint` clips into the pack folder and re-run it to add faster gaits.
- Untick **Use Animated Players** to go back to capsules.

### Actions from match events (kicks, receives, keeper)
- An **action schedule** is built from the events. A `pass` plays a kick (standing kick, running strike, or long kick for aerial or 30 m+ passes; keepers use pass or drop kick). A `touch` plays a receive (keepers catch). A ball arriving at 1.2 m or higher plays a header.
- Actions run on an **Actions** animator layer, **driven by replay time** (they stay correct when pausing, scrubbing or changing speed), and blend in and out over the locomotion.
- **Contact timing:** each clip's contact moment (when the foot, head or hands move fastest) and limb position are measured when the replay starts, on a hidden copy of the player. The action starts early enough to connect at the event time.
- **Contact alignment:** around contact the player is shifted slightly (capped at 1.5 m) so the striking foot lands on the ball, using the **mirrored clip** when the ball is on the other foot's side. On the 60 s clip, foot-to-ball distance at contact went from 0.71 m on average to 0.12 m (median 0.03 m).
- Players turn toward the pass target when kicking and toward the incoming ball when receiving.
- Tunable under **Actions** on the `Replay` object (blend times, running-kick speed, long-pass length, header height, contact alignment, mirroring).

### Players' body motion
- Velocity is measured over a window centered on the current moment, then smoothed.
- **Players watch the ball:** below 3.5 m/s they face it (the 8-direction jogs show backpedals and sidesteps); when running they face where they're going, turned about 25% toward the ball. On clip 1, 95% of players face within 60° of the ball.
- **Goalkeepers always face the ball** (shuffling and backpedaling rather than turning away), except during their own catches and kicks.
- Capsule bodies (fallback) have a chest stripe and shorts band, and lean and bob with speed. Animated players get a lighter lean on top of their clips. All of this follows pause, scrubbing and playback speed.

### Ball and events
- Ball (0.4 m across, adjustable as **Ball Diameter**) with height and a ground shadow, and a **possession ring** under the player on the ball (`owner`).
- **Fitted ball flights** (`ball_flights`): during a flight the ball follows the smooth arc (a straight ground track plus a parabola for height) instead of noisy per-frame points, so high balls never jitter. It starts on the grass and ends at `end_height`: 0 for a landing, above 0 for a header or volley.
- **Ball cleanup (no teleporting):** false ball detections are removed by keeping the longest chain of positions a real ball could travel between (up to 40 m/s across gaps). Gaps are filled from the player on the ball when that fits, "passes" that only existed because of a false detection are dropped, the ball stays visible through gaps of up to 5 s, and it rests in the net after a goal. Clip 2 had jumps of up to 95 m; now neither clip has any movement over 40 m/s.
- **Smoother lobs:** each arc is joined onto the real ball path at takeoff and landing (no snap), aerial passes without a fitted flight get an arc, and arc height is capped at twice what gravity allows for the flight time.
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
- **Rotating sponsor boards** (LED style): every 6 s each board flips to the next sponsor (**Sheffield Thursday**, **CodeJam 16**, **IBM**, **Tower Research Capital**, **1Password**, **Brim**, **Energie3R**, **Lovable**), with neighboring boards showing different sponsors, all on a uniform black background. The IBM logo's white background was made transparent; the five newer logos were rendered from their SVGs to transparent PNGs (Brim recolored white to show on black). Sponsors are a list under **Stadium Builder → Board Ads** (logo, text, size, optional circle crop). Logos are in `Assets/Replay/Textures/`; the CodeJam logo was rendered from its SVG.
- **Goals** with a see-through 12 cm net on the back, roof and sides, rear posts and ground bars. From the broadcast camera the net fades to a light haze.
- Sunny-afternoon lighting with the stand shadow across the pitch, procedural sky, haze, post-processing (ACES, bloom, vignette) and MSAA.
- The broadcast camera sits on a gantry at the front of the main stand. It follows a smoothed play position (0.6 s smoothing, max 18 m/s, falling back to the players when the ball is lost), so a stray ball detection can't swing it to the other end.
- **Shadow Strength** (0.4) controls how dark the stand's shadow across the pitch is.

Almost everything is tunable on the `Replay` object's Inspector: Data cleanup, Body motion, Goalkeepers, Stadium Builder (colors, crowd, boards, sun, fog), and Use Fitted Flights.

## Code layout
| Path | What |
|---|---|
| `Assets/Replay/MatchData.cs` | JSON format classes, loading, coordinate handling, tracking cleanup, tracks, ball flights |
| `Assets/Replay/ReplayController.cs` | Playback, cameras, players, keepers, ball, possession, pass arcs, UI |
| `Assets/Replay/PlayerBody.cs` | Player body: animated Y Bot or capsule, facing, lean, blend-tree driving, action playback |
| `Assets/Replay/MatchActions.cs` | Action clips, the event → action scheduler, contact alignment and mirroring |
| `Assets/Replay/Editor/AnimatedPlayersSetup.cs` | **Soccer Replay → Setup Animated Players**: clip import, blend tree, keeper override |
| `Assets/Characters/` | Soccer Game Pack (Y Bot, X Bot, Mixamo clips) and the locomotion controllers |
| `Assets/Replay/PitchBuilder.cs` | Pitch, markings, goals and nets |
| `Assets/Scripts/PitchOverlay/` | Stats overlay from the analysis side (scoreboard bar, STATS panel, formation lines, measured shapes). Changes, marked `[REPLAY]`: a same-clip check and the live shapes hook in `MatchOverlay.cs`, `view_polygon` in `MatchModels.cs`, new `LiveShapes.cs` |
| `Assets/Replay/PlayerMotionDriver.cs` | Adapter: gives the overlay scripts the replay's clock and on-screen players |
| `Assets/Replay/UI/` | UI Toolkit panel settings and theme for the overlay |
| `Assets/Replay/Editor/BuildShaderVariants.cs` | Keeps code-created shader variants in standalone builds (runs before every build) |
| `Assets/Replay/StadiumBuilder.cs` | Stands, roof, crowd, ad boards, lighting and post-processing |
| `Assets/StandaloneFileBrowser/` | [UnityStandaloneFileBrowser](https://github.com/gkngkc/UnityStandaloneFileBrowser) 1.2 (MIT): native open-file dialog on Windows, Mac and Linux (samples left out) |
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

## Known issues / still to tweak
These are minor polish items. Everything works, but they're worth tuning:
- **Standing-kick timing:** for the short `kick soccerball` clip, "fastest foot" lands in the follow-through (0.52 s of 0.57 s), so standing kicks connect slightly late in the motion. Fix: limit that clip's contact search to its first part.
- **Contact alignment** is good on average (median 0.03 m) but a few kicks are still about 0.6 m off, and actions by estimated (off-camera) players can't be aligned as precisely.
- **Off-camera player estimates** (first version) need tuning: hold time, team-follow amount, how readily tracks are joined (a wrong join can make an estimate glide somewhere odd), and how see-through estimates are.
- **Faster gaits:** there are no running or sprint clips yet, so the jog plays up to 1.6× faster above about 2.4 m/s. Add `Running` / `Fast Run` and re-run **Setup Animated Players**.
- **One character for everyone:** all players share one face and hair; adding a second Mixamo character would add variety. The white kit reads light grey in the stand's shadow.
- **Stats overlay:** shape chips need ~3 s of a phase with the whole outfield on camera, so close-ups show `–`; with little footage the method gives odd shapes (e.g. `6-2-2` when pinned back, `1-5-4` attacking), shown as low confidence (faint chip). It's the shape the team holds, not the team-sheet formation. Score needs `SetScore`; goal threat is a placeholder. With only part of a team on camera the lines can zigzag.
- **Goal nets in the .exe** were reported looking different from the Editor; not yet diagnosed.
- **Camera smoothing** values (0.6 s, 18 m/s) haven't been tuned by eye on every clip yet.
- **Headers** are wired up but haven't been seen yet, since no clip so far has a flight ending above 1.2 m.
- **Data side:** keepers sometimes switch IDs (#22/#23 in match 2), camera calibration coasts most frames, and Flip Y is still needed. These are noted in "Notes for the CV side".

## Next up
- The tweaks above, analysis files for our clips, and player ratings from the analysis model.
- A backup demo video.
