# Hackathon-Codejam
codejam repo 

Soccer match video → computer vision tracking → **3D replay in Unity** → formation and player-rating analysis.

## Repo layout
| Path | What |
|---|---|
| `Assets/Replay/` | Unity replay code: data loader, pitch builder, replay controller, editor menu |
| `Assets/StreamingAssets/matches/` | Tracking files the replay loads (`match_events.json` is the current one) |
| `Assets/Scenes/Replay.unity` | The replay scene |
| `docs/plan.md` | Weekend plan and data format |

The rest of `Assets/` is the Unity "Get Started" template. CV and analysis code can live in their own top-level folders (e.g. `cv/`, `analysis/`); Unity ignores anything outside `Assets/`.

## Running the replay
1. Install **Unity 6.6 (6000.6.10f1)** via Unity Hub and open this folder as a project.
2. Open `Assets/Scenes/Replay.unity` (or use the menu **Soccer Replay → Create Replay Scene and Play**).
3. Press **Play**.

**Controls:** Space play/pause · ←/→ skip 1 s · `[` `]` speed · `1` `2` `3` broadcast / top-down / follow-ball camera · `R` restart

To load a different file, select the `Replay` object and change **Match File** in the Inspector.

## Data format (CV output)
```json
{
  "pitch": { "length": 105, "width": 68, "origin": "center" },
  "fps": 14.98,
  "frames": [
    { "t": 0.0,
      "players": [ { "id": 7, "team": 0, "role": "player", "x": -12.5, "y": 4.2 } ],
      "ball": [x, y, z],
      "owner": 11 }
  ],
  "events": [
    { "type": "pass", "t_start": 4.27, "t_end": 8.14, "from": 14, "to": 11, "team": 0,
      "start": [x, y], "end": [x, y], "length": 16.5, "completed": true, "aerial": true, "max_height": 9.6 }
  ]
}
```
- Positions are in **meters**, with (0, 0) at the center spot. Corner-origin files are detected automatically.
- `team`: 0 or 1. `role`: `player` or `goalkeeper`. `ball` z is the height in meters and is optional. `ball`/`owner` may be `null`.

## What the replay does with the data
- Smooths jitter, hides tracker-noise IDs (fewer than 8 frames), and interpolates short gaps.
- Goalkeepers: when tracked they're drawn solid. When off camera they're drawn semi-transparent ("GK (est.)") near where they were last seen, and a never-seen keeper is estimated from the ball position.
- Ball height with a ground shadow, a possession ring under the ball owner, pass arcs, and an event feed.
