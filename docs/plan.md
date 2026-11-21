# Soccer Match Replay – Code Jam Plan

**Deadline:** Sunday 12:00 PM · **CV data expected:** Saturday morning

## What we're building
Match video → **CV model** (teammate) → tracking data → **Unity 3D replay** (Ariyan) → **analysis** (teammates: formation detection like 4-3-3 / false 9, FotMob-style player ratings), shown inside the Unity replay.

**Demo goal:** play a real clip side by side with its 3D replay, with the formation and player ratings overlaid. Everything working from start to finish matters more than looks.

## Non-goals (this weekend)
- Photoreal / fully animated characters (only if time is left Sunday morning)
- Doing the analysis inside Unity: Unity only *displays* the data team's results

## Data format: CV → Unity (and analysis)
One JSON file per clip. Positions are in **meters on the pitch**, with (0,0) at the center spot. x runs along the length (-52.5 to 52.5) and y across the width (-34 to 34).

```json
{
  "fps": 25,
  "pitch": { "length": 105, "width": 68 },
  "frames": [
    {
      "frame": 0,
      "ball": { "x": 0.0, "y": 0.0 },
      "players": [
        { "id": 7, "team": "home", "x": -12.5, "y": 4.2 },
        { "id": 3, "team": "away", "x": 10.1, "y": -8.0 }
      ]
    }
  ]
}
```

- `team`: `"home"` | `"away"` | `"ref"`
- `id` stays the same for the same player across frames (tracking ID; a jersey number if known)
- A player or the ball can be missing from a frame; Unity fills short gaps
- The CV side converts video pixels to pitch meters (homography). Fallback if that isn't ready: pixel coords plus the 4 pitch corners marked in the video.

## Data format: analysis → Unity (draft, to confirm with the data team)
```json
{
  "formations": [ { "team": "home", "fromFrame": 0, "toFrame": 750, "shape": "4-3-3" } ],
  "ratings":    [ { "id": 7, "team": "home", "rating": 7.8 } ]
}
```

## Timeline
| When | Goal |
|---|---|
| Fri night | Data format agreed · fake data generator · Unity replay (pitch, 22 players, ball, play/pause/scrub) |
| Sat morning | Load real CV data · smooth out jitter, ID switches and missing ball |
| Sat afternoon/evening | Formation lines + player rating cards in Unity |
| Sat night / Sun morning | Camera views (broadcast / top-down / follow ball), team colors and numbers, desktop build |
| Sun 10:00 AM | **Stop adding features** · record a backup demo video |

## Unity setup (my calls)
- Unity 6.6 project `CodeJamProject` (URP). Replay code goes in `Assets/Replay/`, data files in `Assets/StreamingAssets/matches/`
- Players are simple capsules in team colors with number labels; the ball is a sphere
- Positions are smoothed between frames so movement looks fluid at any frame rate
