using System;
using System.Collections.Generic;
using UnityEngine;

namespace SoccerReplay
{
    // ---- Raw file format (matches the CV team's JSON) ----

    [Serializable]
    public class PitchInfo
    {
        public float length = 105f;
        public float width = 68f;
        public string origin;  // optional: "center" or "corner"
    }

    [Serializable]
    public class PlayerSample
    {
        public int id;
        public float x;
        public float y;
        public int team;     // 0 = home, 1 = away
        public string role;  // "player" for now
    }

    [Serializable]
    public class FrameData
    {
        public float t;                    // seconds since clip start
        public bool view;
        public List<PlayerSample> players;
        public float[] ball;               // [x, y] or [x, y, z] (z = height in meters), null when not detected
        public int owner = -1;             // player ID in possession, -1 when nobody
    }

    [Serializable]
    public class MatchEvent
    {
        public string type;                // "touch" | "pass" | ...
        // touch
        public float t;
        public int player;
        public float x, y, duration;
        // pass
        public float t_start, t_end;
        public int from, to;
        public float[] start, end;
        public float length, max_height;
        public bool completed, aerial;
        public int team;

        public float StartTime => type == "pass" ? t_start : t;
        public float EndTime => type == "pass" ? t_end : t + duration;
    }

    [Serializable]
    public class MatchFile
    {
        public PitchInfo pitch;
        public float fps;
        public List<FrameData> frames;
        public List<MatchEvent> events;
    }

    public enum OriginMode
    {
        Auto,    // detect from the data
        Corner,  // (0,0) is a pitch corner, x in [0, length], y in [0, width]
        Center   // (0,0) is the center spot
    }

    // ---- Processed data used by the replay ----

    /// <summary>Positions of one tracked object (a player or the ball) over time, in centered pitch meters.</summary>
    public class Track
    {
        public int id;
        public int team;
        public bool goalkeeper;
        public bool referee;
        public readonly List<float> times = new List<float>();
        public readonly List<Vector2> positions = new List<Vector2>();
        public readonly List<float> heights = new List<float>(); // optional, used by the ball

        public int Count => times.Count;

        /// <summary>Height at time t (0 if this track has no heights).</summary>
        public float HeightAt(float t)
        {
            if (heights.Count != Count || Count == 0) return 0f;
            if (t <= Start) return heights[0];
            if (t >= End) return heights[Count - 1];
            int i = IndexAtOrBefore(t);
            if (i >= Count - 1) return heights[Count - 1];
            return Mathf.Lerp(heights[i], heights[i + 1], (t - times[i]) / (times[i + 1] - times[i]));
        }
        public float Start => times[0];
        public float End => times[times.Count - 1];

        public void Add(float t, Vector2 p)
        {
            times.Add(t);
            positions.Add(p);
        }

        /// <summary>Triangular-weighted smoothing over ±window seconds, independent of frame rate.</summary>
        public void SmoothByTime(float window)
        {
            if (window <= 0f || Count < 3) return;
            var smoothed = new Vector2[Count];
            int lo = 0;
            for (int i = 0; i < Count; i++)
            {
                while (times[lo] < times[i] - window) lo++;
                Vector2 sum = Vector2.zero;
                float wsum = 0f;
                for (int k = lo; k < Count && times[k] <= times[i] + window; k++)
                {
                    float w = 1f - Mathf.Abs(times[k] - times[i]) / (window * 1.0001f);
                    sum += positions[k] * w;
                    wsum += w;
                }
                smoothed[i] = wsum > 0f ? sum / wsum : positions[i];
            }
            for (int i = 0; i < Count; i++) positions[i] = smoothed[i];
        }

        /// <summary>Drops single samples that jump away and straight back (detection glitches). Returns how many were removed.</summary>
        public int RemoveSpikes(float maxSpeed)
        {
            if (Count < 3) return 0;
            var keep = new bool[Count];
            for (int i = 0; i < Count; i++) keep[i] = true;
            for (int i = 1; i < Count - 1; i++)
            {
                float dt1 = times[i] - times[i - 1], dt2 = times[i + 1] - times[i];
                if (dt1 <= 0f || dt2 <= 0f || dt1 > 0.2f || dt2 > 0.2f) continue;
                float v1 = Vector2.Distance(positions[i], positions[i - 1]) / dt1;
                float v2 = Vector2.Distance(positions[i + 1], positions[i]) / dt2;
                float vSkip = Vector2.Distance(positions[i + 1], positions[i - 1]) / (dt1 + dt2);
                if (v1 > maxSpeed && v2 > maxSpeed && vSkip < maxSpeed) keep[i] = false;
            }
            int removed = 0;
            for (int i = Count - 1; i >= 0; i--)
            {
                if (keep[i]) continue;
                times.RemoveAt(i); positions.RemoveAt(i);
                if (heights.Count > i) heights.RemoveAt(i);
                removed++;
            }
            return removed;
        }

        /// <summary>
        /// Splits the track wherever it jumps faster than a person can run (an ID swap) or has a long gap,
        /// so the replay never draws someone zooming across the pitch.
        /// </summary>
        public List<Track> SplitOnJumps(float maxSpeed, float maxGap)
        {
            var parts = new List<Track>();
            var cur = new Track { id = id, team = team, goalkeeper = goalkeeper, referee = referee };
            for (int i = 0; i < Count; i++)
            {
                if (cur.Count > 0)
                {
                    float dt = times[i] - cur.End;
                    // A swap is a move further than a sprinter could cover in that time, plus ~1 m of
                    // detection jitter (at 30 fps normal jitter alone can look like 10+ m/s)
                    float dist = Vector2.Distance(positions[i], cur.positions[cur.Count - 1]);
                    bool jump = dist > maxSpeed * Mathf.Max(0f, dt) + 1.2f;
                    if (dt > maxGap || jump)
                    {
                        parts.Add(cur);
                        cur = new Track { id = id, team = team, goalkeeper = goalkeeper, referee = referee };
                    }
                }
                cur.Add(times[i], positions[i]);
            }
            if (cur.Count > 0) parts.Add(cur);
            return parts;
        }

        /// <summary>Moving-average smoothing to remove detection jitter. Only averages samples close in time.</summary>
        public void Smooth(int radius, float maxDt)
        {
            if (radius <= 0 || Count < 3) return;
            var smoothed = new Vector2[Count];
            for (int i = 0; i < Count; i++)
            {
                Vector2 sum = Vector2.zero;
                int n = 0;
                for (int k = Mathf.Max(0, i - radius); k <= Mathf.Min(Count - 1, i + radius); k++)
                {
                    if (Mathf.Abs(times[k] - times[i]) > maxDt) continue;
                    sum += positions[k];
                    n++;
                }
                smoothed[i] = sum / n;
            }
            for (int i = 0; i < Count; i++) positions[i] = smoothed[i];
        }

        /// <summary>
        /// Position at time t. Interpolates across gaps up to maxGap seconds.
        /// Returns false when the object isn't tracked at that moment.
        /// </summary>
        /// <summary>Last index whose time is &lt;= t (assumes Start &lt;= t).</summary>
        int IndexAtOrBefore(float t)
        {
            int lo = 0, hi = Count - 1;
            while (lo < hi)
            {
                int mid = (lo + hi + 1) / 2;
                if (times[mid] <= t) lo = mid; else hi = mid - 1;
            }
            return lo;
        }

        /// <summary>Position at time t, interpolating across any gap and holding the first/last sample outside the track.</summary>
        public Vector2 SampleClamped(float t)
        {
            if (t <= Start) return positions[0];
            if (t >= End) return positions[Count - 1];
            int i = IndexAtOrBefore(t);
            if (i >= Count - 1) return positions[Count - 1];
            float t0 = times[i], t1 = times[i + 1];
            return Vector2.Lerp(positions[i], positions[i + 1], (t - t0) / (t1 - t0));
        }

        /// <summary>Seconds between t and the closest real sample (0 when t is exactly on a sample).</summary>
        public float TimeToNearestSample(float t)
        {
            if (t <= Start) return Start - t;
            if (t >= End) return t - End;
            int i = IndexAtOrBefore(t);
            if (i >= Count - 1) return t - times[i];
            return Mathf.Min(t - times[i], times[i + 1] - t);
        }

        /// <summary>
        /// Velocity (m/s) measured across a window centered on t. Centered rather than backward-looking,
        /// so motion doesn't lag the data. Near the ends of the track the window shrinks to stay inside it.
        /// </summary>
        public Vector2 VelocityAt(float t, float window)
        {
            if (Count < 2) return Vector2.zero;
            float h = window * 0.5f;
            float a = Mathf.Max(Start, t - h), b = Mathf.Min(End, t + h);
            if (b - a < 1e-3f) return Vector2.zero;
            return (SampleClamped(b) - SampleClamped(a)) / (b - a);
        }

        public float AverageX()
        {
            float s = 0f;
            foreach (var p in positions) s += p.x;
            return Count > 0 ? s / Count : 0f;
        }

        public bool TrySample(float t, float maxGap, out Vector2 p)
        {
            p = default;
            if (Count == 0 || t < Start || t > End) return false;

            int i = IndexAtOrBefore(t);
            if (i == Count - 1)
            {
                p = positions[i];
                return true;
            }

            float t0 = times[i], t1 = times[i + 1];
            if (t1 - t0 > maxGap)
            {
                // Too long a gap to interpolate: hold the nearest sample briefly, otherwise hide.
                const float hold = 0.15f;
                if (t - t0 <= hold) { p = positions[i]; return true; }
                if (t1 - t <= hold) { p = positions[i + 1]; return true; }
                return false;
            }

            p = Vector2.Lerp(positions[i], positions[i + 1], (t - t0) / (t1 - t0));
            return true;
        }
    }

    public class MatchReplayData
    {
        public float length;
        public float width;
        public float fps;
        public float startTime;
        public float endTime;
        public bool cornerOrigin;
        public int rawIdCount;
        public int droppedShortTracks;
        public readonly List<Track> players = new List<Track>();
        public Track ball;
        public bool ballHasHeight;
        public List<MatchEvent> events = new List<MatchEvent>();
        float[] frameTimes;
        int[] frameOwners;

        public float Duration => endTime - startTime;

        /// <summary>Player ID in possession at time t, or -1.</summary>
        public int OwnerAt(float t)
        {
            if (frameTimes == null || frameTimes.Length == 0) return -1;
            int i = Array.BinarySearch(frameTimes, t);
            if (i < 0) i = Mathf.Max(0, ~i - 1);
            return frameOwners[Mathf.Min(i, frameOwners.Length - 1)];
        }

        public int cameraShiftsCorrected;
        public int spikesRemoved;
        public int idSwapsSplit;

        public class CleanOptions
        {
            public float smoothWindow = 0.2f;       // seconds either side
            public float maxPlayerSpeed = 11f;      // m/s; faster movement is treated as an error
            public bool correctCameraShifts = true; // cancel jumps where many players shift together
            public float shiftRecovery = 2f;        // seconds over which a cancelled shift eases back
            public float splitGap = 1.5f;           // seconds; longer gaps start a new track
        }

        public static MatchReplayData FromJson(string json, OriginMode origin, bool flipY, int minTrackSamples, int smoothRadius, CleanOptions clean = null)
        {
            clean ??= new CleanOptions();
            // JsonUtility can't read null into an int, so map "owner": null to -1
            json = System.Text.RegularExpressions.Regex.Replace(json, "\"owner\"\\s*:\\s*null", "\"owner\": -1");
            // Referees (and anything unassigned) come with "team": null
            json = System.Text.RegularExpressions.Regex.Replace(json, "\"team\"\\s*:\\s*null", "\"team\": -1");
            var file = JsonUtility.FromJson<MatchFile>(json);
            if (file == null || file.frames == null || file.frames.Count == 0)
                throw new Exception("Match file has no frames.");

            var data = new MatchReplayData
            {
                length = file.pitch != null && file.pitch.length > 0 ? file.pitch.length : 105f,
                width = file.pitch != null && file.pitch.width > 0 ? file.pitch.width : 68f,
                fps = file.fps,
                startTime = file.frames[0].t,
                endTime = file.frames[file.frames.Count - 1].t,
            };

            // Decide where (0,0) is. Corner-origin data has x values well past half the pitch length.
            string declared = file.pitch?.origin?.ToLowerInvariant();
            if (origin == OriginMode.Auto && (declared == "center" || declared == "corner"))
            {
                data.cornerOrigin = declared == "corner";
            }
            else if (origin == OriginMode.Auto)
            {
                float maxX = float.MinValue;
                foreach (var f in file.frames)
                {
                    if (f.players != null) foreach (var p in f.players) maxX = Mathf.Max(maxX, p.x);
                    if (f.ball != null && f.ball.Length >= 2) maxX = Mathf.Max(maxX, f.ball[0]);
                }
                data.cornerOrigin = maxX > data.length / 2f + 3f;
            }
            else
            {
                data.cornerOrigin = origin == OriginMode.Corner;
            }

            Vector2 ToCentered(float x, float y)
            {
                if (data.cornerOrigin) { x -= data.length / 2f; y -= data.width / 2f; }
                if (flipY) y = -y;
                return new Vector2(x, y);
            }

            var tracks = new Dictionary<int, Track>();
            var teamVotes = new Dictionary<int, int[]>();
            var gkVotes = new Dictionary<int, int>();
            var refVotes = new Dictionary<int, int>();
            data.ball = new Track { id = -1, team = -1 };

            // Camera calibration jumps move everyone at once. Detect frames where the typical (median)
            // player moves faster than people can collectively run, cancel that shift, and let the
            // correction ease out over a couple of seconds so it becomes a gentle slide.
            var offsets = new Vector2[file.frames.Count];
            if (clean.correctCameraShifts)
            {
                Dictionary<int, Vector2> prev = null;
                float prevT = 0f;
                Vector2 off = Vector2.zero;
                var dxs = new List<float>();
                var dys = new List<float>();
                for (int fi = 0; fi < file.frames.Count; fi++)
                {
                    var f = file.frames[fi];
                    var cur = new Dictionary<int, Vector2>();
                    if (f.players != null) foreach (var p in f.players) cur[p.id] = ToCentered(p.x, p.y);
                    if (prev != null)
                    {
                        float dt = f.t - prevT;
                        off *= Mathf.Exp(-Mathf.Max(0f, dt) / Mathf.Max(0.1f, clean.shiftRecovery));
                        dxs.Clear(); dys.Clear();
                        foreach (var kv in cur)
                            if (prev.TryGetValue(kv.Key, out var pp)) { dxs.Add(kv.Value.x - pp.x); dys.Add(kv.Value.y - pp.y); }
                        if (dt > 0f && dt < 0.15f && dxs.Count >= 5)
                        {
                            dxs.Sort(); dys.Sort();
                            var med = new Vector2(dxs[dxs.Count / 2], dys[dys.Count / 2]);
                            if (med.magnitude / dt > clean.maxPlayerSpeed * 0.6f) { off -= med; data.cameraShiftsCorrected++; }
                        }
                    }
                    offsets[fi] = off;
                    prev = cur;
                    prevT = f.t;
                }
            }

            for (int fi = 0; fi < file.frames.Count; fi++)
            {
                var f = file.frames[fi];
                if (f.players != null)
                {
                    foreach (var p in f.players)
                    {
                        if (!tracks.TryGetValue(p.id, out var tr))
                        {
                            tr = new Track { id = p.id };
                            tracks[p.id] = tr;
                            teamVotes[p.id] = new int[2];
                        }
                        tr.Add(f.t, ToCentered(p.x, p.y) + offsets[fi]);
                        if (p.team == 0 || p.team == 1) teamVotes[p.id][p.team]++;
                        if (p.role == "goalkeeper") gkVotes[p.id] = (gkVotes.TryGetValue(p.id, out var g) ? g : 0) + 1;
                        if (p.role == "referee") refVotes[p.id] = (refVotes.TryGetValue(p.id, out var rv) ? rv : 0) + 1;
                    }
                }
                if (f.ball != null && f.ball.Length >= 2)
                {
                    data.ball.Add(f.t, ToCentered(f.ball[0], f.ball[1]) + offsets[fi]);
                    data.ball.heights.Add(f.ball.Length >= 3 ? Mathf.Max(0f, f.ball[2]) : 0f);
                    if (f.ball.Length >= 3) data.ballHasHeight = true;
                }
            }

            data.frameTimes = new float[file.frames.Count];
            data.frameOwners = new int[file.frames.Count];
            for (int i = 0; i < file.frames.Count; i++)
            {
                data.frameTimes[i] = file.frames[i].t;
                data.frameOwners[i] = file.frames[i].owner;
            }

            if (file.events != null)
            {
                foreach (var e in file.events)
                {
                    // Event positions use the same coordinates as the tracking
                    if (e.type == "touch")
                    {
                        var p = ToCentered(e.x, e.y); e.x = p.x; e.y = p.y;
                    }
                    if (e.start != null && e.start.Length >= 2)
                    {
                        var p = ToCentered(e.start[0], e.start[1]); e.start[0] = p.x; e.start[1] = p.y;
                    }
                    if (e.end != null && e.end.Length >= 2)
                    {
                        var p = ToCentered(e.end[0], e.end[1]); e.end[0] = p.x; e.end[1] = p.y;
                    }
                    data.events.Add(e);
                }
                data.events.Sort((a, b) => a.StartTime.CompareTo(b.StartTime));
            }

            data.rawIdCount = tracks.Count;
            foreach (var kv in tracks)
            {
                var tr = kv.Value;
                // IDs that only exist for a moment are almost always tracker noise.
                if (tr.Count < minTrackSamples) { data.droppedShortTracks++; continue; }
                var votes = teamVotes[kv.Key];
                tr.team = votes[1] > votes[0] ? 1 : 0;
                tr.goalkeeper = gkVotes.TryGetValue(kv.Key, out var gk) && gk * 2 > tr.Count;
                tr.referee = refVotes.TryGetValue(kv.Key, out var rf) && rf * 2 > tr.Count;
                if (tr.referee) { tr.team = -1; tr.goalkeeper = false; }

                // Clean: drop one-frame glitches, then split where an ID jumps to a different person
                data.spikesRemoved += tr.RemoveSpikes(clean.maxPlayerSpeed);
                var parts = tr.SplitOnJumps(clean.maxPlayerSpeed, clean.splitGap);
                data.idSwapsSplit += parts.Count - 1;
                foreach (var part in parts)
                {
                    if (part.Count < minTrackSamples) { data.droppedShortTracks++; continue; }
                    part.SmoothByTime(clean.smoothWindow);
                    data.players.Add(part);
                }
            }
            data.spikesRemoved += data.ball.RemoveSpikes(40f); // the ball can genuinely move fast
            data.ball.SmoothByTime(Mathf.Min(0.1f, clean.smoothWindow));
            return data;
        }
    }
}
