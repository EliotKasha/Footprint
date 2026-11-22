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
        public bool completed, aerial, estimated;
        public int team;

        /// <summary>Events that travel from start to end over t_start..t_end (passes, lofted balls).</summary>
        public bool IsFlight => type == "pass" || type == "aerial_ball";
        public float StartTime => IsFlight ? t_start : t;
        public float EndTime => IsFlight ? t_end : t + duration;
    }

    /// <summary>A fitted ball flight from the CV pipeline: takeoff -> landing along a ballistic arc.</summary>
    [Serializable]
    public class BallFlight
    {
        public float t_takeoff, t_land, max_height;
        public float end_height;      // height when the flight ends (0 = lands; >0 = header/volley/catch)
        public float[] start, land;   // ground positions [x, y]
        public string method, kind;   // e.g. "3d" / "estimated", "free ball" / "pass"
        public float fit_px;          // fit error in pixels (3d flights)
        public float speed_ms;        // launch speed
        public int from = -1, team = -1;

        public float Duration => t_land - t_takeoff;
        public bool Contains(float t) => t >= t_takeoff && t <= t_land && Duration > 0.01f;

        // Set by MatchReplayData.PrepareFlights: shifts that join the arc's ends onto the real ball path,
        // and the height actually drawn (capped to what gravity allows for the flight time)
        [NonSerialized] public Vector2 startShift, landShift;
        [NonSerialized] public float drawHeight = -1f;

        /// <summary>
        /// Position on the arc: straight-line ground track at constant speed, and a parabola in time
        /// for height (what gravity produces) that starts on the ground, peaks at max_height and
        /// ends at end_height (0 for a normal landing).
        /// </summary>
        public void Sample(float t, out Vector2 ground, out float height)
        {
            float u = Mathf.Clamp01((t - t_takeoff) / Duration);
            ground = Vector2.Lerp(new Vector2(start[0], start[1]) + startShift, new Vector2(land[0], land[1]) + landShift, u);

            // z(u) = a·u² + b·u with z(0)=0, z(1)=e and a peak of H:
            // a = e − 2H − 2·√(H·(H−e)), b = e − a   (reduces to 4H·u·(1−u) when e = 0)
            float H = Mathf.Max(drawHeight >= 0f ? drawHeight : max_height, 0f);
            float e = Mathf.Clamp(end_height, 0f, H);
            float a = e - 2f * H - 2f * Mathf.Sqrt(H * (H - e));
            float b = e - a;
            height = Mathf.Max(0f, a * u * u + b * u);
        }
    }

    [Serializable]
    public class MatchFile
    {
        public PitchInfo pitch;
        public float fps;
        public List<FrameData> frames;
        public List<MatchEvent> events;
        public List<BallFlight> ball_flights;
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
        public List<BallFlight> flights = new List<BallFlight>();
        /// <summary>Kit colors from the file's "teams" block (team 0, team 1), if present.</summary>
        public bool hasKitColors;
        public Color[] kitColors = { Color.red, Color.blue };
        float[] frameTimes;

        /// <summary>The fitted flight the ball is on at time t, or null.</summary>
        public BallFlight FlightAt(float t)
        {
            foreach (var f in flights) if (f.Contains(t)) return f;
            return null;
        }
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
            public float ballMaxSpeed = 40f;        // m/s; ball movement faster than this is a false detection (hard shots ~35)
            public float ballJumpMargin = 2f;       // m of detection slack added to the reachable distance
            public bool fillBallFromOwner = true;   // ball missing but someone owns it -> put it at their feet
            public bool dropImplausibleEvents = true; // passes/touches that don't match the cleaned ball path
            public float lobGravityCap = 2f;        // arcs no higher than this × what gravity allows for their duration
        }

        public int ballOutliersRemoved, ballSamplesFilled, eventsDropped, flightsAdded, flightsCapped;

        /// <summary>
        /// Makes lobs fluid: adds arcs for aerial passes / loose balls that have no fitted flight, joins every
        /// arc's ends onto the real ball path (no snap at takeoff or landing), and caps each arc's height at
        /// gravityCap × what gravity allows for its flight time (g·T²/8), so lobs can't shoot up and down
        /// unnaturally fast.
        /// </summary>
        void PrepareFlights(float gravityCap)
        {
            // Arcs for aerial events that have no fitted flight
            foreach (var e in events)
            {
                bool aerialEvent = (e.type == "pass" && e.aerial) || e.type == "aerial_ball";
                if (!aerialEvent || e.start == null || e.end == null || e.t_end - e.t_start < 0.2f || e.max_height < 0.3f) continue;
                bool covered = flights.Exists(f => f.t_takeoff < e.t_end && e.t_start < f.t_land);
                if (covered) continue;
                flights.Add(new BallFlight
                {
                    t_takeoff = e.t_start, t_land = e.t_end, max_height = e.max_height,
                    start = new[] { e.start[0], e.start[1] }, land = new[] { e.end[0], e.end[1] },
                    method = "event", kind = e.type == "pass" ? "aerial pass" : "loose ball",
                });
                flightsAdded++;
            }
            flights.Sort((a, b) => a.t_takeoff.CompareTo(b.t_takeoff));

            Vector2? BallNear(float t)
            {
                if (ball.Count == 0) return null;
                int idx = ball.times.BinarySearch(t);
                if (idx < 0) idx = ~idx;
                float near = Mathf.Min(idx > 0 ? t - ball.times[idx - 1] : float.MaxValue, idx < ball.Count ? ball.times[idx] - t : float.MaxValue);
                return near <= 0.3f ? ball.SampleClamped(t) : (Vector2?)null;
            }
            foreach (var f in flights)
            {
                // Join the arc onto where the ball really is at takeoff and landing (small shifts only)
                var s = BallNear(f.t_takeoff - 0.02f);
                var l = BallNear(f.t_land + 0.02f);
                // (fits can start several metres from where the ball really was; join them fully so the ball never jumps)
                if (s.HasValue) f.startShift = Vector2.ClampMagnitude(s.Value - new Vector2(f.start[0], f.start[1]), 12f);
                if (l.HasValue) f.landShift = Vector2.ClampMagnitude(l.Value - new Vector2(f.land[0], f.land[1]), 12f);

                float physical = 9.81f * f.Duration * f.Duration / 8f;   // peak of a ball in the air for that long
                float cap = Mathf.Max(physical * gravityCap, f.end_height);
                f.drawHeight = Mathf.Min(f.max_height, cap);
                if (f.drawHeight < f.max_height - 0.05f) flightsCapped++;
            }
        }

        /// <summary>
        /// Removes false ball detections: splits the ball track wherever it jumps further than a ball could
        /// travel (maxSpeed × gap + margin, so jumps across gaps count too), then keeps the chain of pieces
        /// with the most samples in which every hand-over between pieces is physically reachable. Short
        /// islands of detections elsewhere on the pitch (another object, a line marking) are dropped.
        /// </summary>
        void CleanBall(float maxSpeed, float margin)
        {
            var b = ball;
            int n = b.Count;
            if (n < 3) return;
            bool Reachable(int i, int j) // can the ball get from sample i to sample j?
            {
                float dt = Mathf.Abs(b.times[j] - b.times[i]);
                return Vector2.Distance(b.positions[i], b.positions[j]) <= maxSpeed * dt + margin;
            }

            var segs = new List<(int s, int e)>();
            int s0 = 0;
            for (int i = 1; i < n; i++)
                if (!Reachable(i - 1, i)) { segs.Add((s0, i - 1)); s0 = i; }
            segs.Add((s0, n - 1));

            // Best consistent chain of pieces (dynamic programming over pieces in time order). Each piece scores
            // its sample count; a hand-over between pieces that needs a fast ball across a gap costs points in
            // proportion to the speed above 15 m/s, so a stray detection that would need a near-impossible dash
            // loses to the explanation that fits (e.g. a shot continuing into the net).
            float HandOverCost(int i, int j)
            {
                float dt = Mathf.Max(1f / 60f, Mathf.Abs(b.times[j] - b.times[i]));
                float v = Mathf.Max(0f, Vector2.Distance(b.positions[i], b.positions[j]) - margin) / dt;
                return Mathf.Max(0f, v - 25f) * 0.3f;   // shots reach ~35 m/s, so only penalise beyond pass speeds
            }
            int S = segs.Count;
            var best = new float[S];
            var prev = new int[S];
            for (int i = 0; i < S; i++)
            {
                float w = segs[i].e - segs[i].s + 1;
                best[i] = w; prev[i] = -1;
                for (int j = 0; j < i; j++)
                {
                    if (!Reachable(segs[j].e, segs[i].s)) continue;
                    float score = best[j] + w - HandOverCost(segs[j].e, segs[i].s);
                    if (score > best[i]) { best[i] = score; prev[i] = j; }
                }
            }
            int last = 0;
            for (int i = 1; i < S; i++) if (best[i] > best[last]) last = i;
            var keep = new bool[S];
            for (int i = last; i >= 0; i = prev[i]) keep[i] = true;

            var times = new List<float>(); var pos = new List<Vector2>(); var hts = new List<float>();
            for (int k = 0; k < S; k++)
            {
                if (!keep[k]) continue;
                for (int i = segs[k].s; i <= segs[k].e; i++)
                {
                    times.Add(b.times[i]); pos.Add(b.positions[i]);
                    hts.Add(b.heights.Count == n ? b.heights[i] : 0f);
                }
            }
            ballOutliersRemoved += n - times.Count;
            b.times.Clear(); b.times.AddRange(times);
            b.positions.Clear(); b.positions.AddRange(pos);
            b.heights.Clear(); b.heights.AddRange(hts);
        }

        /// <summary>
        /// Where the ball is missing but a player owns it, place it just in front of that player's feet,
        /// provided that's reachable from the real ball samples before and after (so a wrongly assigned
        /// owner can't drag the ball somewhere impossible).
        /// </summary>
        void FillBallFromOwner(float maxSpeed, float margin)
        {
            if (frameTimes == null || ball.Count == 0) return;
            var addT = new List<float>(); var addP = new List<Vector2>();
            for (int f = 0; f < frameTimes.Length; f++)
            {
                int owner = frameOwners[f];
                if (owner < 0) continue;
                float t = frameTimes[f];
                // Real sample at (almost) this time? Then nothing to fill.
                int idx = ball.times.BinarySearch(t);
                if (idx < 0) idx = ~idx;
                float before = idx > 0 ? t - ball.times[idx - 1] : float.MaxValue;
                float after = idx < ball.Count ? ball.times[idx] - t : float.MaxValue;
                if (Mathf.Min(before, after) < 0.05f) continue;

                Track piece = null; Vector2 p = default;
                foreach (var tr in players)
                    if (tr.id == owner && tr.TrySample(t, 0.5f, out p)) { piece = tr; break; }
                if (piece == null) continue;
                var v = piece.VelocityAt(t, 0.4f);
                Vector2 c = p + (v.sqrMagnitude > 0.25f ? v.normalized * 0.5f : Vector2.zero);

                bool ok = true;
                if (idx > 0 && Vector2.Distance(c, ball.positions[idx - 1]) > maxSpeed * before + margin) ok = false;
                if (idx < ball.Count && Vector2.Distance(c, ball.positions[idx]) > maxSpeed * after + margin) ok = false;
                if (!ok) continue;
                addT.Add(t); addP.Add(c);
            }
            if (addT.Count == 0) return;

            // Merge into the ball track in time order (filled samples sit on the ground)
            var times = new List<float>(); var pos = new List<Vector2>(); var hts = new List<float>();
            int a = 0, r = 0;
            while (a < addT.Count || r < ball.Count)
            {
                bool takeReal = r < ball.Count && (a >= addT.Count || ball.times[r] <= addT[a]);
                if (takeReal) { times.Add(ball.times[r]); pos.Add(ball.positions[r]); hts.Add(ball.heights.Count == ball.Count ? ball.heights[r] : 0f); r++; }
                else { times.Add(addT[a]); pos.Add(addP[a]); hts.Add(0f); a++; }
            }
            ballSamplesFilled += addT.Count;
            ball.times.Clear(); ball.times.AddRange(times);
            ball.positions.Clear(); ball.positions.AddRange(pos);
            ball.heights.Clear(); ball.heights.AddRange(hts);
        }

        /// <summary>
        /// Drops passes and touches that don't fit the cleaned ball path (e.g. a "pass" to a false
        /// detection on the far touchline), so no kick or arc is shown towards a phantom ball.
        /// </summary>
        void DropImplausibleEvents(float maxSpeed, float tolerance)
        {
            if (ball.Count == 0) return;
            bool Fits(float t, float x, float y)
            {
                // Only judge where there's a real/filled ball sample close in time
                int idx = ball.times.BinarySearch(t);
                if (idx < 0) idx = ~idx;
                float near = Mathf.Min(idx > 0 ? t - ball.times[idx - 1] : float.MaxValue, idx < ball.Count ? ball.times[idx] - t : float.MaxValue);
                if (near > 0.5f) return true;
                return Vector2.Distance(ball.SampleClamped(t), new Vector2(x, y)) <= tolerance;
            }
            int before = events.Count;
            events.RemoveAll(e =>
            {
                if (e.type == "pass" && e.start != null && e.end != null)
                {
                    float dur = Mathf.Max(0.05f, e.t_end - e.t_start);
                    float dist = Vector2.Distance(new Vector2(e.start[0], e.start[1]), new Vector2(e.end[0], e.end[1]));
                    return dist / dur > maxSpeed * 1.2f || !Fits(e.t_start, e.start[0], e.start[1]) || !Fits(e.t_end, e.end[0], e.end[1]);
                }
                if (e.type == "touch") return !Fits(e.t, e.x, e.y);
                return false;
            });
            eventsDropped += before - events.Count;
        }

        public static MatchReplayData FromJson(string json, OriginMode origin, bool flipY, int minTrackSamples, int smoothRadius, CleanOptions clean = null)
        {
            clean ??= new CleanOptions();
            // JsonUtility can't read null into an int, so map "owner": null to -1
            json = System.Text.RegularExpressions.Regex.Replace(json, "\"owner\"\\s*:\\s*null", "\"owner\": -1");
            // Referees, unattributed events (e.g. aerial_ball) etc. use null for int IDs
            json = System.Text.RegularExpressions.Regex.Replace(json, "\"(team|player|from|to)\"\\s*:\\s*null", "\"$1\": -1");
            var file = JsonUtility.FromJson<MatchFile>(json);
            if (file == null || file.frames == null || file.frames.Count == 0)
                throw new Exception("Match file has no frames.");

            // "teams": {"0": {"kit_hex": "#a0adba", ...}, "1": {...}} — a dictionary JsonUtility can't read,
            // so pull the kit colors out directly. Only the kit block has kit_hex inside the braces.
            var kits = new Color?[2];
            for (int team = 0; team < 2; team++)
            {
                var m = System.Text.RegularExpressions.Regex.Match(json, "\"" + team + "\"\\s*:\\s*\\{[^{}]*\"kit_hex\"\\s*:\\s*\"(#[0-9a-fA-F]{6})\"");
                if (m.Success && ColorUtility.TryParseHtmlString(m.Groups[1].Value, out var c)) kits[team] = c;
            }

            var data = new MatchReplayData
            {
                length = file.pitch != null && file.pitch.length > 0 ? file.pitch.length : 105f,
                width = file.pitch != null && file.pitch.width > 0 ? file.pitch.width : 68f,
                fps = file.fps,
                startTime = file.frames[0].t,
                endTime = file.frames[file.frames.Count - 1].t,
            };
            if (kits[0].HasValue && kits[1].HasValue)
            {
                data.hasKitColors = true;
                data.kitColors = new[] { kits[0].Value, kits[1].Value };
            }

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

            if (file.ball_flights != null)
            {
                foreach (var fl in file.ball_flights)
                {
                    if (fl.start == null || fl.land == null || fl.start.Length < 2 || fl.land.Length < 2 || fl.Duration <= 0.01f) continue;
                    var s = ToCentered(fl.start[0], fl.start[1]); fl.start = new[] { s.x, s.y };
                    var l = ToCentered(fl.land[0], fl.land[1]); fl.land = new[] { l.x, l.y };
                    data.flights.Add(fl);
                }
                data.flights.Sort((a, b) => a.t_takeoff.CompareTo(b.t_takeoff));
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
            // False detections elsewhere on the pitch, then fill gaps from the player on the ball
            data.CleanBall(clean.ballMaxSpeed, clean.ballJumpMargin);
            if (clean.fillBallFromOwner) data.FillBallFromOwner(clean.ballMaxSpeed, clean.ballJumpMargin);
            if (clean.dropImplausibleEvents) data.DropImplausibleEvents(clean.ballMaxSpeed, 8f);
            data.ball.SmoothByTime(Mathf.Min(0.1f, clean.smoothWindow));
            data.PrepareFlights(clean.lobGravityCap);
            return data;
        }
    }
}
