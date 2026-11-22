// LiveShapes.cs  [REPLAY]  (new file)
// C# port of formation-ml's shape.py ("measured shape"), so the formation chips work from match_events.json
// alone - no Python, no .analysis.json. Same rules as shape.py / broadcast.team_frames / infer_possession:
//   - about 3 moments per second, per team, split into "defending" (other team has the ball) and "attacking"
//   - a moment counts when all 10 outfield players are on camera, or 9 if the camera view shows which
//     line the missing player must be in (off-screen in front = forward, behind = defender)
//   - skip moments when the team is pinned in its own box (deepest outfield player < 12 m from own goal line)
//   - split the players into 3 lines by depth, choosing the tightest split; each moment votes
// The one difference: instead of voting over the whole clip, Apply() votes over the last `window` seconds
// of the replay, so the chip shows the shape of the current phase of play.
// With window <= 0 and minMoments = 20 it gives exactly what export_for_unity.py writes (see Summary()).
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PitchOverlay
{
    public class LiveShapes
    {
        public const int MomentsPerSecond = 3;     // shape.py MOMENTS_PER_SECOND
        const double MinDepth = 12.0;              // shape.py MIN_DEPTH
        const double L = 105.0, W = 68.0;
        const double BallDist = 2.5;               // broadcast.py BALL_DIST
        static readonly string[] Phases = { "defending", "attacking" };

        struct Moment { public float t; public int shape; }     // shape: packed line sizes, -1 = unreadable

        // [team, phase] -> moments in time order (phase 0 = defending, 1 = attacking)
        readonly List<Moment>[,] moments = new List<Moment>[2, 2];
        readonly MeasuredShape[,] live = new MeasuredShape[2, 2];
        public readonly bool[] attacksRight = new bool[2];

        public LiveShapes(MatchFile match)
        {
            for (int t = 0; t < 2; t++)
                for (int ph = 0; ph < 2; ph++) { moments[t, ph] = new List<Moment>(); live[t, ph] = new MeasuredShape { rough = ph == 1 }; }
            if (match == null || match.frames == null || match.frames.Count == 0) return;
            var frames = match.frames;
            var holders = InferPossession(frames);

            // Attack direction from the goalkeepers' average x over the clip (broadcast.team_frames)
            double[] gkSum = new double[2]; int[] gkN = new int[2];
            foreach (var f in frames)
                foreach (var p in f.players)
                    if (p.role == "goalkeeper" && p.team.HasValue && (p.team == 0 || p.team == 1)) { gkSum[p.team.Value] += p.x; gkN[p.team.Value]++; }
            bool[] known = new bool[2];
            for (int t = 0; t < 2; t++) if (gkN[t] > 0) { attacksRight[t] = gkSum[t] / gkN[t] < 0; known[t] = true; }
            for (int t = 0; t < 2; t++) if (!known[t] && known[1 - t]) { attacksRight[t] = !attacksRight[1 - t]; known[t] = true; }
            if (!known[0] || !known[1])
            {
                double[] mean = new double[2];
                for (int k = 0; k < 2; k++)
                {
                    double s = 0; int n = 0;
                    foreach (var f in frames) foreach (var p in f.players) if (p.team == k) { s += p.x; n++; }
                    mean[k] = n > 0 ? s / n : 0;
                }
                for (int t = 0; t < 2; t++) if (!known[t]) attacksRight[t] = mean[t] < mean[1 - t];
            }

            int[,] lastSlot = { { int.MinValue, int.MinValue }, { int.MinValue, int.MinValue } };
            var pts = new List<double[]>();
            for (int fi = 0; fi < frames.Count; fi++)
            {
                var f = frames[fi];
                if (f.view_polygon == null || f.view_polygon.Length == 0) continue;
                for (int t = 0; t < 2; t++)
                {
                    int hasBall = holders[fi] < 0 ? 0 : (holders[fi] == t ? 1 : -1);
                    if (hasBall == 0) continue;
                    int ph = hasBall == 1 ? 1 : 0;
                    pts.Clear();
                    foreach (var p in f.players)
                        if (p.team == t && p.role == "player")
                        {
                            double x = p.x + L / 2, y = p.y + W / 2;
                            if (!attacksRight[t]) x = L - x;
                            pts.Add(new[] { x, y });
                        }
                    if (pts.Count != 9 && pts.Count != 10) continue;
                    int slot = (int)System.Math.Floor(f.t * MomentsPerSecond);
                    if (slot == lastSlot[t, ph]) continue;
                    lastSlot[t, ph] = slot;
                    var view = new double[f.view_polygon.Length][];
                    for (int i = 0; i < view.Length; i++)
                    {
                        double vx = f.view_polygon[i][0] + L / 2, vy = f.view_polygon[i][1] + W / 2;
                        view[i] = new[] { attacksRight[t] ? vx : L - vx, vy };
                    }
                    moments[t, ph].Add(new Moment { t = f.t, shape = MomentShape(pts, view) });
                }
            }
        }

        // ---------- the live result ----------

        // Measure each team's shape over the replay time (now - window, now] and copy it into the analytics,
        // where StatsPanel shows it. Returns the same MeasuredShape objects every call (updated in place).
        public void Apply(MatchAnalytics analytics, float now, float window, int minMoments)
        {
            for (int t = 0; t < 2; t++)
            {
                for (int ph = 0; ph < 2; ph++) Measure(moments[t, ph], now, window, minMoments, live[t, ph]);
                analytics.team[t].defending = live[t, 0];
                analytics.team[t].attacking = live[t, 1];
            }
        }

        static void Measure(List<Moment> ms, float now, float window, int minMoments, MeasuredShape into)
        {
            var votes = new Dictionary<int, int>();
            var order = new List<int>();                 // first-seen order, to break ties like Counter does
            foreach (var m in ms)
            {
                if (m.t > now) break;
                if (window > 0f && m.t <= now - window) continue;
                if (m.shape < 0) continue;
                int c;
                if (votes.TryGetValue(m.shape, out c)) votes[m.shape] = c + 1;
                else { votes[m.shape] = 1; order.Add(m.shape); }
            }
            int used = votes.Values.Sum();
            into.seconds_seen = (float)System.Math.Round((double)used / MomentsPerSecond, 1);
            into.runners_up.Clear();
            if (used < minMoments || used == 0)
            {
                into.shape = null; into.confidence = null; into.agreement = 0f;
                return;
            }
            var ranked = order.OrderByDescending(s => votes[s]).ToList();     // stable: ties keep first-seen order
            double share = (double)votes[ranked[0]] / used;
            into.shape = Unpack(ranked[0]);
            into.agreement = (float)System.Math.Round(share, 3);
            into.confidence = Confidence(share, used);
            for (int i = 1; i < ranked.Count && i < 4; i++) into.runners_up.Add(Unpack(ranked[i]));
        }

        // shape.py confidence()
        public static string Confidence(double share, int used)
        {
            double seconds = (double)used / MomentsPerSecond;
            if (seconds < 8 || share < 0.20) return "low";
            if (seconds >= 20 && share >= 0.35) return "high";
            return "medium";
        }

        // ---------- shape.py ----------

        static int MomentShape(List<double[]> pts, double[][] view)
        {
            double lo = double.MaxValue, hi = double.MinValue;
            foreach (var p in pts) { if (p[0] < lo) lo = p[0]; if (p[0] > hi) hi = p[0]; }
            if (lo < MinDepth) return -1;                                   // pinned in own box
            var depths = pts.Select(p => p[0]).OrderBy(d => d).ToArray();
            int[] sizes = BestSplit3(depths);
            if (pts.Count == 10) return Pack(sizes);
            double ahead = (VisibleShare(view, hi + 6) + VisibleShare(view, hi + 12) + VisibleShare(view, hi + 18)) / 3.0;
            double behind = (VisibleShare(view, lo - 6) + VisibleShare(view, lo - 12) + VisibleShare(view, lo - 18)) / 3.0;
            if (ahead < 0.25 && behind > 0.75) sizes[2]++;                  // missing player is a forward
            else if (behind < 0.25 && ahead > 0.75) sizes[0]++;             // missing player is a defender
            else return -1;
            return Pack(sizes);
        }

        // Tightest split of sorted depths into 3 lines (itertools.combinations order, strict < like numpy code)
        static int[] BestSplit3(double[] d)
        {
            int n = d.Length;
            int[] best = null; double bestCost = double.PositiveInfinity;
            for (int i = 1; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    double cost = Spread(d, 0, i) + Spread(d, i, j) + Spread(d, j, n);
                    if (cost < bestCost) { bestCost = cost; best = new[] { i, j - i, n - j }; }
                }
            return best;
        }

        static double Spread(double[] d, int a, int b)
        {
            double mean = 0; for (int i = a; i < b; i++) mean += d[i]; mean /= (b - a);
            double s = 0; for (int i = a; i < b; i++) s += (d[i] - mean) * (d[i] - mean);
            return s;
        }

        // Share of 8 points across the pitch width (y = 6..62 m) at depth x that are inside the camera view
        static double VisibleShare(double[][] view, double x)
        {
            int inside = 0;
            for (int k = 0; k < 8; k++)
            {
                double y = 6.0 + k * (56.0 / 7.0);
                if (Inside(view, x, y)) inside++;
            }
            return inside / 8.0;
        }

        static bool Inside(double[][] v, double x, double y)
        {
            bool inside = false;
            for (int i = 0, j = v.Length - 1; i < v.Length; j = i++)
            {
                double xi = v[i][0], yi = v[i][1], xj = v[j][0], yj = v[j][1];
                if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
            }
            return inside;
        }

        static int Pack(int[] s) { return s[0] * 10000 + s[1] * 100 + s[2]; }
        static string Unpack(int k) { return (k / 10000) + "-" + (k / 100 % 100) + "-" + (k % 100); }

        // broadcast.infer_possession: the tracker's owner, else nearest player within 2.5 m, else last holder
        static int[] InferPossession(List<FrameData> frames)
        {
            var outArr = new int[frames.Count];
            int holder = -1;
            for (int i = 0; i < frames.Count; i++)
            {
                var fr = frames[i];
                int? ownerTeam = null;
                if (fr.owner.HasValue)
                    foreach (var p in fr.players) if (p.id == fr.owner.Value) { ownerTeam = p.team; break; }
                if (ownerTeam.HasValue) holder = ownerTeam.Value;
                else if (fr.ball != null && fr.ball.Length >= 2 && fr.players.Count > 0)
                {
                    PlayerSample near = null; double bestD = double.MaxValue;
                    foreach (var p in fr.players)
                    {
                        double d = (p.x - fr.ball[0]) * (double)(p.x - fr.ball[0]) + (p.y - fr.ball[1]) * (double)(p.y - fr.ball[1]);
                        if (d < bestD) { bestD = d; near = p; }
                    }
                    if (near.team.HasValue && System.Math.Sqrt(bestD) < BallDist) holder = near.team.Value;
                }
                outArr[i] = holder;
            }
            return outArr;
        }

        // ---------- checking against formation-ml ----------

        public static LiveShapes FromFile(string path)
        {
            return new LiveShapes(Newtonsoft.Json.JsonConvert.DeserializeObject<MatchFile>(System.IO.File.ReadAllText(path)));
        }

        // Whole-clip result in the same layout as export_for_unity.py, for comparing with the Python output.
        public string Summary(int minMoments = 20)
        {
            var sb = new StringBuilder();
            var m = new MeasuredShape();
            for (int t = 0; t < 2; t++)
                for (int ph = 0; ph < 2; ph++)
                {
                    Measure(moments[t, ph], float.MaxValue, 0f, minMoments, m);
                    sb.Append($"  team{t} {Phases[ph],-9} {m.shape ?? "None"} {m.confidence ?? "None"} agree={m.agreement} seen={m.seconds_seen}\n");
                }
            return sb.ToString();
        }
    }
}
