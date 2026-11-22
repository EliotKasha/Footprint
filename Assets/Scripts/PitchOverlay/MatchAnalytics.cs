// MatchAnalytics.cs
// Pure C# (no GameObjects). Replays the data up to a point in time and keeps the numbers the stats
// panel shows. Also contains FormationShape, which turns player positions into the pink "links".
//
// Where each number comes from:
//   possession   - frame "owner" field; if it is null, the formation-ml rule from broadcast.py
//                  (nearest player within 2.5 m of the ball, otherwise whoever had it last)
//   attack side  - formation-ml rule: a team attacks away from its goalkeeper; if no keeper has been
//                  seen, it attacks the opposite way to the other team
//   passes/touch - the "events" list in match_events.json
//   width/depth  - spread of the team's visible outfield players (smoothed)
//   threat       - PLACEHOLDER heuristic (see ComputeThreat). Not in the data, swap in a real model.
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace PitchOverlay
{
    public class TeamLive
    {
        public float possessionPct = 50f;
        public int passes, passesCompleted, touches, aerialPasses;
        public float threat;               // 0..100
        public float width, depth;         // metres, smoothed
        public float lineHeight;           // metres from the team's own goal line, smoothed
        public string formation;           // optional - filled by SetFormation (hidden if null)
        public float formationConf;        // 0..1
        public MeasuredShape defending;    // [FORMATION-ML] measured shape when defending (from analysis.json)
        public MeasuredShape attacking;    // [FORMATION-ML] measured shape when attacking (from analysis.json)
        public float PassAccuracy { get { return passes > 0 ? 100f * passesCompleted / passes : -1f; } }
    }

    public struct Pt { public int id; public Vector2 p; }          // p = (world x, world z) in metres
    public struct LinkPair { public Pt a, b; }

    public static class FormationShape
    {
        // Split a team's visible outfield players into 1-3 "lines" (back / middle / front) by depth.
        // Brute-forces the split that makes each line as tight as possible. Needs >= 2 players per line.
        public static List<List<Pt>> BuildLines(List<Pt> pts, float attackSign)
        {
            var order = pts.OrderBy(q => attackSign * q.p.x).ToList();
            int n = order.Count;
            int k = n < 4 ? 1 : (n < 6 ? 2 : 3);
            if (k == 1) return new List<List<Pt>> { order };

            float[] dep = order.Select(q => attackSign * q.p.x).ToArray();
            float best = float.MaxValue; int bestA = 0, bestB = 0;

            if (k == 2)
            {
                for (int a = 2; a <= n - 2; a++)
                {
                    float c = Cost(dep, 0, a) + Cost(dep, a, n);
                    if (c < best) { best = c; bestA = a; }
                }
                return new List<List<Pt>> { order.GetRange(0, bestA), order.GetRange(bestA, n - bestA) };
            }

            for (int a = 2; a <= n - 4; a++)
                for (int b = a + 2; b <= n - 2; b++)
                {
                    float c = Cost(dep, 0, a) + Cost(dep, a, b) + Cost(dep, b, n);
                    if (c < best) { best = c; bestA = a; bestB = b; }
                }
            return new List<List<Pt>>
            {
                order.GetRange(0, bestA),
                order.GetRange(bestA, bestB - bestA),
                order.GetRange(bestB, n - bestB)
            };
        }

        static float Cost(float[] v, int s, int e)
        {
            float mean = 0f; for (int i = s; i < e; i++) mean += v[i]; mean /= (e - s);
            float c = 0f; for (int i = s; i < e; i++) c += (v[i] - mean) * (v[i] - mean);
            return c;
        }

        // Link neighbours inside a line, left to right across the pitch. A link is only made if the two
        // players are really side by side (not too far apart, not at very different depths).
        public static List<LinkPair> BuildLinks(List<List<Pt>> lines, float attackSign,
                                                float maxLateral = 24f, float maxDepth = 10f)
        {
            var result = new List<LinkPair>();
            foreach (var line in lines)
            {
                var s = line.OrderBy(q => q.p.y).ToList();
                for (int i = 0; i < s.Count - 1; i++)
                {
                    float lateral = Mathf.Abs(s[i].p.y - s[i + 1].p.y);
                    float depth = Mathf.Abs(attackSign * (s[i].p.x - s[i + 1].p.x));
                    if (lateral <= maxLateral && depth <= maxDepth)
                        result.Add(new LinkPair { a = s[i], b = s[i + 1] });
                }
            }
            return result;
        }
    }

    public class MatchAnalytics
    {
        public const float HalfL = 52.5f;
        const float BallRadius = 2.5f;     // same as BALL_DIST in formation-ml/broadcast.py

        enum Kind { Touch, PassStart, PassEnd }
        struct Timed { public float t; public Kind kind; public MatchEvent e; }
        struct PassDone { public int a, b; public float t; }

        public readonly TeamLive[] team = { new TeamLive(), new TeamLive() };
        public float time;
        public int holder = -1;                       // team with the ball, -1 unknown
        public float ballX, ballY, ballZ;
        public string lastEventText = "";
        public float lastEventTime = -999f;
        public int lastEventTeam = -1;

        readonly MatchFile match;
        readonly List<Timed> timeline = new List<Timed>();
        readonly List<PassDone> passesDone = new List<PassDone>();
        readonly Dictionary<int, int> teamOf = new Dictionary<int, int>();
        readonly float[] gkSum = new float[2];
        readonly int[] gkN = new int[2];
        readonly float[] holdTime = new float[2];
        readonly bool[] shapeInit = new bool[2];
        int fi, ei;
        float prevFrameT = -1f;

        public MatchAnalytics(MatchFile m)
        {
            match = m;
            foreach (var e in m.events)
            {
                if (e.type == "touch" && e.t.HasValue)
                    timeline.Add(new Timed { t = e.t.Value, kind = Kind.Touch, e = e });
                else if (e.type == "pass")
                {
                    if (e.t_start.HasValue) timeline.Add(new Timed { t = e.t_start.Value, kind = Kind.PassStart, e = e });
                    if (e.t_end.HasValue) timeline.Add(new Timed { t = e.t_end.Value, kind = Kind.PassEnd, e = e });
                }
            }
            timeline.Sort((x, y) => x.t.CompareTo(y.t));
        }

        public void Reset()
        {
            for (int i = 0; i < 2; i++)
            {
                var keepFormation = team[i].formation; var keepConf = team[i].formationConf;
                // [FORMATION-ML] the measured shapes are fields of TeamLive, which is reused (not recreated),
                // so they survive this reset automatically - nothing to restore.
                team[i].possessionPct = 50f; team[i].passes = team[i].passesCompleted = 0;
                team[i].touches = team[i].aerialPasses = 0; team[i].threat = 0f;
                team[i].width = team[i].depth = team[i].lineHeight = 0f;
                team[i].formation = keepFormation; team[i].formationConf = keepConf;
                holdTime[i] = 0f; gkSum[i] = 0f; gkN[i] = 0; shapeInit[i] = false;
            }
            passesDone.Clear(); teamOf.Clear();
            fi = ei = 0; prevFrameT = -1f; holder = -1; lastEventTime = -999f; lastEventText = "";
        }

        public void AdvanceTo(float t)
        {
            if (t < time - 0.5f) Reset();          // clip looped / user scrubbed backwards
            while (fi < match.frames.Count && match.frames[fi].t <= t) ProcessFrame(match.frames[fi++]);
            while (ei < timeline.Count && timeline[ei].t <= t) ProcessEvent(timeline[ei++]);
            time = t;
        }

        // +1 = team attacks towards +x, -1 = towards -x.  (formation-ml: team_frames in broadcast.py)
        public float AttackSign(int tm)
        {
            if (gkN[tm] > 0) return (gkSum[tm] / gkN[tm]) < 0f ? 1f : -1f;
            int other = 1 - tm;
            if (gkN[other] > 0) return (gkSum[other] / gkN[other]) < 0f ? -1f : 1f;
            return tm == 0 ? 1f : -1f;
        }

        public int PassesBetween(int a, int b, float now, float window)
        {
            int c = 0;
            foreach (var p in passesDone)
                if (p.t >= now - window && ((p.a == a && p.b == b) || (p.a == b && p.b == a))) c++;
            return c;
        }

        // Optional: call this from your formation model. The panel only shows it if conf >= 0.40
        // (the NOT_SURE threshold in formation-ml/broadcast.py).
        public void SetFormation(int tm, string label, float conf)
        {
            team[tm].formation = label; team[tm].formationConf = conf;
        }

        void ProcessFrame(FrameData f)
        {
            float dt = prevFrameT < 0f ? 0f : Mathf.Min(f.t - prevFrameT, 0.5f);
            prevFrameT = f.t;

            var outfield = new List<Vector2>[] { new List<Vector2>(), new List<Vector2>() };
            foreach (var p in f.players)
            {
                if (!p.team.HasValue || p.team.Value < 0 || p.team.Value > 1) continue;
                int tm = p.team.Value;
                teamOf[p.id] = tm;
                if (p.role == "goalkeeper") { gkSum[tm] += p.x; gkN[tm]++; }
                else outfield[tm].Add(new Vector2(p.x, p.y));
            }

            bool hasBall = f.ball != null && f.ball.Length >= 2;
            if (hasBall) { ballX = f.ball[0]; ballY = f.ball[1]; ballZ = f.ball.Length > 2 ? f.ball[2] : 0f; }

            // --- possession ---
            int h = -1;
            int ownerTeam;
            if (f.owner.HasValue && teamOf.TryGetValue(f.owner.Value, out ownerTeam)) h = ownerTeam;
            else if (hasBall)
            {
                float bestD = BallRadius;
                foreach (var p in f.players)
                {
                    if (!p.team.HasValue) continue;
                    float d = Mathf.Sqrt((p.x - ballX) * (p.x - ballX) + (p.y - ballY) * (p.y - ballY));
                    if (d < bestD) { bestD = d; h = p.team.Value; }
                }
            }
            if (h >= 0) holder = h;
            if (holder >= 0) holdTime[holder] += dt;
            float total = holdTime[0] + holdTime[1];
            for (int i = 0; i < 2; i++) team[i].possessionPct = total > 0f ? 100f * holdTime[i] / total : 50f;

            // --- shape + threat ---
            float ema = dt > 0f ? 1f - Mathf.Exp(-dt / 3f) : 1f;
            float emaT = dt > 0f ? 1f - Mathf.Exp(-dt / 1.5f) : 1f;
            for (int tm = 0; tm < 2; tm++)
            {
                float sign = AttackSign(tm);
                var pts = outfield[tm];
                if (pts.Count >= 2)
                {
                    float yMin = pts.Min(v => v.y), yMax = pts.Max(v => v.y);
                    float dMin = pts.Min(v => sign * v.x), dMax = pts.Max(v => sign * v.x);
                    float line = pts.Average(v => sign * v.x) + HalfL;
                    float k = shapeInit[tm] ? ema : 1f;
                    team[tm].width = Mathf.Lerp(team[tm].width, yMax - yMin, k);
                    team[tm].depth = Mathf.Lerp(team[tm].depth, dMax - dMin, k);
                    team[tm].lineHeight = Mathf.Lerp(team[tm].lineHeight, line, k);
                    shapeInit[tm] = true;
                }
                team[tm].threat = Mathf.Lerp(team[tm].threat, ComputeThreat(tm, sign, pts, hasBall), emaT);
            }
        }

        // PLACEHOLDER "goal threat" 0..100. Not part of the data or the formation-ml code, so this is a
        // simple, explainable stand-in: how close the ball is to the goal this team attacks, plus how many
        // of its players are near that goal, scaled by whether the team has the ball.
        // Replace this one function with a real model (e.g. expected threat) when you have one.
        float ComputeThreat(int tm, float sign, List<Vector2> outfield, bool hasBall)
        {
            if (!hasBall) return 0f;
            float gx = sign * HalfL;
            float dist = Mathf.Sqrt((ballX - gx) * (ballX - gx) + ballY * ballY);
            float zone = Mathf.Exp(-Mathf.Pow(dist / 28f, 1.6f));            // ~0.85 at 10 m, ~0.07 at 52 m
            int near = outfield.Count(v => Mathf.Sqrt((v.x - gx) * (v.x - gx) + v.y * v.y) < 25f);
            float crowd = Mathf.Min(1f, near / 3f);
            float poss = holder == tm ? 1f : (holder < 0 ? 0.5f : 0.05f);   // no ball = almost no threat
            return 100f * Mathf.Clamp01((0.75f * zone + 0.25f * crowd) * poss);
        }

        void ProcessEvent(Timed te)
        {
            var e = te.e;
            int tm = e.team.HasValue ? e.team.Value : -1;
            bool ok = tm == 0 || tm == 1;
            switch (te.kind)
            {
                case Kind.Touch:
                    if (ok) team[tm].touches++;
                    Say(tm, te.t, "Touch  #" + e.player);
                    break;
                case Kind.PassStart:
                    if (ok) { team[tm].passes++; if (e.aerial == true) team[tm].aerialPasses++; }
                    Say(tm, te.t, "Pass  #" + e.passer + " \u2192 #" + e.receiver +
                        (e.length.HasValue ? "  \u00B7  " + e.length.Value.ToString("0.#") + " m" : "") +
                        (e.aerial == true ? "  \u00B7  aerial" : ""));
                    break;
                case Kind.PassEnd:
                    if (ok && e.completed == true && e.passer.HasValue && e.receiver.HasValue)
                    {
                        team[tm].passesCompleted++;
                        passesDone.Add(new PassDone { a = e.passer.Value, b = e.receiver.Value, t = te.t });
                    }
                    break;
            }
        }

        void Say(int tm, float t, string text) { lastEventText = text; lastEventTime = t; lastEventTeam = tm; }
    }
}
