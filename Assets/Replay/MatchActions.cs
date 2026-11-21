using System.Collections.Generic;
using UnityEngine;

namespace SoccerReplay
{
    /// <summary>
    /// An action animation (kick, receive, header, keeper catch...) with its measured contact moment.
    /// Filled in by Soccer Replay > Setup Animated Players.
    /// </summary>
    [System.Serializable]
    public class ActionClip
    {
        public string name;          // e.g. "Kick", "Receive", "KeeperCatch"
        public AnimationClip clip;
        public string state;         // Animator state name on the Actions layer
        [Tooltip("Seconds into the clip when the foot / head / hands meet the ball")]
        public float contact;
        [Tooltip("Where in the clip playback starts (skips long wind-ups)")]
        public float clipStart;
        [Tooltip("Where in the clip playback ends (skips long follow-throughs)")]
        public float clipEnd;
        [Tooltip("Position of the contact foot / head / hands at contact, relative to the character (x = right, z = forward)")]
        public Vector3 contactOffset;
        [Tooltip("True if the contact limb is on the character's right side (right foot / hand)")]
        public bool contactRightSide = true;
        [Tooltip("0 = feet, 1 = head, 2 = hands")]
        public int contactPart;
        [Tooltip("Keep at most this much of the clip before / after contact (s)")]
        public float maxLead = 0.8f, maxFollow = 0.6f;
        [Tooltip("Only look for the contact moment in the first N seconds of the clip")]
        public float searchEnd = 99f;
        [System.NonSerialized] public bool measured;

        public int StateHash => Animator.StringToHash(state);
        public float LeadTime => contact - clipStart;     // seconds of animation before contact
        public float Duration => clipEnd - clipStart;
    }

    /// <summary>One action a specific player performs, placed on the replay timeline.</summary>
    public class ScheduledAction
    {
        public int playerId;
        public ActionClip clip;
        public float contactTime;    // replay time of ball contact
        public float start, end;     // replay time the animation starts / ends
        public Vector2 faceTarget;   // where the player turns during the action
        public int priority;         // higher wins when two actions collide
        public bool mirror;          // play the mirrored clip (e.g. left foot) because the ball is on the other side
        public Vector2 nudge;        // shift at contact so the foot / head / hands meet the ball

        /// <summary>How much of the contact nudge applies at time t: rises to 1 at contact and back to 0.</summary>
        public float NudgeWeight(float t, float window)
        {
            float w = 1f - Mathf.Clamp01(Mathf.Abs(t - contactTime) / Mathf.Max(0.01f, window));
            return w * w * (3f - 2f * w);
        }

        /// <summary>Normalized time in the clip and a blend weight (ramps in/out) at replay time t.</summary>
        public void Evaluate(float t, float blendIn, float blendOut, out float normalized, out float weight)
        {
            float local = clip.clipStart + (t - start);
            normalized = clip.clip != null && clip.clip.length > 0f ? Mathf.Clamp01(local / clip.clip.length) : 0f;
            float wIn = blendIn > 0f ? Mathf.Clamp01((t - start) / blendIn) : 1f;
            float wOut = blendOut > 0f ? Mathf.Clamp01((end - t) / blendOut) : 1f;
            weight = Mathf.Min(wIn, wOut);
            weight = weight * weight * (3f - 2f * weight);
        }
    }

    /// <summary>Turns match events into per-player actions.</summary>
    public static class ActionScheduler
    {
        public struct Settings
        {
            public float runningKickSpeed;   // passer faster than this (m/s) uses the running kick
            public float longPassLength;     // passes longer than this (m) use the long kick
            public float headerMinHeight;    // ball arriving at least this high -> header
            public float chestMinHeight;     // ball arriving between this and header height -> receive
            public float nearestPlayerRange; // how close a player must be to a landing ball to react (m)
            public float dedupWindow;        // actions by one player closer than this (s) keep the higher priority
            public float maxNudge;           // largest shift (m) used to bring the foot to the ball
            public bool allowMirror;         // use the mirrored clip when the ball is on the other side
        }

        const int PriorityReceive = 1, PriorityKick = 2, PriorityHeader = 3;

        /// <param name="isKeeper">true if a tracking ID belongs to a goalkeeper</param>
        public static Dictionary<int, List<ScheduledAction>> Build(MatchReplayData data, Dictionary<string, ActionClip> clips,
            System.Func<int, bool> isKeeper, Settings s, out string summary)
        {
            var all = new List<ScheduledAction>();
            ActionClip C(string n) => clips.TryGetValue(n, out var c) && c.clip != null ? c : null;
            int passes = 0, receives = 0, headers = 0, keeperActions = 0, skipped = 0;

            // Player speed at a moment, from whichever track piece has them then
            float SpeedOf(int id, float t)
            {
                foreach (var tr in data.players)
                    if (tr.id == id && t >= tr.Start - 0.2f && t <= tr.End + 0.2f)
                        return tr.VelocityAt(Mathf.Clamp(t, tr.Start, tr.End), 0.4f).magnitude;
                return 0f;
            }

            Vector2? PlayerAt(int id, float t)
            {
                foreach (var tr in data.players)
                    if (tr.id == id && tr.Count > 0 && t >= tr.Start - 0.5f && t <= tr.End + 0.5f) return tr.SampleClamped(t);
                return null;
            }

            void Add(int id, ActionClip clip, float contact, Vector2 face, int priority, Vector2 ball)
            {
                if (clip == null || id < 0) { skipped++; return; }
                var a = new ScheduledAction
                {
                    playerId = id, clip = clip, contactTime = contact, faceTarget = face, priority = priority,
                    start = contact - clip.LeadTime, end = contact - clip.LeadTime + clip.Duration,
                };

                // Contact alignment: which side is the ball on, and how far must the player shift so the
                // contact limb (from the clip, mirrored if needed) lands on the ball at the contact moment
                var p = PlayerAt(id, contact);
                if (p.HasValue)
                {
                    Vector2 dir = face - p.Value;
                    dir = dir.sqrMagnitude > 0.01f ? dir.normalized : new Vector2(0f, 1f);
                    var right = new Vector2(dir.y, -dir.x);
                    float side = Vector2.Dot(ball - p.Value, right);
                    a.mirror = s.allowMirror && Mathf.Abs(side) > 0.05f && (clip.contactRightSide ? side < 0f : side > 0f);
                    float ox = a.mirror ? -clip.contactOffset.x : clip.contactOffset.x;
                    Vector2 limb = p.Value + right * ox + dir * clip.contactOffset.z;
                    a.nudge = Vector2.ClampMagnitude(ball - limb, s.maxNudge);
                }
                all.Add(a);
            }

            foreach (var e in data.events)
            {
                if (e.type == "pass" && e.start != null && e.end != null)
                {
                    var target = new Vector2(e.end[0], e.end[1]);
                    var ballAtKick = new Vector2(e.start[0], e.start[1]);   // where the ball was when it was struck
                    bool longBall = e.aerial || e.length > s.longPassLength;
                    if (isKeeper(e.from))
                    {
                        Add(e.from, longBall ? C("KeeperDropKick") : C("KeeperPass"), e.t_start, target, PriorityKick, ballAtKick);
                        keeperActions++;
                    }
                    else
                    {
                        var clip = longBall ? C("LongKick") : SpeedOf(e.from, e.t_start) > s.runningKickSpeed ? C("KickRun") : C("Kick");
                        Add(e.from, clip, e.t_start, target, PriorityKick, ballAtKick);
                        passes++;
                    }
                }
                else if (e.type == "touch" && e.player >= 0)
                {
                    // Face where the ball came from: its position a moment before the touch
                    Vector2 from = data.ball.Count > 0 ? data.ball.SampleClamped(e.t - 0.4f) : new Vector2(e.x, e.y);
                    var ballAtTouch = new Vector2(e.x, e.y);
                    if (isKeeper(e.player)) { Add(e.player, C("KeeperCatch"), e.t, from, PriorityReceive, ballAtTouch); keeperActions++; }
                    else { Add(e.player, C("Receive"), e.t, from, PriorityReceive, ballAtTouch); receives++; }
                }
            }

            // High balls arriving at a player: header (or receive/catch at chest height)
            foreach (var f in data.flights)
            {
                if (f.end_height < s.chestMinHeight || f.land == null) continue;
                var land = new Vector2(f.land[0], f.land[1]);
                int best = -1; float bestD = s.nearestPlayerRange;
                foreach (var tr in data.players)
                {
                    if (tr.referee || !tr.TrySample(f.t_land, 0.5f, out var p)) continue;
                    float d = Vector2.Distance(p, land);
                    if (d < bestD) { bestD = d; best = tr.id; }
                }
                if (best < 0) continue;
                var startPt = new Vector2(f.start[0], f.start[1]);
                if (isKeeper(best)) { Add(best, C("KeeperCatch"), f.t_land, startPt, PriorityHeader, land); keeperActions++; }
                else if (f.end_height >= s.headerMinHeight) { Add(best, C("Header"), f.t_land, startPt, PriorityHeader, land); headers++; }
                else { Add(best, C("Receive"), f.t_land, startPt, PriorityReceive, land); receives++; }
            }

            // Per player: sort, and where two actions collide keep the higher priority
            var byPlayer = new Dictionary<int, List<ScheduledAction>>();
            foreach (var a in all)
            {
                if (!byPlayer.TryGetValue(a.playerId, out var list)) byPlayer[a.playerId] = list = new List<ScheduledAction>();
                list.Add(a);
            }
            int merged = 0;
            foreach (var list in byPlayer.Values)
            {
                list.Sort((a, b) => a.contactTime.CompareTo(b.contactTime));
                for (int i = list.Count - 1; i > 0; i--)
                {
                    var a = list[i - 1]; var b = list[i];
                    if (b.contactTime - a.contactTime > s.dedupWindow) continue;
                    if (b.priority >= a.priority) list.RemoveAt(i - 1); else list.RemoveAt(i);
                    merged++;
                }
                // A following action cuts the previous one short so they don't overlap
                for (int i = 0; i + 1 < list.Count; i++)
                    if (list[i].end > list[i + 1].start) list[i].end = Mathf.Max(list[i].contactTime + 0.15f, list[i + 1].start);
            }
            summary = $"{passes} kicks, {receives} receives, {headers} headers, {keeperActions} keeper actions ({merged} merged, {skipped} skipped: missing clip or player)";
            return byPlayer;
        }

        /// <summary>The action active at time t in a time-sorted list, or null.</summary>
        public static ScheduledAction ActiveAt(List<ScheduledAction> list, float t)
        {
            if (list == null) return null;
            ScheduledAction active = null;
            foreach (var a in list)
            {
                if (a.start > t) break;
                if (t <= a.end) active = a;
            }
            return active;
        }
    }
}
