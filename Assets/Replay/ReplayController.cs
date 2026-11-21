using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.InputSystem;

namespace SoccerReplay
{
    /// <summary>
    /// Loads a tracking JSON from StreamingAssets and replays it in 3D.
    /// Controls: Space play/pause · Left/Right skip 1s · [ ] speed · 1/2/3 camera · R restart
    /// </summary>
    public class ReplayController : MonoBehaviour
    {
        public enum CameraMode { Broadcast, TopDown, FollowBall }

        [Header("Data")]
        [Tooltip("Path inside Assets/StreamingAssets")]
        public string matchFile = "matches/match_events_v2.json";
        public OriginMode origin = OriginMode.Auto;
        [Tooltip("Mirror the pitch across its length if the replay looks left/right swapped vs. the video")]
        public bool flipY = false;
        [Tooltip("Tracks with fewer samples than this are treated as tracker noise and hidden")]
        public int minTrackSamples = 8;
        [Range(0, 5)] public int smoothRadius = 2; // legacy, superseded by smoothWindow

        [Header("Data cleanup")]
        [Tooltip("Seconds either side used to smooth jitter (works at any frame rate)")]
        [Range(0f, 0.6f)] public float smoothWindow = 0.2f;
        [Tooltip("Movement faster than this (m/s) is treated as a tracking error, not a real run")]
        public float maxPlayerSpeed = 11f;
        [Tooltip("Cancel moments where every player jumps together (camera calibration jumps)")]
        public bool correctCameraShifts = true;
        [Tooltip("Seconds players take to fade in/out when tracking starts or stops")]
        public float fadeTime = 0.25f;
        [Tooltip("Longest gap (seconds) a player is interpolated across before being hidden")]
        public float playerMaxGap = 0.4f;
        [Tooltip("Longest gap (seconds) the ball is interpolated across before being hidden")]
        public float ballMaxGap = 1.0f;

        [Header("Look")]
        public Color homeColor = new Color(0.85f, 0.15f, 0.15f);
        public Color awayColor = new Color(0.15f, 0.35f, 0.9f);
        public Color homeKeeperColor = new Color(1f, 0.85f, 0.1f);
        public Color awayKeeperColor = new Color(0.95f, 0.3f, 0.85f);

        [Header("Body motion")]
        public BodyMotionSettings bodyMotion = new BodyMotionSettings();

        [Header("Goalkeepers")]
        [Tooltip("Always show both keepers, estimating their position when the CV doesn't see them")]
        public bool estimateKeepers = true;
        [Tooltip("Work out which goal each team defends from where its tracked keeper stands")]
        public bool autoDetectSides = true;
        [Tooltip("Used when sides can't be detected: does team 0 defend the goal at negative x?")]
        public bool homeDefendsWest = true;
        [Tooltip("Seconds over which an estimated keeper blends into / out of the real tracked position")]
        public float keeperBlendTime = 2.5f;
        [Range(0.1f, 1f)] public float estimatedAlpha = 0.55f;
        [Tooltip("How much an untracked keeper reacts to the ball, relative to where he was last seen (0 = stands still)")]
        [Range(0f, 1f)] public float keeperDriftFactor = 0.2f;
        [Tooltip("Furthest an untracked keeper wanders from where he was last/next seen (meters)")]
        public float keeperMaxDrift = 2.5f;
        [Tooltip("Top speed of the displayed keeper (m/s), so he never sprints between positions")]
        public float keeperMaxSpeed = 3f;
        [Tooltip("Distance off the goal line when the ball is near / far (meters)")]
        public float keeperMinDepth = 3f;
        public float keeperMaxDepth = 12f;

        class Keeper
        {
            public int team;
            public int side;        // -1 defends the goal at x = -length/2, +1 the goal at +length/2
            public Track track;     // null when the CV never saw this keeper
            public PlayerBody body;
            public Material mat, trimMat;
            public Color color;
            public TextMeshPro label;
            public Vector2 smoothedEstimate;
            public Vector2 displayPos, prevDisplayPos, vel;
            public float realWeight; // 1 = fully tracked, 0 = fully estimated
        }
        readonly List<Keeper> keepers = new List<Keeper>();
        bool snapKeepers = true;
        public bool showLabels = true;

        [Header("Playback")]
        public float speed = 1f;
        public bool loop = true;
        public CameraMode cameraMode = CameraMode.Broadcast;

        MatchReplayData data;
        string loadError;
        float time;
        bool playing = true;

        Camera cam;
        Transform ballGo;
        readonly List<(Track track, PlayerBody body)> players = new List<(Track, PlayerBody)>();
        float motionDt;          // replay time elapsed this frame (0 while paused)
        bool motionSnap = true;  // jump bodies straight to their state (after seeks / loops)
        int visibleCount;
        bool ballVisible;
        Vector2 lastBall;
        float lastBallHeight;

        StadiumBuilder stadium;
        Transform ballShadow;
        Material ballShadowMat;
        Transform ownerRing;
        LineRenderer passLine;
        readonly Dictionary<int, List<Transform>> actorsById = new Dictionary<int, List<Transform>>();

        void RegisterActor(int id, Transform t)
        {
            if (!actorsById.TryGetValue(id, out var list)) actorsById[id] = list = new List<Transform>();
            list.Add(t);
        }

        Transform ActiveActor(int id)
        {
            if (!actorsById.TryGetValue(id, out var list)) return null;
            foreach (var t in list) if (t.gameObject.activeInHierarchy) return t;
            return null;
        }
        const int PassLinePoints = 32;
        const float BallRadius = 0.25f;

        static readonly float[] Speeds = { 0.25f, 0.5f, 1f, 2f, 4f };

        void Start()
        {
            cam = Camera.main;
            Application.runInBackground = true; // keep playing when the window loses focus
            string path = Path.Combine(Application.streamingAssetsPath, matchFile);
            try
            {
                var clean = new MatchReplayData.CleanOptions
                {
                    smoothWindow = smoothWindow,
                    maxPlayerSpeed = maxPlayerSpeed,
                    correctCameraShifts = correctCameraShifts,
                };
                data = MatchReplayData.FromJson(File.ReadAllText(path), origin, flipY, minTrackSamples, smoothRadius, clean);
            }
            catch (System.Exception e)
            {
                loadError = $"Could not load {path}\n{e.Message}";
                Debug.LogError(loadError);
                return;
            }

            Debug.Log($"[Replay] Loaded {matchFile}: {data.Duration:F1}s, {data.players.Count} player tracks " +
                      $"({data.droppedShortTracks} of {data.rawIdCount} IDs dropped as noise, " +
                      $"{data.players.FindAll(p => p.goalkeeper).Count} goalkeepers), " +
                      $"{data.ball.Count} ball samples, origin={(data.cornerOrigin ? "corner" : "center")}");
            Debug.Log($"[Replay] Cleanup: {data.cameraShiftsCorrected} camera-jump frames corrected, " +
                      $"{data.spikesRemoved} glitch samples removed, {data.idSwapsSplit} ID swaps split");

            PitchBuilder.Build(transform, data.length, data.width);
            stadium = GetComponent<StadiumBuilder>();
            if (stadium != null && stadium.enabled) stadium.Build(data.length, data.width, cam, homeColor, awayColor);
            else stadium = null;
            BuildActors();
            time = data.startTime;
            if (cam != null) SnapCamera();
        }

        void BuildActors()
        {
            var homeMat = PitchBuilder.MakeMaterial(homeColor);
            var awayMat = PitchBuilder.MakeMaterial(awayColor);
            var refMat = PitchBuilder.MakeMaterial(new Color(0.08f, 0.08f, 0.09f));
            var homeGkMat = PitchBuilder.MakeMaterial(homeKeeperColor);
            var awayGkMat = PitchBuilder.MakeMaterial(awayKeeperColor);
            var trimMat = PitchBuilder.MakeMaterial(new Color(0.95f, 0.95f, 0.95f));
            var refTrimMat = PitchBuilder.MakeMaterial(new Color(1f, 0.85f, 0.1f));
            var actors = new GameObject("Players").transform;
            actors.SetParent(transform, false);

            foreach (var tr in data.players)
            {
                if (estimateKeepers && tr.goalkeeper) continue; // keepers get their own handling below
                var mat = tr.referee ? refMat
                        : tr.goalkeeper ? (tr.team == 1 ? awayGkMat : homeGkMat)
                        : (tr.team == 1 ? awayMat : homeMat);
                string kind = tr.referee ? "Referee" : tr.goalkeeper ? "GK" : "Player";
                var body = PlayerBody.Create($"{kind} {tr.id}", actors, mat, tr.referee ? refTrimMat : trimMat,
                                             tr.referee ? "REF" : tr.id.ToString(), showLabels);
                body.SetActive(false);
                players.Add((tr, body));
            }

            if (estimateKeepers) BuildKeepers(actors);

            var ballMat = PitchBuilder.MakeMaterial(Color.white);
            ballGo = PitchBuilder.Primitive(PrimitiveType.Sphere, "Ball", transform, ballMat).transform;
            ballGo.localScale = Vector3.one * BallRadius * 2f; // real ball is 0.22m; enlarged so it reads from a distance
            ballGo.gameObject.SetActive(false);

            // Shadow on the grass under the ball, so height is readable
            ballShadowMat = PitchBuilder.MakeTransparentMaterial(new Color(0f, 0f, 0f, 0.45f));
            ballShadow = PitchBuilder.Primitive(PrimitiveType.Cylinder, "Ball Shadow", transform, ballShadowMat).transform;
            ballShadow.localScale = new Vector3(0.6f, 0.005f, 0.6f);
            ballShadow.gameObject.SetActive(false);

            // Ring under the player in possession
            var ringMat = PitchBuilder.MakeTransparentMaterial(new Color(1f, 0.95f, 0.3f, 0.8f));
            ownerRing = PitchBuilder.Primitive(PrimitiveType.Cylinder, "Possession Ring", transform, ringMat).transform;
            ownerRing.localScale = new Vector3(1.8f, 0.01f, 1.8f);
            ownerRing.gameObject.SetActive(false);

            // Arc drawn while a pass is in the air
            var lineGo = new GameObject("Pass Line");
            lineGo.transform.SetParent(transform, false);
            passLine = lineGo.AddComponent<LineRenderer>();
            passLine.useWorldSpace = false;
            passLine.positionCount = PassLinePoints;
            passLine.widthMultiplier = 0.25f;
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            passLine.material = unlit != null ? new Material(unlit) : PitchBuilder.MakeTransparentMaterial(Color.white);
            if (unlit != null)
            {
                // Transparent unlit so the arc stays pure white regardless of lighting
                var m = passLine.material;
                m.SetFloat("_Surface", 1f);
                m.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
                m.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                m.SetFloat("_ZWrite", 0f);
                m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            }
            passLine.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            passLine.enabled = false;

            foreach (var (track, body) in players) RegisterActor(track.id, body.root);
            foreach (var k in keepers) if (k.track != null) RegisterActor(k.track.id, k.body.root);
        }

        void BuildKeepers(Transform parent)
        {
            // Stitch every keeper-labelled piece of a team into one track (keepers don't swap with anyone)
            var tracks = new Track[2];
            for (int team = 0; team < 2; team++)
            {
                var pieces = data.players.FindAll(p => p.goalkeeper && p.team == team);
                if (pieces.Count == 0) continue;
                var samples = new List<(float t, Vector2 p)>();
                foreach (var piece in pieces)
                    for (int i = 0; i < piece.Count; i++) samples.Add((piece.times[i], piece.positions[i]));
                samples.Sort((a, b) => a.t.CompareTo(b.t));
                var merged = new Track { id = pieces[0].id, team = team, goalkeeper = true };
                foreach (var s in samples)
                    if (merged.Count == 0 || s.t > merged.End) merged.Add(s.t, s.p);
                tracks[team] = merged;
            }

            // Which goal does each team defend?
            int homeSide = homeDefendsWest ? -1 : 1;
            if (autoDetectSides)
            {
                if (tracks[0] != null) homeSide = tracks[0].AverageX() < 0 ? -1 : 1;
                else if (tracks[1] != null) homeSide = tracks[1].AverageX() < 0 ? 1 : -1;
            }

            for (int team = 0; team < 2; team++)
            {
                var k = new Keeper
                {
                    team = team,
                    side = team == 0 ? homeSide : -homeSide,
                    track = tracks[team],
                    color = team == 0 ? homeKeeperColor : awayKeeperColor,
                };
                k.mat = PitchBuilder.MakeTransparentMaterial(k.color);
                k.trimMat = PitchBuilder.MakeTransparentMaterial(new Color(0.12f, 0.12f, 0.15f));
                k.body = PlayerBody.Create($"Keeper team {team}", parent, k.mat, k.trimMat, "", showLabels);
                k.label = k.body.label;

                keepers.Add(k);
                Debug.Log($"[Replay] Team {team} keeper defends the {(k.side < 0 ? "west (-x)" : "east (+x)")} goal, " +
                          (k.track != null ? $"tracked as ID {k.track.id} for {k.track.Count} frames" : "never tracked: fully estimated"));
            }
        }

        /// <summary>Where a keeper would typically stand: on the line from the goal center to the ball.</summary>
        Vector2 EstimateKeeper(Keeper k, Vector2 ball)
        {
            var goal = new Vector2(k.side * data.length / 2f, 0f);
            Vector2 toBall = ball - goal;
            float dist = toBall.magnitude;
            if (dist < 0.01f) { toBall = new Vector2(-k.side, 0f); dist = 1f; }

            // Further off the line when play is far away; close down when the ball is near
            float depth = Mathf.Lerp(keeperMinDepth, keeperMaxDepth, Mathf.Clamp01((dist - 15f) / 45f));
            depth = Mathf.Min(depth, dist * 0.6f);
            Vector2 p = goal + toBall / dist * depth;

            float minFromLine = 0.5f;
            p.x = k.side > 0 ? Mathf.Min(p.x, goal.x - minFromLine) : Mathf.Max(p.x, goal.x + minFromLine);
            p.y = Mathf.Clamp(p.y, -12f, 12f);
            return p;
        }

        void UpdateKeepers(float dt)
        {
            if (keepers.Count == 0) return;
            Vector2 BallAt(float t) => data.ball.Count > 0 ? data.ball.SampleClamped(t) : Vector2.zero;
            Vector2 ball = BallAt(time);
            float follow = 1f - Mathf.Exp(-1.5f * dt); // slow smoothing so estimates drift rather than snap

            foreach (var k in keepers)
            {
                Vector2 target;
                float w = 0f;
                if (k.track != null)
                {
                    // Anchor on the real track: inside it this is the (interpolated) tracked position,
                    // outside it the first/last place he was actually seen.
                    float anchorT = Mathf.Clamp(time, k.track.Start, k.track.End);
                    Vector2 anchor = k.track.SampleClamped(time);

                    // While unseen, nudge him slightly the way a keeper would react to the ball since then,
                    // capped so he never wanders far from where he'll be seen again.
                    Vector2 shift = EstimateKeeper(k, ball) - EstimateKeeper(k, BallAt(anchorT));
                    Vector2 drift = Vector2.ClampMagnitude(shift * keeperDriftFactor, keeperMaxDrift);
                    k.smoothedEstimate = snapKeepers ? drift : Vector2.Lerp(k.smoothedEstimate, drift, follow);

                    float d = k.track.TimeToNearestSample(time);
                    w = 1f - Mathf.Clamp01((d - playerMaxGap * 0.5f) / Mathf.Max(0.01f, keeperBlendTime));
                    w = w * w * (3f - 2f * w); // smoothstep

                    // Real position plus a drift that fades out as tracking takes over
                    target = anchor + k.smoothedEstimate * (1f - w);
                }
                else
                {
                    // Never seen: follow the typical keeper position, slowly
                    Vector2 est = EstimateKeeper(k, ball);
                    k.smoothedEstimate = snapKeepers ? est : Vector2.Lerp(k.smoothedEstimate, est, follow);
                    target = k.smoothedEstimate;
                }

                // Speed cap: the displayed keeper walks toward the target, never sprints
                if (snapKeepers || dt <= 0f) k.displayPos = target;
                else
                {
                    // Tracked keepers may move quickly (dives); only estimated movement is held to a walk
                    float maxSpeed = Mathf.Lerp(keeperMaxSpeed, 9f, w);
                    k.displayPos = Vector2.MoveTowards(k.displayPos, target, maxSpeed * dt);
                }

                k.realWeight = w;

                // Keeper velocity from how the displayed position actually moves (smoothed)
                if (snapKeepers || dt <= 0f)
                {
                    if (snapKeepers) k.vel = Vector2.zero;
                }
                else
                {
                    var inst = (k.displayPos - k.prevDisplayPos) / dt;
                    k.vel = Vector2.Lerp(k.vel, inst, 1f - Mathf.Exp(-6f * dt));
                }
                k.prevDisplayPos = k.displayPos;
                k.body.UpdateMotion(k.displayPos, k.vel, ball, 1f, dt, snapKeepers, bodyMotion);

                var c = k.color;
                c.a = Mathf.Lerp(estimatedAlpha, 1f, w);
                k.mat.color = c;
                var tc = k.trimMat.color;
                tc.a = c.a;
                k.trimMat.color = tc;
                if (k.label != null)
                {
                    k.label.text = w > 0.5f ? $"GK {k.track.id}" : "GK (est.)";
                    k.label.color = new Color(1f, 1f, 1f, Mathf.Lerp(0.6f, 1f, w));
                }
            }
            snapKeepers = false;
        }

        void Update()
        {
            if (data == null) return;
            HandleInput();

            motionDt = playing ? Time.deltaTime * speed : 0f;
            if (playing)
            {
                time += motionDt;
                if (time > data.endTime)
                {
                    if (loop) { time = data.startTime; motionSnap = true; snapKeepers = true; }
                    else { time = data.endTime; playing = false; }
                }
            }
            ApplyTime();
            UpdateKeepers(motionDt);
            UpdatePossessionAndPasses();
            motionSnap = false;
        }

        void HandleInput()
        {
            var kb = Keyboard.current;
            if (kb == null) return;
            if (kb.spaceKey.wasPressedThisFrame) playing = !playing;
            if (kb.rightArrowKey.wasPressedThisFrame) Seek(time + 1f);
            if (kb.leftArrowKey.wasPressedThisFrame) Seek(time - 1f);
            if (kb.rKey.wasPressedThisFrame) Seek(data.startTime);
            if (kb.rightBracketKey.wasPressedThisFrame) StepSpeed(1);
            if (kb.leftBracketKey.wasPressedThisFrame) StepSpeed(-1);
            if (kb.digit1Key.wasPressedThisFrame) cameraMode = CameraMode.Broadcast;
            if (kb.digit2Key.wasPressedThisFrame) cameraMode = CameraMode.TopDown;
            if (kb.digit3Key.wasPressedThisFrame) cameraMode = CameraMode.FollowBall;
        }

        public void Seek(float t)
        {
            if (data == null) return;
            time = Mathf.Clamp(t, data.startTime, data.endTime);
            motionSnap = true;
            motionDt = 0f;
            ApplyTime();
            snapKeepers = true;
            UpdateKeepers(0f);
            UpdatePossessionAndPasses();
            motionSnap = false;
            if (cam != null) SnapCamera();
        }

        public float CurrentTime => time;
        public bool IsPlaying { get => playing; set => playing = value; }

        void StepSpeed(int dir)
        {
            int idx = System.Array.IndexOf(Speeds, speed);
            if (idx < 0) idx = 2;
            speed = Speeds[Mathf.Clamp(idx + dir, 0, Speeds.Length - 1)];
        }

        void ApplyTime()
        {
            visibleCount = 0;

            // Ball first, so players can turn to face its current position
            if (data.ball.Count > 0 && data.ball.TrySample(time, ballMaxGap, out var ballNow)) { lastBall = ballNow; ballVisible = true; }
            else ballVisible = false;

            foreach (var (track, body) in players)
            {
                bool vis = track.TrySample(time, playerMaxGap, out var p);
                bool justAppeared = vis && !body.Active;
                body.SetActive(vis);
                if (!vis) continue;
                if (!track.referee) visibleCount++;

                // Grow in / shrink out at the ends of a track instead of popping
                // (except at the very start/end of the clip, where tracks are simply cut off)
                float fade = 1f;
                if (fadeTime > 0f)
                {
                    float fin = track.Start - data.startTime < 0.05f ? 1f : (time - track.Start) / fadeTime;
                    float fout = data.endTime - track.End < 0.05f ? 1f : (track.End - time) / fadeTime;
                    fade = Mathf.Clamp01(Mathf.Min(fin, fout));
                    fade = fade * fade * (3f - 2f * fade);
                }

                Vector2 vel = track.VelocityAt(time, bodyMotion.velocityWindow);
                body.UpdateMotion(p, vel, lastBall, fade, motionDt, motionSnap || justAppeared, bodyMotion);
            }

            if (ballGo.gameObject.activeSelf != ballVisible) ballGo.gameObject.SetActive(ballVisible);
            if (ballVisible)
            {
                lastBallHeight = data.ball.HeightAt(time);
                ballGo.localPosition = new Vector3(lastBall.x, BallRadius + lastBallHeight, lastBall.y);
            }

            // Shadow grows and fades as the ball rises
            if (ballShadow.gameObject.activeSelf != ballVisible) ballShadow.gameObject.SetActive(ballVisible);
            if (ballVisible)
            {
                float h01 = Mathf.Clamp01(lastBallHeight / 10f);
                float s = Mathf.Lerp(0.6f, 1.1f, h01);
                ballShadow.localPosition = new Vector3(lastBall.x, 0.04f, lastBall.y);
                ballShadow.localScale = new Vector3(s, 0.005f, s);
                ballShadowMat.color = new Color(0f, 0f, 0f, Mathf.Lerp(0.45f, 0.15f, h01));
            }
        }

        void UpdatePossessionAndPasses()
        {
            // Possession ring follows the owner while they're on screen
            int owner = data.OwnerAt(time);
            var ownerGo = owner >= 0 ? ActiveActor(owner) : null;
            bool showRing = ownerGo != null;
            if (ownerRing.gameObject.activeSelf != showRing) ownerRing.gameObject.SetActive(showRing);
            if (showRing)
            {
                var p = ownerGo.localPosition;
                ownerRing.localPosition = new Vector3(p.x, 0.05f, p.z);
            }

            // Pass arc: visible while the pass is travelling and briefly after it arrives
            MatchEvent pass = null;
            foreach (var e in data.events)
                if (e.type == "pass" && e.start != null && e.end != null && time >= e.t_start && time <= e.t_end + 1f) pass = e;
            passLine.enabled = pass != null;
            if (pass != null)
            {
                var a = new Vector3(pass.start[0], 0.1f, pass.start[1]);
                var b = new Vector3(pass.end[0], 0.1f, pass.end[1]);
                float peak = pass.aerial ? Mathf.Max(0.5f, pass.max_height) : 0f;
                for (int i = 0; i < PassLinePoints; i++)
                {
                    float u = i / (float)(PassLinePoints - 1);
                    var p = Vector3.Lerp(a, b, u);
                    p.y += 4f * peak * u * (1f - u); // parabola peaking at max_height
                    passLine.SetPosition(i, p);
                }
                float fade = time > pass.t_end ? 1f - (time - pass.t_end) : 1f;
                var c = pass.completed ? new Color(1f, 1f, 1f, 0.8f * fade) : new Color(1f, 0.3f, 0.3f, 0.8f * fade);
                passLine.startColor = passLine.endColor = c;
                passLine.material.color = c;
            }
        }

        void LateUpdate()
        {
            if (data == null || cam == null) return;
            GetCameraTarget(out var pos, out var look);
            float k = 1f - Mathf.Exp(-3f * Time.deltaTime);
            cam.transform.position = Vector3.Lerp(cam.transform.position, pos, k);
            var targetRot = Quaternion.LookRotation(look - pos);
            cam.transform.rotation = Quaternion.Slerp(cam.transform.rotation, targetRot, k);

            // Billboard labels toward the camera
            foreach (var (_, body) in players)
                if (body.Active && body.label != null) body.label.transform.rotation = cam.transform.rotation;
            foreach (var keeper in keepers)
                if (keeper.label != null) keeper.label.transform.rotation = cam.transform.rotation;
        }

        void SnapCamera()
        {
            GetCameraTarget(out var pos, out var look);
            cam.transform.position = pos;
            cam.transform.rotation = Quaternion.LookRotation(look - pos);
        }

        void GetCameraTarget(out Vector3 pos, out Vector3 look)
        {
            Vector2 focus = lastBall;
            float hw = data.width / 2f;
            switch (cameraMode)
            {
                case CameraMode.TopDown:
                    pos = new Vector3(0f, 80f, -0.01f);
                    look = Vector3.zero;
                    break;
                case CameraMode.FollowBall:
                    // Stay in front of the near stand when there is one
                    float minZ = stadium != null ? -stadium.FrontHalfWidth + 1f : float.MinValue;
                    pos = new Vector3(focus.x, 14f, Mathf.Max(focus.y - 20f, minZ));
                    look = new Vector3(focus.x, 0f, focus.y);
                    break;
                default: // Broadcast: gantry at the front of the near stand, panning with play
                    if (stadium != null)
                        pos = new Vector3(focus.x * 0.6f, 21f, -stadium.FrontHalfWidth - 6f);
                    else
                        pos = new Vector3(focus.x * 0.7f, 24f, -hw - 28f);
                    look = new Vector3(focus.x * 0.85f, 0f, focus.y * 0.3f);
                    break;
            }
        }

        GUIStyle feedStyle, feedActiveStyle;

        void DrawEventFeed()
        {
            if (data.events.Count == 0) return;
            if (feedStyle == null)
            {
                feedStyle = new GUIStyle(GUI.skin.label) { fontSize = 13, richText = true };
                feedActiveStyle = new GUIStyle(feedStyle) { fontStyle = FontStyle.Bold };
                feedActiveStyle.normal.textColor = new Color(1f, 0.95f, 0.4f);
            }

            var lines = new List<(string text, bool active)>();
            foreach (var e in data.events)
            {
                if (e.StartTime > time) continue;
                bool active = time <= e.EndTime;
                float rel = e.StartTime - data.startTime;
                string text = e.type switch
                {
                    "pass" => $"{rel,5:F1}s  Pass  #{e.from} → #{e.to}   {e.length:F1} m{(e.aerial ? $", aerial ({e.max_height:F1} m high)" : "")}  {(e.completed ? "✓" : "✗")}",
                    "touch" => $"{rel,5:F1}s  Touch  #{e.player}{(e.duration > 1f ? $"  (on ball {e.duration:F1}s)" : "")}",
                    _ => $"{rel,5:F1}s  {e.type}",
                };
                lines.Add((text, active));
            }
            if (lines.Count == 0) lines.Add(("Waiting for first event…", false));
            if (lines.Count > 6) lines.RemoveRange(0, lines.Count - 6);

            float w = 420f, lh = 20f;
            var box = new Rect(10, 10, w, 28 + lines.Count * lh);
            GUI.Box(box, "Match events");
            for (int i = 0; i < lines.Count; i++)
                GUI.Label(new Rect(box.x + 10, box.y + 24 + i * lh, w - 20, lh), lines[i].text, lines[i].active ? feedActiveStyle : feedStyle);
        }

        void OnGUI()
        {
            if (loadError != null)
            {
                GUI.Box(new Rect(10, 10, 600, 60), loadError);
                return;
            }
            if (data == null) return;

            DrawEventFeed();

            const float h = 64f;
            var area = new Rect(10, Screen.height - h - 10, Screen.width - 20, h);
            GUI.Box(area, GUIContent.none);
            GUILayout.BeginArea(new Rect(area.x + 8, area.y + 6, area.width - 16, area.height - 12));

            GUILayout.BeginHorizontal();
            if (GUILayout.Button(playing ? "Pause" : "Play", GUILayout.Width(70))) playing = !playing;
            if (GUILayout.Button("Restart", GUILayout.Width(70))) Seek(data.startTime);
            GUILayout.Label($"{time - data.startTime:F1}s / {data.Duration:F1}s", GUILayout.Width(110));
            float t = GUILayout.HorizontalSlider(time, data.startTime, data.endTime, GUILayout.ExpandWidth(true));
            if (!Mathf.Approximately(t, time)) Seek(t);
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            GUILayout.Label("Speed", GUILayout.Width(45));
            foreach (var s in Speeds)
                if (GUILayout.Toggle(Mathf.Approximately(speed, s), $"{s}x", "Button", GUILayout.Width(48))) speed = s;
            GUILayout.Space(16);
            GUILayout.Label("Camera", GUILayout.Width(55));
            if (GUILayout.Toggle(cameraMode == CameraMode.Broadcast, "Broadcast", "Button", GUILayout.Width(85))) cameraMode = CameraMode.Broadcast;
            if (GUILayout.Toggle(cameraMode == CameraMode.TopDown, "Top-down", "Button", GUILayout.Width(85))) cameraMode = CameraMode.TopDown;
            if (GUILayout.Toggle(cameraMode == CameraMode.FollowBall, "Follow ball", "Button", GUILayout.Width(85))) cameraMode = CameraMode.FollowBall;
            GUILayout.FlexibleSpace();
            int keepersReal = keepers.FindAll(k => k.realWeight > 0.5f).Count;
            string keeperInfo = keepers.Count > 0 ? $"   Keepers: {keepersReal} tracked, {keepers.Count - keepersReal} estimated" : "";
            GUILayout.Label($"Players tracked: {visibleCount}{keeperInfo}   Ball: {(ballVisible ? "yes" : "lost")}");
            GUILayout.EndHorizontal();

            GUILayout.EndArea();
        }
    }
}
