using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.InputSystem;

namespace SoccerReplay
{
    /// <summary>
    /// Loads a tracking JSON from StreamingAssets and replays it in 3D.
    /// Controls: Space play/pause · Left/Right skip 1s · [ ] speed · 1/2/3 camera, 4 free camera · L load match · F formation lines · R restart
    /// </summary>
    public class ReplayController : MonoBehaviour
    {
        public enum CameraMode { Broadcast, TopDown, FollowBall, Free }

        [Header("Free camera (key 4)")]
        [Tooltip("Fly speed (m/s); mouse wheel changes it while flying, Shift triples it")]
        public float freeMoveSpeed = 12f;
        [Tooltip("Degrees of turn per pixel of mouse movement (hold the right mouse button to look)")]
        public float freeLookSensitivity = 0.15f;
        float freeYaw, freePitch;

        [Header("Data")]
        [Tooltip("Path inside Assets/StreamingAssets")]
        public string matchFile = "matches/match_events_v2.json";
        [Tooltip("Off: pressing Play shows the empty stadium and waits for Load match… (L). On: plays Match File straight away")]
        public bool loadMatchOnStart = false;
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
        public float ballMaxGap = 5.0f;
        [Tooltip("Ball movement faster than this (m/s) is treated as a false detection and removed")]
        public float ballMaxSpeed = 40f;
        [Tooltip("When the ball isn't detected but a player owns it, place it at their feet")]
        public bool fillBallFromOwner = true;
        [Tooltip("Drop passes/touches that don't fit the cleaned ball path (phantom passes)")]
        public bool dropImplausibleEvents = true;
        [Tooltip("Lobs are drawn no higher than this × what gravity allows for their flight time (1 = strictly physical)")]
        [Range(1f, 4f)] public float lobGravityCap = 2f;
        [Tooltip("During fitted ball flights (ball_flights), draw the ball on the smooth arc instead of frame-by-frame points")]
        public bool useFittedFlights = true;

        [Header("Look")]
        public Color homeColor = new Color(0.95f, 0.95f, 0.95f);   // white kit (team 0)
        public Color awayColor = new Color(0.06f, 0.06f, 0.07f);   // black kit (team 1)
        [Tooltip("Referee shirt (yellow, so it doesn't clash with a black or white kit)")]
        public Color refereeColor = new Color(1f, 0.85f, 0.1f);
        public Color homeKeeperColor = new Color(1f, 0.85f, 0.1f);
        public Color awayKeeperColor = new Color(0.95f, 0.3f, 0.85f);

        [Tooltip("Use the kit colors from the file's \"teams\" block when it has one")]
        public bool useKitColors = true;

        [Header("Off-camera players")]
        [Tooltip("Keep showing outfield players while they're off camera, estimated and drawn see-through (like the keepers)")]
        public bool estimateOffscreenPlayers = true;
        [Tooltip("Longest gap (s) between two tracks that can still be joined as the same player")]
        public float stitchMaxGap = 8f;
        [Tooltip("Speed (m/s) a player could plausibly cover while unseen; used to decide whether two tracks can be the same person")]
        public float stitchRunSpeed = 7f;
        [Tooltip("How long (s) to keep showing a player before their first / after their last sighting")]
        public float ghostHoldTime = 8f;
        [Tooltip("How much an unseen player moves with their team's overall shape (0 = stands still)")]
        [Range(0f, 1f)] public float ghostTeamFollow = 0.6f;
        [Tooltip("Furthest (m) an unseen player drifts from where they were last/next seen")]
        public float ghostMaxDrift = 10f;
        [Tooltip("Never show more than this many outfield players per team (real + estimated)")]
        public int maxOutfieldPerTeam = 10;
        [Tooltip("Top speed (m/s) of an estimated player's displayed position")]
        public float ghostMaxSpeed = 8f;
        [Tooltip("Seconds over which a player fades between solid (tracked) and see-through (estimated)")]
        public float ghostBlendTime = 1f;
        [Range(0.1f, 1f)] public float ghostAlpha = 0.4f;

        class Identity
        {
            public readonly List<Track> pieces = new List<Track>();   // same person, in time order
            public int team;
            public bool referee;
            public PlayerBody body;
            public Material solidMat, solidTrim, ghostMat, ghostTrim, ghostShorts;
            public Color color, trimColor;
            public Vector2 target, displayPos, prevDisplayPos, vel;
            public Track currentPiece;     // piece being tracked right now (null when estimated)
            public bool real, candidate;   // tracked now / worth showing as an estimate
            public float confidence;       // seconds to the nearest real sighting (0 when tracked)
            public float realWeight;       // 1 = solid, 0 = fully see-through
            public float presence;         // 0..1 fade for appearing / being culled
            public List<ScheduledAction> actions;  // this person's actions across all their track pieces
        }
        readonly List<Identity> identities = new List<Identity>();
        int estimatedCount;
        float[][] centroidX, centroidY;
        const float CentroidStep = 0.1f;

        [Header("Animated players")]
        [Tooltip("Use the animated humanoid model instead of capsules")]
        public bool useAnimatedPlayers = true;
        [Tooltip("Humanoid character model (Y Bot)")]
        public GameObject playerModel;
        [Tooltip("Animator for outfield players and referees (idle + 8-direction jog blend tree)")]
        public RuntimeAnimatorController playerAnimator;
        [Tooltip("Animator for goalkeepers (goalkeeper idle); falls back to the player animator")]
        public RuntimeAnimatorController keeperAnimator;

        [Header("Actions (kicks, receives, headers, keeper)")]
        [Tooltip("Play action animations from the match events")]
        public bool playActions = true;
        [Tooltip("Filled in by Soccer Replay > Setup Animated Players")]
        public List<ActionClip> actionClips = new List<ActionClip>();
        public float actionBlendIn = 0.15f;
        public float actionBlendOut = 0.25f;
        [Tooltip("Passer faster than this (m/s) uses the running kick")]
        public float runningKickSpeed = 2f;
        [Tooltip("Passes longer than this (m), or aerial passes, use the long kick")]
        public float longPassLength = 30f;
        [Tooltip("A ball arriving at least this high (m) at a player is headed")]
        public float headerMinHeight = 1.2f;
        [Tooltip("A ball arriving between this and header height is controlled (receive)")]
        public float chestMinHeight = 0.5f;
        [Tooltip("Shift players so the kicking foot / head / hands meet the ball at contact")]
        public bool alignContact = true;
        [Tooltip("Largest shift (m) used to bring the foot to the ball")]
        public float maxContactNudge = 1.5f;
        [Tooltip("Seconds either side of contact over which the shift blends in and out")]
        public float contactNudgeWindow = 0.45f;
        [Tooltip("Use the mirrored (other-foot) clip when the ball is on the player's other side")]
        public bool mirrorForBallSide = true;

        Dictionary<int, List<ScheduledAction>> actionsByPlayer = new Dictionary<int, List<ScheduledAction>>();
        int actionsPlaying;

        GameObject ModelOrNull => useAnimatedPlayers && playerModel != null && playerAnimator != null ? playerModel : null;
        RuntimeAnimatorController KeeperController => keeperAnimator != null ? keeperAnimator : playerAnimator;

        [Header("Body motion")]
        public BodyMotionSettings bodyMotion = new BodyMotionSettings();

        // Team colors actually used this session (kit colors from the file, or homeColor/awayColor)
        Color team0Color, team1Color;

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
        [Tooltip("A 'goalkeeper' whose average position is further than this from his own goal line is treated as a misclassified outfield player")]
        public float keeperMaxDistanceFromGoal = 25f;

        // Keeper plan (decided before players are built)
        readonly HashSet<Track> keeperPieces = new HashSet<Track>();
        readonly Track[] keeperTrackByTeam = new Track[2];
        int plannedHomeSide = -1;
        int rejectedKeepers;

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
            public List<ScheduledAction> actions;
        }
        readonly List<Keeper> keepers = new List<Keeper>();
        bool snapKeepers = true;
        public bool showLabels = true;

        [Header("Camera")]
        [Tooltip("Seconds the camera takes to settle on where play is (higher = calmer)")]
        public float cameraFocusSmoothing = 0.6f;
        [Tooltip("Fastest the camera's focus point may travel (m/s), so it can't whip across the pitch")]
        public float cameraFocusMaxSpeed = 18f;
        Vector2 cameraFocus, cameraFocusVel;
        bool cameraFocusInit;

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
        [Tooltip("Displayed ball diameter (m). A real ball is 0.22 m; slightly enlarged so it reads from the broadcast camera")]
        public float ballDiameter = 0.4f;
        float BallRadius => ballDiameter * 0.5f;

        static readonly float[] Speeds = { 0.25f, 0.5f, 1f, 2f, 4f };

        // ---------------- Loading a match from disk (Load match… / L) ----------------

        /// <summary>
        /// File picked with Load match…, kept across the scene reload that rebuilds the replay.
        /// Static, so it lasts for this play session only; the scene's Match File is unchanged.
        /// </summary>
        static string pickedMatchPath;
        static string pickedMatchNote;
        string loadedName;

        /// <summary>Open a native file dialog for the pipeline's JSON (match.json and/or match_events.json) and replay it.</summary>
        public void OpenMatchDialog()
        {
            string dir = "";
            try { dir = PlayerPrefs.GetString("Replay.LastDir", ""); } catch { }
            if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) dir = Application.streamingAssetsPath;
            var picked = SFB.StandaloneFileBrowser.OpenFilePanel(
                "Load match: match_events.json and/or match.json", dir,
                new[] { new SFB.ExtensionFilter("Match JSON", "json") }, true);
            if (picked == null || picked.Length == 0) return;   // cancelled
            LoadMatchFiles(picked);
        }

        /// <summary>Replay the given file(s) from disk (absolute paths): match_events.json and/or match.json.</summary>
        public void LoadMatchFiles(params string[] picked)
        {
            string chosen = ChooseMatchFile(picked, out string note);
            if (chosen == null) { loadError = note; return; }
            try { PlayerPrefs.SetString("Replay.LastDir", Path.GetDirectoryName(chosen)); PlayerPrefs.Save(); } catch { }
            pickedMatchPath = chosen;
            pickedMatchNote = note;
            Debug.Log($"[Replay] Loading {chosen} ({note})");
            // Reload this scene so the replay rebuilds from scratch with the new file
#if UNITY_EDITOR
            UnityEditor.SceneManagement.EditorSceneManager.LoadSceneInPlayMode(gameObject.scene.path,
                new UnityEngine.SceneManagement.LoadSceneParameters(UnityEngine.SceneManagement.LoadSceneMode.Single));
#else
            UnityEngine.SceneManagement.SceneManager.LoadScene(gameObject.scene.buildIndex);
#endif
        }

        // Each press of Play starts from the scene's own Match File (also when domain reload is disabled)
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetPickedMatch() { pickedMatchPath = null; pickedMatchNote = null; }

        /// <summary>
        /// The pipeline writes two files: match.json (tracking) and match_events.json (the same tracking plus
        /// possession, passes, ball height and flights). Whichever of them are picked, use the one with events,
        /// looking next to the picked file for its partner, and fall back to tracking only.
        /// </summary>
        static string ChooseMatchFile(string[] picked, out string note)
        {
            var candidates = new List<string>();
            foreach (var p in picked)
            {
                if (string.IsNullOrEmpty(p) || !File.Exists(p)) continue;
                candidates.Add(p);
                string folder = Path.GetDirectoryName(p);
                string stem = Path.GetFileNameWithoutExtension(p);
                foreach (var partner in new[] { "match_events.json", stem + "_events.json", "match.json" })
                {
                    string q = Path.Combine(folder, partner);
                    if (File.Exists(q) && !candidates.Contains(q)) candidates.Add(q);
                }
            }
            if (candidates.Count == 0) { note = "No readable JSON file was picked."; return null; }

            foreach (var c in candidates)
            {
                string text;
                try { text = File.ReadAllText(c); } catch { continue; }
                if (text.Contains("\"events\"") || text.Contains("\"ball_flights\""))
                {
                    bool fromPicked = System.Array.IndexOf(picked, c) >= 0;
                    note = fromPicked ? "with events" : $"with events, found next to {Path.GetFileName(picked[0])}";
                    return c;
                }
            }
            note = "tracking only, no events file found";
            return candidates[0];
        }

        void Start()
        {
            cam = Camera.main;
            Application.runInBackground = true; // keep playing when the window loses focus
            SetOverlayVisible(false);   // the stats overlay appears once a match is loaded
            if (pickedMatchPath == null && (!loadMatchOnStart || string.IsNullOrEmpty(matchFile)))
            {
                BuildEmptyStadium();   // wait for Load match…
                return;
            }
            string path = pickedMatchPath ?? Path.Combine(Application.streamingAssetsPath, matchFile);
            loadedName = pickedMatchPath != null ? $"{Path.GetFileName(Path.GetDirectoryName(pickedMatchPath))}/{Path.GetFileName(pickedMatchPath)} ({pickedMatchNote})" : matchFile;
            try
            {
                var clean = new MatchReplayData.CleanOptions
                {
                    smoothWindow = smoothWindow,
                    maxPlayerSpeed = maxPlayerSpeed,
                    correctCameraShifts = correctCameraShifts,
                    ballMaxSpeed = ballMaxSpeed,
                    fillBallFromOwner = fillBallFromOwner,
                    dropImplausibleEvents = dropImplausibleEvents,
                    lobGravityCap = lobGravityCap,
                };
                data = MatchReplayData.FromJson(File.ReadAllText(path), origin, flipY, minTrackSamples, smoothRadius, clean);
            }
            catch (System.Exception e)
            {
                loadError = $"Could not load {path}\n{e.Message}";
                Debug.LogError(loadError);
                return;
            }

            Debug.Log($"[Replay] Loaded {loadedName}: {data.Duration:F1}s, {data.players.Count} player tracks " +
                      $"({data.droppedShortTracks} of {data.rawIdCount} IDs dropped as noise, " +
                      $"{data.players.FindAll(p => p.goalkeeper).Count} goalkeepers), " +
                      $"{data.ball.Count} ball samples, origin={(data.cornerOrigin ? "corner" : "center")}");
            Debug.Log($"[Replay] Cleanup: {data.cameraShiftsCorrected} camera-jump frames corrected, " +
                      $"{data.spikesRemoved} glitch samples removed, {data.idSwapsSplit} ID swaps split");
            Debug.Log($"[Replay] Ball cleanup: {data.ballOutliersRemoved} false detections removed, " +
                      $"{data.ballSamplesFilled} gap samples filled from the player on the ball, {data.eventsDropped} phantom events dropped; " +
                      $"lobs: {data.flights.Count} arcs ({data.flightsAdded} added for aerial passes, {data.flightsCapped} height-capped)");
            LoadOverlay(path);

            bool kits = useKitColors && data.hasKitColors;
            team0Color = kits ? data.kitColors[0] : homeColor;
            team1Color = kits ? data.kitColors[1] : awayColor;
            if (kits) Debug.Log($"[Replay] Using kit colors from file: team 0 #{ColorUtility.ToHtmlStringRGB(team0Color)}, team 1 #{ColorUtility.ToHtmlStringRGB(team1Color)}");

            PitchBuilder.Build(transform, data.length, data.width);
            stadium = GetComponent<StadiumBuilder>();
            if (stadium != null && stadium.enabled) stadium.Build(data.length, data.width, cam, team0Color, team1Color);
            else stadium = null;
            BuildActors();
            BuildActionSchedule();
            time = data.startTime;
            if (cam != null) SnapCamera();
        }

        /// <summary>No match yet: show the pitch and stadium from the broadcast gantry, and the Load match prompt.</summary>
        void BuildEmptyStadium()
        {
            loadedName = null;
            const float length = 105f, width = 68f;
            PitchBuilder.Build(transform, length, width);
            stadium = GetComponent<StadiumBuilder>();
            if (stadium != null && stadium.enabled) stadium.Build(length, width, cam, homeColor, awayColor);
            else stadium = null;
            if (cam != null)
            {
                cam.transform.position = new Vector3(0f, 22f, -(width / 2f + 34f));
                cam.transform.rotation = Quaternion.LookRotation(new Vector3(0f, 0f, 0f) - cam.transform.position);
            }
        }

        void DrawStartPanel()
        {
            var r = new Rect(Screen.width / 2f - 210f, Screen.height / 2f - 70f, 420f, 130f);
            GUI.Box(r, "");
            GUILayout.BeginArea(new Rect(r.x + 16, r.y + 14, r.width - 32, r.height - 28));
            var title = new GUIStyle(GUI.skin.label) { fontSize = 18, fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleCenter };
            GUILayout.Label("No match loaded", title);
            GUILayout.Label("Pick the pipeline's match_events.json to replay it.",
                            new GUIStyle(GUI.skin.label) { alignment = TextAnchor.MiddleCenter, wordWrap = true });
            GUILayout.Space(8);
            if (GUILayout.Button("Load match…  (L)", GUILayout.Height(32))) OpenMatchDialog();
            GUILayout.EndArea();
        }

        void BuildActors()
        {
            var homeMat = PitchBuilder.MakeMaterial(team0Color);
            var awayMat = PitchBuilder.MakeMaterial(team1Color);
            var refMat = PitchBuilder.MakeMaterial(refereeColor);
            var homeGkMat = PitchBuilder.MakeMaterial(homeKeeperColor);
            var awayGkMat = PitchBuilder.MakeMaterial(awayKeeperColor);
            // Chest stripe that contrasts with each kit (dark on light kits, white on dark kits)
            Material TrimFor(Color kit) => PitchBuilder.MakeMaterial(kit.grayscale > 0.55f ? new Color(0.1f, 0.1f, 0.14f) : new Color(0.95f, 0.95f, 0.95f));
            var homeTrimMat = TrimFor(team0Color);
            var awayTrimMat = TrimFor(team1Color);
            var refTrimMat = PitchBuilder.MakeMaterial(new Color(0.08f, 0.08f, 0.09f));
            var actors = new GameObject("Players").transform;
            actors.SetParent(transform, false);

            if (estimateKeepers) PlanKeepers();

            if (estimateOffscreenPlayers)
            {
                // Join tracks into people and draw one body per person (solid when tracked, see-through when estimated)
                var pool = data.players.FindAll(p => !(estimateKeepers && keeperPieces.Contains(p)));
                BuildCentroids(pool);
                foreach (var chain in StitchIdentities(pool))
                {
                    var first = chain[0];
                    var id = new Identity { team = first.team, referee = first.referee };
                    id.pieces.AddRange(chain);
                    id.solidMat = first.referee ? refMat : first.goalkeeper ? (first.team == 1 ? awayGkMat : homeGkMat) : (first.team == 1 ? awayMat : homeMat);
                    id.solidTrim = first.referee ? refTrimMat : first.team == 1 ? awayTrimMat : homeTrimMat;
                    id.color = id.solidMat.color;
                    id.trimColor = id.solidTrim.color;
                    id.ghostMat = PitchBuilder.MakeTransparentMaterial(id.color);
                    id.ghostTrim = PitchBuilder.MakeTransparentMaterial(id.trimColor);
                    id.ghostShorts = PitchBuilder.MakeTransparentMaterial(new Color(0.1f, 0.1f, 0.14f));
                    string kind = first.referee ? "Referee" : "Person";
                    id.body = PlayerBody.Create($"{kind} {first.id}", actors, id.solidMat, id.solidTrim, "", showLabels,
                                                ModelOrNull, first.goalkeeper ? KeeperController : playerAnimator);
                    id.body.SetActive(false);
                    identities.Add(id);
                }
                Debug.Log($"[Replay] Off-camera players: {pool.Count} tracks joined into {identities.Count} people");
            }

            foreach (var tr in data.players)
            {
                if (estimateOffscreenPlayers) break;                                 // handled as identities above
                if (estimateKeepers && keeperPieces.Contains(tr)) continue; // validated keepers get their own handling below
                var mat = tr.referee ? refMat
                        : tr.goalkeeper ? (tr.team == 1 ? awayGkMat : homeGkMat)
                        : (tr.team == 1 ? awayMat : homeMat);
                var trim = tr.referee ? refTrimMat : tr.team == 1 ? awayTrimMat : homeTrimMat;
                string kind = tr.referee ? "Referee" : tr.goalkeeper ? "GK" : "Player";
                var body = PlayerBody.Create($"{kind} {tr.id}", actors, mat, trim,
                                             tr.referee ? "REF" : tr.id.ToString(), showLabels,
                                             ModelOrNull, tr.goalkeeper ? KeeperController : playerAnimator);
                body.SetActive(false);
                players.Add((tr, body));
            }

            if (estimateKeepers) BuildKeepers(actors);

            var ballMat = PitchBuilder.MakeMaterial(Color.white);
            ballGo = PitchBuilder.Primitive(PrimitiveType.Sphere, "Ball", transform, ballMat).transform;
            ballGo.localScale = Vector3.one * ballDiameter;
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
            foreach (var id in identities) foreach (var piece in id.pieces) RegisterActor(piece.id, id.body.root);
            foreach (var k in keepers) if (k.track != null) RegisterActor(k.track.id, k.body.root);
        }

        /// <summary>
        /// Measures each action clip on a hidden copy of the animated player: the contact moment is when the
        /// foot / head / hands move fastest relative to the hips, and the contact offset is where that limb is
        /// then, relative to the character. Done at runtime because humanoid poses only evaluate reliably here.
        /// </summary>
        void MeasureActionClips()
        {
            if (ModelOrNull == null || actionClips == null || actionClips.Count == 0) return;
            var go = Instantiate(playerModel, new Vector3(0f, -200f, 0f), Quaternion.identity);
            go.name = "Action Measurer";
            foreach (var r in go.GetComponentsInChildren<Renderer>()) r.enabled = false;
            var anim = go.GetComponent<Animator>();
            if (anim == null || anim.avatar == null) { Destroy(go); return; }
            anim.runtimeAnimatorController = playerAnimator;
            anim.applyRootMotion = false;
            anim.cullingMode = AnimatorCullingMode.AlwaysAnimate;
            anim.SetLayerWeight(1, 1f);
            var hips = anim.GetBoneTransform(HumanBodyBones.Hips);
            var report = new System.Text.StringBuilder();

            foreach (var c in actionClips)
            {
                if (c == null || c.clip == null) continue;
                var bones = c.contactPart == 0 ? new[] { HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot }
                          : c.contactPart == 1 ? new[] { HumanBodyBones.Head }
                          : new[] { HumanBodyBones.LeftHand, HumanBodyBones.RightHand };
                const float step = 1f / 60f;
                float len = c.clip.length, end = Mathf.Min(len, c.searchEnd);
                var prev = new Vector3[bones.Length];
                float bestSpeed = 0f, bestT = c.contact;
                for (int i = 0; i * step <= end; i++)
                {
                    float t = i * step;
                    anim.Play(c.StateHash, 1, t / len);
                    anim.Update(0f);
                    for (int b = 0; b < bones.Length; b++)
                    {
                        var bone = anim.GetBoneTransform(bones[b]);
                        if (bone == null || hips == null) continue;
                        var rel = bone.position - hips.position;
                        if (i > 0)
                        {
                            float speed = (rel - prev[b]).magnitude / step;
                            if (speed > bestSpeed)
                            {
                                bestSpeed = speed; bestT = t;
                                c.contactOffset = go.transform.InverseTransformPoint(bone.position);
                                c.contactRightSide = bones[b] == HumanBodyBones.RightFoot || bones[b] == HumanBodyBones.RightHand ||
                                                     (bones[b] == HumanBodyBones.Head && c.contactOffset.x >= 0f);
                            }
                        }
                        prev[b] = rel;
                    }
                }
                c.measured = bestSpeed > 0.5f;
                if (c.measured)
                {
                    c.contact = bestT;
                    c.clipStart = Mathf.Max(0f, bestT - c.maxLead);
                    c.clipEnd = Mathf.Min(len, bestT + c.maxFollow);
                }
                report.Append($"\n  {c.name,-15} contact {c.contact:F2}s of {len:F2}s (peak {bestSpeed:F1} m/s{(c.measured ? "" : ", NOT measured")}), " +
                              $"{(c.contactRightSide ? "right" : "left")} limb at ({c.contactOffset.x:F2}, {c.contactOffset.y:F2}, {c.contactOffset.z:F2})");
            }
            Destroy(go);
            Debug.Log($"[Replay] Measured action clips:{report}");
        }

        /// <summary>Turns the match events into per-player action animations and hands each body its list.</summary>
        void BuildActionSchedule()
        {
            actionsByPlayer = new Dictionary<int, List<ScheduledAction>>();
            if (!playActions || ModelOrNull == null || actionClips == null || actionClips.Count == 0) return;
            MeasureActionClips();
            var clips = new Dictionary<string, ActionClip>();
            foreach (var c in actionClips) if (c != null && c.clip != null) clips[c.name] = c;

            var keeperIds = new HashSet<int>();
            foreach (var p in keeperPieces) keeperIds.Add(p.id);
            actionsByPlayer = ActionScheduler.Build(data, clips, id => keeperIds.Contains(id), new ActionScheduler.Settings
            {
                runningKickSpeed = runningKickSpeed,
                longPassLength = longPassLength,
                headerMinHeight = headerMinHeight,
                chestMinHeight = chestMinHeight,
                nearestPlayerRange = 4f,
                dedupWindow = 0.35f,
                maxNudge = alignContact ? maxContactNudge : 0f,
                allowMirror = mirrorForBallSide,
            }, out var summary);

            List<ScheduledAction> Collect(IEnumerable<int> ids)
            {
                var list = new List<ScheduledAction>();
                foreach (var id in ids) if (actionsByPlayer.TryGetValue(id, out var l)) list.AddRange(l);
                list.Sort((a, b) => a.start.CompareTo(b.start));
                return list;
            }
            foreach (var id in identities)
            {
                var ids = new List<int>();
                foreach (var p in id.pieces) ids.Add(p.id);
                id.actions = Collect(ids);
            }
            foreach (var k in keepers)
            {
                var ids = new List<int>();
                foreach (var p in keeperPieces) if (p.team == k.team) ids.Add(p.id);
                k.actions = Collect(ids);
            }
            Debug.Log($"[Replay] Actions: {summary}");
        }

        /// <summary>
        /// Applies the active action (if any) to a body. Returns the point to face (if the action needs one)
        /// and the contact shift to add to the body's position this frame.
        /// </summary>
        Vector2? ApplyAction(PlayerBody body, List<ScheduledAction> actions, out Vector2 nudge)
        {
            nudge = Vector2.zero;
            if (!body.Animated) return null;
            var act = ActionScheduler.ActiveAt(actions, time);
            if (act == null) { body.SetAction(0, 0f, 0f); return null; }
            act.Evaluate(time, actionBlendIn, actionBlendOut, out float norm, out float weight);
            body.SetAction(act.clip.StateHash, norm, weight, act.mirror);
            if (weight > 0.01f) actionsPlaying++;
            if (alignContact) nudge = act.nudge * act.NudgeWeight(time, contactNudgeWindow) * weight;
            return weight > 0.2f ? act.faceTarget : (Vector2?)null;
        }

        /// <summary>
        /// Joins tracks that are probably the same person: a later track of the same team that starts
        /// within stitchMaxGap seconds of an earlier one ending, close enough to have run there unseen.
        /// Greedy in start-time order, picking the cheapest (shortest gap, shortest run) chain to extend.
        /// </summary>
        List<List<Track>> StitchIdentities(List<Track> pool)
        {
            var sorted = new List<Track>(pool);
            sorted.Sort((a, b) => a.Start.CompareTo(b.Start));
            var chains = new List<List<Track>>();
            foreach (var q in sorted)
            {
                int best = -1;
                float bestCost = float.MaxValue;
                for (int i = 0; i < chains.Count; i++)
                {
                    var p = chains[i][chains[i].Count - 1];
                    if (p.team != q.team || p.referee != q.referee) continue;
                    float gap = q.Start - p.End;
                    if (gap < 0f || gap > stitchMaxGap) continue;                // overlapping tracks are different people
                    float dist = Vector2.Distance(q.positions[0], p.positions[p.Count - 1]);
                    if (dist > stitchRunSpeed * gap + 3f) continue;              // couldn't have got there
                    float cost = gap + dist / Mathf.Max(0.1f, stitchRunSpeed);
                    if (cost < bestCost) { bestCost = cost; best = i; }
                }
                if (best >= 0) chains[best].Add(q);
                else chains.Add(new List<Track> { q });
            }
            return chains;
        }

        /// <summary>Average position of each team's tracked outfield players over time (0.1 s steps).</summary>
        void BuildCentroids(List<Track> pool)
        {
            int n = Mathf.Max(1, Mathf.CeilToInt(data.Duration / CentroidStep) + 1);
            centroidX = new[] { new float[n], new float[n] };
            centroidY = new[] { new float[n], new float[n] };
            for (int team = 0; team < 2; team++)
            {
                float lastX = 0f, lastY = 0f;
                for (int i = 0; i < n; i++)
                {
                    float t = data.startTime + i * CentroidStep;
                    float sx = 0f, sy = 0f; int c = 0;
                    foreach (var tr in pool)
                    {
                        if (tr.team != team || tr.referee || tr.goalkeeper) continue;
                        if (tr.TrySample(t, playerMaxGap, out var p)) { sx += p.x; sy += p.y; c++; }
                    }
                    if (c > 0) { lastX = sx / c; lastY = sy / c; }
                    centroidX[team][i] = lastX;
                    centroidY[team][i] = lastY;
                }
            }
        }

        Vector2 CentroidAt(int team, float t)
        {
            if (centroidX == null || team < 0 || team > 1) return Vector2.zero;
            float f = Mathf.Clamp((t - data.startTime) / CentroidStep, 0f, centroidX[team].Length - 1);
            int i = Mathf.FloorToInt(f), j = Mathf.Min(i + 1, centroidX[team].Length - 1);
            float k = f - i;
            return new Vector2(Mathf.Lerp(centroidX[team][i], centroidX[team][j], k), Mathf.Lerp(centroidY[team][i], centroidY[team][j], k));
        }

        /// <summary>Where this person is (or is estimated to be) at the current time.</summary>
        void EvaluateIdentity(Identity id)
        {
            id.real = false; id.candidate = false; id.currentPiece = null;
            Track prev = null, next = null, spanning = null;
            foreach (var p in id.pieces)
            {
                if (p.TrySample(time, playerMaxGap, out var pos))
                {
                    id.real = true; id.candidate = true; id.target = pos; id.currentPiece = p;
                    id.confidence = 0f; id.realWeight = 1f;
                    return;
                }
                if (p.Start <= time && time <= p.End) spanning = p;   // inside a track, but in a long dropout
                else if (p.End < time) prev = p;
                else if (next == null && p.Start > time) next = p;
            }

            float d;
            if (spanning != null)
            {
                id.target = spanning.SampleClamped(time);
                d = spanning.TimeToNearestSample(time);
                id.candidate = true;
            }
            else if (prev != null && next != null)
            {
                // Between two sightings: glide from where they were last seen to where they'll be seen next
                float u = Mathf.Clamp01((time - prev.End) / Mathf.Max(0.01f, next.Start - prev.End));
                u = u * u * (3f - 2f * u);
                id.target = Vector2.Lerp(prev.positions[prev.Count - 1], next.positions[0], u);
                d = Mathf.Min(time - prev.End, next.Start - time);
                id.candidate = true;
            }
            else
            {
                // Before the first or after the last sighting: stay near that spot, moving with the team's shape
                var anchorTrack = next ?? prev;
                bool before = next != null;
                float anchorT = before ? anchorTrack.Start : anchorTrack.End;
                Vector2 anchor = before ? anchorTrack.positions[0] : anchorTrack.positions[anchorTrack.Count - 1];
                Vector2 drift = Vector2.ClampMagnitude((CentroidAt(id.team, time) - CentroidAt(id.team, anchorT)) * ghostTeamFollow, ghostMaxDrift);
                id.target = anchor + drift;
                d = Mathf.Abs(time - anchorT);
                id.candidate = d <= ghostHoldTime;
            }
            id.confidence = d;
            float w = 1f - Mathf.Clamp01((d - playerMaxGap * 0.5f) / Mathf.Max(0.01f, ghostBlendTime));
            id.realWeight = w * w * (3f - 2f * w);
        }

        void UpdateIdentities(float dt, bool snap)
        {
            if (identities.Count == 0) return;
            estimatedCount = 0;
            foreach (var id in identities) EvaluateIdentity(id);

            // Keep at most maxOutfieldPerTeam outfield players per team: estimates closest in time to real data win
            for (int team = 0; team < 2; team++)
            {
                int real = 0;
                var ghosts = new List<Identity>();
                foreach (var id in identities)
                {
                    if (id.team != team || id.referee) continue;
                    if (id.real) real++;
                    else if (id.candidate) ghosts.Add(id);
                }
                ghosts.Sort((a, b) => a.confidence.CompareTo(b.confidence));
                int allowed = Mathf.Max(0, maxOutfieldPerTeam - real);
                for (int i = allowed; i < ghosts.Count; i++) ghosts[i].candidate = false;
            }

            foreach (var id in identities)
            {
                bool want = id.real || id.candidate;
                float targetPresence = want ? 1f : 0f;
                id.presence = snap ? targetPresence : Mathf.MoveTowards(id.presence, targetPresence, dt / 0.5f);
                if (id.presence <= 0.001f && !want) { id.body.SetActive(false); continue; }

                bool appearing = !id.body.Active;
                id.body.SetActive(true);
                bool jump = snap || appearing;

                // Displayed position: tracked players follow the data; estimates glide at a capped speed
                if (jump || dt <= 0f) { if (jump) id.displayPos = id.target; }
                else id.displayPos = Vector2.MoveTowards(id.displayPos, id.target, (id.real ? 20f : ghostMaxSpeed) * dt);

                // Velocity for facing/lean: from the track when tracked, from the displayed motion when estimated
                if (id.real && id.currentPiece != null) id.vel = id.currentPiece.VelocityAt(time, bodyMotion.velocityWindow);
                else if (jump) id.vel = Vector2.zero;
                else if (dt > 0f) id.vel = Vector2.Lerp(id.vel, (id.displayPos - id.prevDisplayPos) / dt, 1f - Mathf.Exp(-6f * dt));
                id.prevDisplayPos = id.displayPos;

                // Solid when tracked, see-through when estimated, fading with presence
                float a = Mathf.Lerp(ghostAlpha, 1f, id.realWeight) * id.presence;
                if (a >= 0.99f) id.body.SetLook(id.solidMat, id.solidTrim);
                else
                {
                    var c = id.color; c.a = a; id.ghostMat.color = c;
                    var tc = id.trimColor; tc.a = a; id.ghostTrim.color = tc;
                    id.ghostShorts.color = new Color(0.1f, 0.1f, 0.14f, a);
                    id.body.SetLook(id.ghostMat, id.ghostTrim, id.ghostShorts);
                }
                if (id.body.label != null)
                {
                    id.body.label.text = id.referee ? "REF" : id.real ? id.currentPiece.id.ToString() : "";
                    id.body.label.color = new Color(1f, 1f, 1f, Mathf.Max(a, 0.001f));
                }

                var face = ApplyAction(id.body, id.actions, out var idNudge);
                id.body.UpdateMotion(id.displayPos + idNudge, id.vel, lastBall, 1f, dt, jump, bodyMotion, face);

                if (!id.referee)
                {
                    if (id.real) visibleCount++;
                    else if (id.candidate) estimatedCount++;
                }
            }
        }

        /// <summary>
        /// Decides which goal each team defends and which keeper-labelled tracks are real keepers.
        /// Sides come from the team with the most keeper data; a "keeper" far from his own goal is
        /// treated as a misclassified outfield player. Real keeper pieces are stitched into one track.
        /// </summary>
        void PlanKeepers()
        {
            keeperPieces.Clear();
            rejectedKeepers = 0;
            var pieces = new[] { new List<Track>(), new List<Track>() };
            foreach (var tr in data.players)
                if (tr.goalkeeper && tr.team >= 0 && tr.team < 2) pieces[tr.team].Add(tr);

            int Samples(List<Track> l) { int n = 0; foreach (var t in l) n += t.Count; return n; }
            float AvgX(List<Track> l)
            {
                float s = 0f; int n = 0;
                foreach (var t in l) for (int i = 0; i < t.Count; i++) { s += t.positions[i].x; n++; }
                return n > 0 ? s / n : 0f;
            }

            plannedHomeSide = homeDefendsWest ? -1 : 1;
            if (autoDetectSides)
            {
                int best = Samples(pieces[0]) >= Samples(pieces[1]) ? 0 : 1;
                if (Samples(pieces[best]) > 0)
                {
                    int side = AvgX(pieces[best]) < 0 ? -1 : 1;
                    plannedHomeSide = best == 0 ? side : -side;
                }
            }

            for (int team = 0; team < 2; team++)
            {
                int side = team == 0 ? plannedHomeSide : -plannedHomeSide;
                float goalX = side * data.length / 2f;
                var accepted = new List<Track>();
                foreach (var piece in pieces[team])
                {
                    if (Mathf.Abs(piece.AverageX() - goalX) <= keeperMaxDistanceFromGoal) accepted.Add(piece);
                    else
                    {
                        piece.goalkeeper = false; // misclassified: draw as a normal outfield player
                        rejectedKeepers++;
                        Debug.Log($"[Replay] Track {piece.id} (team {team}) is labelled goalkeeper but averages {Mathf.Abs(piece.AverageX() - goalX):F0} m from its goal; treating as outfield");
                    }
                }
                keeperTrackByTeam[team] = null;
                if (accepted.Count == 0) continue;
                foreach (var a in accepted) keeperPieces.Add(a);

                var samples = new List<(float t, Vector2 p)>();
                foreach (var piece in accepted)
                    for (int i = 0; i < piece.Count; i++) samples.Add((piece.times[i], piece.positions[i]));
                samples.Sort((x, y) => x.t.CompareTo(y.t));
                var merged = new Track { id = accepted[0].id, team = team, goalkeeper = true };
                foreach (var s in samples)
                    if (merged.Count == 0 || s.t > merged.End) merged.Add(s.t, s.p);
                keeperTrackByTeam[team] = merged;
            }
        }

        void BuildKeepers(Transform parent)
        {
            var tracks = keeperTrackByTeam;
            int homeSide = plannedHomeSide;

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
                // Light joints read better than dark ones on the bright keeper kits
                k.trimMat = PitchBuilder.MakeTransparentMaterial(new Color(0.9f, 0.9f, 0.93f));
                k.body = PlayerBody.Create($"Keeper team {team}", parent, k.mat, k.trimMat, "", showLabels,
                                           ModelOrNull, KeeperController);
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
                // Keepers always face the ball (they shuffle and backpedal rather than turn their back on play);
                // only their own actions (catch, kicks) override this
                var keeperFace = ApplyAction(k.body, k.actions, out var keeperNudge) ?? ball;
                k.body.UpdateMotion(k.displayPos + keeperNudge, k.vel, ball, 1f, dt, snapKeepers, bodyMotion, keeperFace);

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
            if (data == null)
            {
                var keys = Keyboard.current;
                if (keys != null && keys.lKey.wasPressedThisFrame && loadError == null) OpenMatchDialog();
                return;
            }
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
            UpdateOverlayBridge();
            motionSnap = false;
        }

        // ---------------- Stats overlay (PitchOverlay: scoreboard bar, STATS panel, formation links) ----------------

        [Header("Stats overlay")]
        [Tooltip("The PitchOverlay MatchOverlay in the scene (scoreboard bar + STATS panel). Optional")]
        public PitchOverlay.MatchOverlay matchOverlay;
        [Tooltip("Show the pink formation lines on the pitch (toggle in the app with the Lines button or F)")]
        public bool showFormationLines = true;
        bool overlayVisible;

        /// <summary>Turn the formation lines on or off (Lines button / F key).</summary>
        public void ToggleFormationLines()
        {
            showFormationLines = !showFormationLines;
            ApplyFormationLinesVisibility();
        }

        /// <summary>The lines' object is active only while a match is loaded and the lines are switched on.
        /// Switching the whole object off hides every line at once (unticking the component would freeze them).</summary>
        void ApplyFormationLinesVisibility()
        {
            if (matchOverlay == null || matchOverlay.links == null) return;
            matchOverlay.links.gameObject.SetActive(overlayVisible && showFormationLines);
        }

        /// <summary>Show or hide the overlay's UI and pitch lines (hidden until a match is loaded).</summary>
        void SetOverlayVisible(bool visible)
        {
            overlayVisible = visible;
            if (matchOverlay == null) return;
            // Hide the UI with a style rather than switching its object off: re-enabling a UIDocument rebuilds
            // its visual tree, which would throw away the bar the StatsPanel built at startup
            var doc = matchOverlay.panel != null ? matchOverlay.panel.GetComponent<UnityEngine.UIElements.UIDocument>() : null;
            if (doc != null && doc.rootVisualElement != null)
                doc.rootVisualElement.style.display = visible ? UnityEngine.UIElements.DisplayStyle.Flex : UnityEngine.UIElements.DisplayStyle.None;
            ApplyFormationLinesVisibility();
        }

        /// <summary>Give the overlay the same file the replay is showing.</summary>
        void LoadOverlay(string path)
        {
            if (matchOverlay == null) return;
            SetOverlayVisible(true);
            try { matchOverlay.LoadMatchFile(path); }
            catch (System.Exception e) { Debug.LogWarning($"[Replay] Stats overlay couldn't read {path}: {e.Message}"); SetOverlayVisible(false); return; }
            // No team names from an analysis file: name the teams after the kits we draw them in
            var panel = matchOverlay.panel;
            if (panel != null && panel.HasDefaultNames && !useKitColors) panel.SetTeamNames("White", "Black");
        }

        /// <summary>
        /// Feed the overlay the replay's clock and the players on screen (see PlayerMotionDriver), so its stats
        /// and formation lines follow pause, scrubbing and speed exactly like the 3D view.
        /// </summary>
        void UpdateOverlayBridge()
        {
            PlayerMotionDriver.Clock = time;
            PlayerMotionDriver.ById.Clear();
            if (matchOverlay == null) return;
            if (estimateOffscreenPlayers)
            {
                foreach (var id in identities)
                {
                    // Only players the camera is tracking right now (estimates have no tracking id)
                    if (id.referee || !id.real || id.currentPiece == null || !id.body.Active) continue;
                    RegisterOverlayPlayer(id.body, id.currentPiece.id, id.team, id.currentPiece.goalkeeper);
                }
            }
            else
            {
                foreach (var (tr, body) in players)
                    if (!tr.referee && body.Active) RegisterOverlayPlayer(body, tr.id, tr.team, tr.goalkeeper);
            }
        }

        static void RegisterOverlayPlayer(PlayerBody body, int id, int team, bool keeper)
        {
            var d = body.root.GetComponent<PlayerMotionDriver>();
            if (d == null) d = body.root.gameObject.AddComponent<PlayerMotionDriver>();
            d.Set(id, team, keeper, true);
            PlayerMotionDriver.ById[id] = d;
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
            if (kb.digit4Key.wasPressedThisFrame) EnterFreeCamera();
            if (kb.lKey.wasPressedThisFrame) OpenMatchDialog();
            if (kb.fKey.wasPressedThisFrame) ToggleFormationLines();
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
            UpdateOverlayBridge();
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
            actionsPlaying = 0;

            // Ball first, so players can turn to face its current position.
            // In a fitted flight, use the smooth arc; otherwise the frame-by-frame track.
            var flight = useFittedFlights ? data.FlightAt(time) : null;
            float ballHeightNow = 0f;
            if (flight != null)
            {
                flight.Sample(time, out lastBall, out ballHeightNow);
                ballVisible = true;
            }
            else if (data.ball.Count > 0 && data.ball.TrySample(time, ballMaxGap, out var ballNow))
            {
                lastBall = ballNow;
                // When the file has fitted flights, they are the authority on when the ball is airborne:
                // between flights it's on the ground (frame heights there come from gap-filling and conflict with the fits)
                ballHeightNow = useFittedFlights && data.flights.Count > 0 ? 0f : data.ball.HeightAt(time);
                ballVisible = true;
            }
            else if (data.ball.Count > 0 && time > data.ball.End && BallInGoal(data.ball.positions[data.ball.Count - 1]))
            {
                // Last seen in the net (a goal): leave it resting there instead of vanishing
                lastBall = data.ball.positions[data.ball.Count - 1];
                ballHeightNow = 0f;
                ballVisible = true;
            }
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
                actionsByPlayer.TryGetValue(track.id, out var trackActions);
                var trackFace = ApplyAction(body, trackActions, out var trackNudge);
                body.UpdateMotion(p + trackNudge, vel, lastBall, fade, motionDt, motionSnap || justAppeared, bodyMotion, trackFace);
            }

            UpdateIdentities(motionDt, motionSnap);

            if (ballGo.gameObject.activeSelf != ballVisible) ballGo.gameObject.SetActive(ballVisible);
            if (ballVisible)
            {
                lastBallHeight = ballHeightNow;
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
                if (e.IsFlight && e.start != null && e.end != null && time >= e.t_start && time <= e.t_end + 1f) pass = e;
            passLine.enabled = pass != null;
            if (pass != null)
            {
                var a = new Vector3(pass.start[0], 0.1f, pass.start[1]);
                var b = new Vector3(pass.end[0], 0.1f, pass.end[1]);
                bool lofted = pass.aerial || pass.type == "aerial_ball";
                // Same (gravity-capped) height as the ball's own arc, so the drawn path matches the ball
                var arcFlight = data.FlightAt(0.5f * (pass.t_start + pass.t_end));
                float peak = !lofted ? 0f : arcFlight != null && arcFlight.drawHeight >= 0f ? arcFlight.drawHeight : Mathf.Max(0.5f, pass.max_height);
                for (int i = 0; i < PassLinePoints; i++)
                {
                    float u = i / (float)(PassLinePoints - 1);
                    var p = Vector3.Lerp(a, b, u);
                    p.y += 4f * peak * u * (1f - u); // parabola peaking at max_height
                    passLine.SetPosition(i, p);
                }
                float fade = time > pass.t_end ? 1f - (time - pass.t_end) : 1f;
                // White = completed pass, red = failed pass, amber = loose lofted ball (not a pass)
                var c = pass.type == "aerial_ball" ? new Color(1f, 0.75f, 0.3f, 0.7f * fade)
                      : pass.completed ? new Color(1f, 1f, 1f, 0.8f * fade)
                      : new Color(1f, 0.3f, 0.3f, 0.8f * fade);
                passLine.startColor = passLine.endColor = c;
                passLine.material.color = c;
            }
        }

        /// <summary>True if a position is inside one of the goals (past the goal line, between the posts).</summary>
        bool BallInGoal(Vector2 p) => Mathf.Abs(p.x) > data.length / 2f - 0.3f && Mathf.Abs(p.x) < data.length / 2f + 3f && Mathf.Abs(p.y) < 3.66f + 1f;

        /// <summary>Where play is: the ball when it's visible, otherwise the middle of the visible players.</summary>
        Vector2 PlayFocusTarget()
        {
            if (ballVisible) return lastBall;
            Vector2 sum = Vector2.zero; int n = 0;
            foreach (var id in identities) if (id.body.Active && id.real && !id.referee) { sum += id.displayPos; n++; }
            foreach (var (track, body) in players) if (body.Active && !track.referee) { var p = body.root.localPosition; sum += new Vector2(p.x, p.z); n++; }
            return n > 0 ? sum / n : cameraFocus;
        }

        /// <summary>
        /// Smoothed, speed-limited camera focus: follows play calmly, and a brief wrong ball position can
        /// only nudge it a little instead of swinging the camera to the other end of the pitch.
        /// </summary>
        void UpdateCameraFocus(bool snap)
        {
            var target = PlayFocusTarget();
            if (snap || !cameraFocusInit) { cameraFocus = target; cameraFocusVel = Vector2.zero; cameraFocusInit = true; return; }
            float dt = Time.deltaTime;
            if (dt <= 0f) return;
            cameraFocus = Vector2.SmoothDamp(cameraFocus, target, ref cameraFocusVel, cameraFocusSmoothing, cameraFocusMaxSpeed, dt);
        }

        /// <summary>Switch to the free camera, starting from wherever the camera is now.</summary>
        void EnterFreeCamera()
        {
            if (cam == null) return;
            var e = cam.transform.eulerAngles;
            freeYaw = e.y;
            freePitch = e.x > 180f ? e.x - 360f : e.x;
            cameraMode = CameraMode.Free;
        }

        /// <summary>
        /// Free-fly camera: hold the right mouse button to look, WASD to move, E / Q up / down,
        /// Shift for speed, mouse wheel to change fly speed. Works while paused (uses real time).
        /// </summary>
        void UpdateFreeCamera()
        {
            var kb = Keyboard.current;
            var mouse = Mouse.current;
            float dt = Time.unscaledDeltaTime;
            if (mouse != null)
            {
                if (mouse.rightButton.isPressed)
                {
                    var d = mouse.delta.ReadValue();
                    freeYaw += d.x * freeLookSensitivity;
                    freePitch = Mathf.Clamp(freePitch - d.y * freeLookSensitivity, -89f, 89f);
                }
                float scroll = mouse.scroll.ReadValue().y;
                if (Mathf.Abs(scroll) > 0.01f) freeMoveSpeed = Mathf.Clamp(freeMoveSpeed * (scroll > 0f ? 1.15f : 1f / 1.15f), 1f, 80f);
            }
            cam.transform.rotation = Quaternion.Euler(freePitch, freeYaw, 0f);

            if (kb != null)
            {
                Vector3 move = Vector3.zero;
                var t = cam.transform;
                if (kb.wKey.isPressed) move += t.forward;
                if (kb.sKey.isPressed) move -= t.forward;
                if (kb.dKey.isPressed) move += t.right;
                if (kb.aKey.isPressed) move -= t.right;
                if (kb.eKey.isPressed) move += Vector3.up;
                if (kb.qKey.isPressed) move -= Vector3.up;
                float spd = freeMoveSpeed * (kb.leftShiftKey.isPressed || kb.rightShiftKey.isPressed ? 3f : 1f);
                if (move.sqrMagnitude > 0f) t.position += move.normalized * spd * dt;
                if (t.position.y < 0.3f) t.position = new Vector3(t.position.x, 0.3f, t.position.z); // stay above the grass
            }
        }

        void LateUpdate()
        {
            if (data == null || cam == null) return;
            UpdateCameraFocus(false);
            if (cameraMode == CameraMode.Free)
            {
                UpdateFreeCamera();
                BillboardLabels();
                return;
            }
            GetCameraTarget(out var pos, out var look);
            float k = 1f - Mathf.Exp(-3f * Time.deltaTime);
            cam.transform.position = Vector3.Lerp(cam.transform.position, pos, k);
            var targetRot = Quaternion.LookRotation(look - pos);
            cam.transform.rotation = Quaternion.Slerp(cam.transform.rotation, targetRot, k);
            BillboardLabels();
        }

        void BillboardLabels()
        {

            // Billboard labels toward the camera
            foreach (var (_, body) in players)
                if (body.Active && body.label != null) body.label.transform.rotation = cam.transform.rotation;
            foreach (var id in identities)
                if (id.body.Active && id.body.label != null) id.body.label.transform.rotation = cam.transform.rotation;
            foreach (var keeper in keepers)
                if (keeper.label != null) keeper.label.transform.rotation = cam.transform.rotation;
        }

        void SnapCamera()
        {
            UpdateCameraFocus(true);
            GetCameraTarget(out var pos, out var look);
            cam.transform.position = pos;
            cam.transform.rotation = Quaternion.LookRotation(look - pos);
        }

        void GetCameraTarget(out Vector3 pos, out Vector3 look)
        {
            Vector2 focus = cameraFocus;
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
                    "aerial_ball" => $"{rel,5:F1}s  Ball in the air  {e.max_height:F1} m high{(e.estimated ? " (est.)" : "")}",
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
                if (GUI.Button(new Rect(10, 76, 140, 28), "Load match…")) { loadError = null; OpenMatchDialog(); }
                return;
            }
            if (data == null) { DrawStartPanel(); return; }

            DrawEventFeed();

            const float h = 64f;
            var area = new Rect(10, Screen.height - h - 10, Screen.width - 20, h);
            GUI.Box(area, GUIContent.none);
            GUILayout.BeginArea(new Rect(area.x + 8, area.y + 6, area.width - 16, area.height - 12));

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Load match…", GUILayout.Width(100))) OpenMatchDialog();
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
            if (GUILayout.Toggle(cameraMode == CameraMode.Free, "Free", "Button", GUILayout.Width(60)) && cameraMode != CameraMode.Free) EnterFreeCamera();
            if (matchOverlay != null && GUILayout.Button("Stats", GUILayout.Width(60))) matchOverlay.ToggleStats();
            if (matchOverlay != null && matchOverlay.links != null &&
                GUILayout.Toggle(showFormationLines, "Lines", "Button", GUILayout.Width(60)) != showFormationLines) ToggleFormationLines();
            if (cameraMode == CameraMode.Free)
                GUILayout.Label($"  Right-drag look · WASD move · E/Q up/down · Shift fast · wheel speed ({freeMoveSpeed:F0} m/s)");
            GUILayout.FlexibleSpace();
            int keepersReal = keepers.FindAll(k => k.realWeight > 0.5f).Count;
            string keeperInfo = keepers.Count > 0 ? $"   Keepers: {keepersReal} tracked, {keepers.Count - keepersReal} estimated" : "";
            string estInfo = identities.Count > 0 ? $"  (+{estimatedCount} estimated off camera)" : "";
            GUILayout.Label($"{loadedName}   Players tracked: {visibleCount}{estInfo}{keeperInfo}   Ball: {(ballVisible ? "yes" : "lost")}");
            GUILayout.EndHorizontal();

            GUILayout.EndArea();
        }
    }
}
