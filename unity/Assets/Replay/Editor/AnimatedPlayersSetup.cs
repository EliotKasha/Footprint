using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

namespace SoccerReplay.EditorTools
{
    /// <summary>
    /// One-click setup for animated players from the Soccer Game Pack (Mixamo, Y Bot):
    /// clip import settings, a locomotion blend tree, a keeper variant, and wiring on the Replay object.
    /// Re-run it after adding clips (e.g. walk/run/sprint) and it rebuilds everything.
    /// </summary>
    public static class AnimatedPlayersSetup
    {
        const string PackDir = "Assets/Characters/SoccerGamePack";
        const string ControllerPath = "Assets/Characters/PlayerLocomotion.controller";
        const string KeeperOverridePath = "Assets/Characters/KeeperLocomotion.overrideController";

        // Center of the blend tree, and the 8-direction jog ring around it
        const string Idle = "offensive idle";
        const string KeeperIdle = "goalkeeper idle";
        static readonly string[] JogClips =
        {
            "jog forward", "jog backward", "jog strafe left", "jog strafe right",
            "jog forward diagonal", "jog forward diagonal (2)", "jog backward diagonal", "jog backward diagonal (2)",
        };
        // Optional forward gaits: added to the blend tree at their measured speed if the file is in the pack
        static readonly string[] OptionalGaits = { "Walking", "Running", "Run", "Fast Run", "Sprint" };

        enum Contact { Feet, Head, Hands }

        // Action clips: name used by the scheduler, file, which body part meets the ball,
        // and how much of the clip to keep before / after contact (trims long wind-ups and follow-throughs)
        static readonly (string name, string file, Contact part, float maxLead, float maxFollow, float searchEnd)[] Actions =
        {
            ("Kick",           "kick soccerball",      Contact.Feet,  0.6f, 0.5f, 99f),
            ("KickRun",        "strike foward jog",    Contact.Feet,  0.8f, 0.6f, 99f),
            ("LongKick",       "soccer penalty kick",  Contact.Feet,  0.9f, 0.6f, 99f),
            ("Receive",        "receive soccerball",   Contact.Feet,  0.6f, 0.7f, 2.0f),
            ("Header",         "header soccerball",    Contact.Head,  0.8f, 0.6f, 99f),
            ("KeeperCatch",    "goalkeeper catch",     Contact.Hands, 0.7f, 0.8f, 99f),
            ("KeeperDropKick", "goalkeeper drop kick", Contact.Feet,  1.2f, 0.8f, 99f),
            ("KeeperPass",     "goalkeeper pass",      Contact.Feet,  1.0f, 0.7f, 99f),
        };

        [MenuItem("Soccer Replay/Setup Animated Players")]
        public static void Setup()
        {
            // 1) Looping clips: loop, and bake rotation/height into the pose so the clip can't turn or lift
            //    the character away from the tracking data. Horizontal root motion stays as root motion,
            //    which the Animator ignores (applyRootMotion is off), so players stay exactly on their track.
            var gaits = OptionalGaits.Where(g => AssetImporter.GetAtPath($"{PackDir}/{g}.fbx") != null).ToArray();
            foreach (var g in gaits) EnsureHumanoid(g);
            foreach (var name in JogClips.Concat(gaits).Concat(new[] { Idle, KeeperIdle, "goalkeeper idle (2)" }))
                ConfigureLoop(name);
            foreach (var a in Actions) ConfigureLoop(a.file, loop: false);

            // 2) Locomotion controller: 2D freeform-directional blend tree on MoveX/MoveY (body-relative velocity)
            AssetDatabase.DeleteAsset(ControllerPath);
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(ControllerPath);
            ctrl.AddParameter("MoveX", AnimatorControllerParameterType.Float);
            ctrl.AddParameter("MoveY", AnimatorControllerParameterType.Float);
            ctrl.AddParameter("Speed", AnimatorControllerParameterType.Float);
            ctrl.AddParameter("Mirror", AnimatorControllerParameterType.Bool);

            var state = ctrl.CreateBlendTreeInController("Locomotion", out var tree, 0);
            tree.blendType = BlendTreeType.FreeformDirectional2D;
            tree.blendParameter = "MoveX";
            tree.blendParameterY = "MoveY";
            tree.useAutomaticThresholds = false;
            tree.AddChild(Clip(Idle), Vector2.zero);
            float jogSpeedSum = 0f;
            foreach (var name in JogClips)
            {
                var c = Clip(name);
                // Place each jog at its own measured ground velocity, so the blend reproduces real movement
                var v = c.averageSpeed;
                tree.AddChild(c, new Vector2(v.x, v.z));
                jogSpeedSum += new Vector2(v.x, v.z).magnitude;
            }
            float topSpeed = jogSpeedSum / JogClips.Length;
            var gaitInfo = new System.Text.StringBuilder();
            foreach (var g in gaits)
            {
                var c = Clip(g);
                var v = c.averageSpeed;
                float s = new Vector2(v.x, v.z).magnitude;
                if (s < 0.2f) { Debug.LogWarning($"[Replay] {g} has no forward motion (was it downloaded 'In Place'?) - skipped"); continue; }
                // Forward gaits go straight ahead at their natural speed
                tree.AddChild(c, new Vector2(0f, s));
                topSpeed = Mathf.Max(topSpeed, s);
                gaitInfo.Append($" {g}={s:F2}m/s");
            }
            ctrl.layers[0].stateMachine.defaultState = state;

            // Actions layer: full-body override, weight driven from code (0 when no action is playing)
            ctrl.AddLayer("Actions");
            var layers = ctrl.layers;
            layers[1].blendingMode = AnimatorLayerBlendingMode.Override;
            layers[1].defaultWeight = 0f;
            ctrl.layers = layers;
            var actionSm = ctrl.layers[1].stateMachine;
            actionSm.defaultState = actionSm.AddState("Empty");
            var model = AssetDatabase.LoadAssetAtPath<GameObject>(PackDir + "/Y Bot.fbx");
            var actionClips = new List<ActionClip>();
            var contactInfo = new System.Text.StringBuilder();
            foreach (var a in Actions)
            {
                AnimationClip clip;
                try { clip = Clip(a.file); } catch { Debug.LogWarning($"[Replay] Action clip missing: {a.file}"); continue; }
                var st = actionSm.AddState(a.name);
                st.motion = clip;
                st.writeDefaultValues = false;
                st.mirrorParameterActive = true;   // "Mirror" plays the other-foot version
                st.mirrorParameter = "Mirror";
                float contact = MeasureContact(model, clip, a.part, a.searchEnd, out bool measured, out var offset, out bool rightSide);
                float start = Mathf.Max(0f, contact - a.maxLead);
                float end = Mathf.Min(clip.length, contact + a.maxFollow);
                actionClips.Add(new ActionClip
                {
                    name = a.name, clip = clip, state = a.name, contact = contact, clipStart = start, clipEnd = end,
                    contactOffset = offset, contactRightSide = rightSide,
                    contactPart = (int)a.part, maxLead = a.maxLead, maxFollow = a.maxFollow, searchEnd = a.searchEnd,
                });
                contactInfo.Append($"\n  {a.name,-15} {a.file,-22} contact {contact:F2}s of {clip.length:F2}s{(measured ? "" : " (estimated)")}, plays {start:F2}-{end:F2}s, " +
                                   $"{(rightSide ? "right" : "left")} limb at ({offset.x:F2}, {offset.y:F2}, {offset.z:F2})");
            }
            EditorUtility.SetDirty(ctrl);

            // 3) Goalkeepers: same tree, goalkeeper idle in the middle
            AssetDatabase.DeleteAsset(KeeperOverridePath);
            var keeper = new AnimatorOverrideController(ctrl) { name = "KeeperLocomotion" };
            keeper[Clip(Idle)] = Clip(KeeperIdle);
            AssetDatabase.CreateAsset(keeper, KeeperOverridePath);
            AssetDatabase.SaveAssets();

            // 4) Wire up the Replay object
            var rc = Object.FindAnyObjectByType<ReplayController>();
            if (rc != null)
            {
                Undo.RecordObject(rc, "Setup animated players");
                rc.playerModel = AssetDatabase.LoadAssetAtPath<GameObject>(PackDir + "/Y Bot.fbx");
                rc.playerAnimator = ctrl;
                rc.keeperAnimator = keeper;
                rc.useAnimatedPlayers = true;
                rc.bodyMotion.jogClipSpeed = topSpeed;   // fastest gait in the tree; beyond it clips play faster
                rc.actionClips = actionClips;
                EditorUtility.SetDirty(rc);
                EditorSceneManager.MarkSceneDirty(rc.gameObject.scene);
                EditorSceneManager.SaveScene(rc.gameObject.scene);
            }
            Debug.Log($"[Replay] Animated players set up: {ControllerPath} (idle + {JogClips.Length} jogs, avg jog speed {jogSpeedSum / JogClips.Length:F2} m/s; extra gaits:{(gaitInfo.Length > 0 ? gaitInfo.ToString() : " none")}; top speed {topSpeed:F2} m/s), keeper override, Replay object wired: {rc != null}");
            Debug.Log($"[Replay] Action clips ({actionClips.Count}):{contactInfo}");
        }

        /// <summary>
        /// Finds when the foot / head / hands meet the ball: plays the clip on a hidden copy of the
        /// character and returns the moment that body part moves fastest relative to the hips.
        /// </summary>
        static float MeasureContact(GameObject model, AnimationClip clip, Contact part, float searchEnd, out bool measured,
                                    out Vector3 contactOffset, out bool rightSide)
        {
            measured = false;
            contactOffset = new Vector3(0.2f, 0.1f, 0.4f);   // fallback: in front of the right foot
            rightSide = true;
            float fallback = clip.length * 0.45f;
            if (model == null) return fallback;
            var go = Object.Instantiate(model);
            go.hideFlags = HideFlags.HideAndDontSave;
            go.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            var animator = go.GetComponent<Animator>();
            if (animator == null || animator.avatar == null || !animator.avatar.isHuman) { Object.DestroyImmediate(go); return fallback; }
            animator.applyRootMotion = false;   // keep the character at the origin so limb positions are relative to it

            var bones = part == Contact.Feet ? new[] { HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot }
                      : part == Contact.Head ? new[] { HumanBodyBones.Head }
                      : new[] { HumanBodyBones.LeftHand, HumanBodyBones.RightHand };
            var graph = PlayableGraph.Create("MeasureContact");
            try
            {
                graph.SetTimeUpdateMode(DirectorUpdateMode.Manual);
                var output = AnimationPlayableOutput.Create(graph, "out", animator);
                var playable = AnimationClipPlayable.Create(graph, clip);
                playable.SetApplyFootIK(false);
                output.SetSourcePlayable(playable);

                var hips = animator.GetBoneTransform(HumanBodyBones.Hips);
                const float step = 1f / 60f;
                var prev = new Vector3[bones.Length];
                float bestSpeed = 0f, bestT = fallback;
                float end = Mathf.Min(clip.length, searchEnd);
                for (int i = 0; i * step <= end; i++)
                {
                    float t = i * step;
                    playable.SetTime(t);
                    graph.Evaluate();
                    for (int b = 0; b < bones.Length; b++)
                    {
                        var bone = animator.GetBoneTransform(bones[b]);
                        if (bone == null || hips == null) continue;
                        var rel = bone.position - hips.position;
                        if (i > 0)
                        {
                            float speed = (rel - prev[b]).magnitude / step;
                            if (speed > bestSpeed)
                            {
                                bestSpeed = speed; bestT = t;
                                contactOffset = go.transform.InverseTransformPoint(bone.position);
                                rightSide = bones[b] == HumanBodyBones.RightFoot || bones[b] == HumanBodyBones.RightHand ||
                                            (bones[b] == HumanBodyBones.Head && contactOffset.x >= 0f);
                            }
                        }
                        prev[b] = rel;
                    }
                }
                measured = bestSpeed > 0.5f;
                return measured ? bestT : fallback;
            }
            finally
            {
                graph.Destroy();
                Object.DestroyImmediate(go);
            }
        }

        /// <summary>Files downloaded separately (e.g. with skin) get a humanoid rig built from their own skeleton.</summary>
        static void EnsureHumanoid(string name)
        {
            if (!(AssetImporter.GetAtPath($"{PackDir}/{name}.fbx") is ModelImporter imp)) return;
            if (imp.animationType == ModelImporterAnimationType.Human) return;
            imp.animationType = ModelImporterAnimationType.Human;
            imp.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
            imp.materialImportMode = ModelImporterMaterialImportMode.None;
            imp.SaveAndReimport();
        }

        static void ConfigureLoop(string name, bool loop = true)
        {
            var path = $"{PackDir}/{name}.fbx";
            if (!(AssetImporter.GetAtPath(path) is ModelImporter imp)) { Debug.LogWarning($"[Replay] Missing clip {path}"); return; }
            var clips = imp.defaultClipAnimations;
            foreach (var c in clips)
            {
                c.loopTime = loop;
                c.lockRootRotation = true;          // bake rotation into pose
                c.keepOriginalOrientation = true;
                c.lockRootHeightY = true;           // bake Y into pose
                c.keepOriginalPositionY = true;
                c.lockRootPositionXZ = false;       // XZ stays root motion (ignored at runtime)
            }
            imp.clipAnimations = clips;
            imp.SaveAndReimport();
        }

        static AnimationClip Clip(string name)
        {
            var clip = AssetDatabase.LoadAllAssetsAtPath($"{PackDir}/{name}.fbx")
                .OfType<AnimationClip>().FirstOrDefault(c => !c.name.StartsWith("__preview__"));
            if (clip == null) throw new System.Exception($"Animation clip not found: {name}");
            return clip;
        }
    }
}
