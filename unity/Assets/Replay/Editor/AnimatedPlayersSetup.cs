using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEngine;

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

            // 2) Locomotion controller: 2D freeform-directional blend tree on MoveX/MoveY (body-relative velocity)
            AssetDatabase.DeleteAsset(ControllerPath);
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(ControllerPath);
            ctrl.AddParameter("MoveX", AnimatorControllerParameterType.Float);
            ctrl.AddParameter("MoveY", AnimatorControllerParameterType.Float);
            ctrl.AddParameter("Speed", AnimatorControllerParameterType.Float);

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
                EditorUtility.SetDirty(rc);
                EditorSceneManager.MarkSceneDirty(rc.gameObject.scene);
                EditorSceneManager.SaveScene(rc.gameObject.scene);
            }
            Debug.Log($"[Replay] Animated players set up: {ControllerPath} (idle + {JogClips.Length} jogs, avg jog speed {jogSpeedSum / JogClips.Length:F2} m/s; extra gaits:{(gaitInfo.Length > 0 ? gaitInfo.ToString() : " none")}; top speed {topSpeed:F2} m/s), keeper override, Replay object wired: {rc != null}");
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

        static void ConfigureLoop(string name)
        {
            var path = $"{PackDir}/{name}.fbx";
            if (!(AssetImporter.GetAtPath(path) is ModelImporter imp)) { Debug.LogWarning($"[Replay] Missing clip {path}"); return; }
            var clips = imp.defaultClipAnimations;
            foreach (var c in clips)
            {
                c.loopTime = true;
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
