using TMPro;
using UnityEngine;

namespace SoccerReplay
{
    /// <summary>Shared tuning for how player bodies move.</summary>
    [System.Serializable]
    public class BodyMotionSettings
    {
        [Tooltip("Velocity is measured over this many seconds, centered on the current moment")]
        public float velocityWindow = 0.3f;
        [Tooltip("SmoothDamp time on speed, so jitter doesn't make players twitch")]
        public float speedSmooth = 0.15f;
        [Tooltip("Speeds above this (m/s) are treated as tracking glitches")]
        public float maxSpeed = 11f;
        [Tooltip("Below this speed (m/s) players face the ball instead of their direction of travel")]
        public float faceMoveAbove = 1.2f;
        [Tooltip("How fast players can turn (degrees per second)")]
        public float turnRate = 540f;
        [Tooltip("Forward lean in degrees per m/s of speed")]
        public float leanPerMs = 1.6f;
        public float maxLean = 13f;
        [Tooltip("Vertical bob at full jog (meters)")]
        public float bobHeight = 0.06f;

        [Header("Animated players")]
        [Tooltip("Natural ground speed of the jog clips (m/s); the jog blend is driven up to this speed")]
        public float jogClipSpeed = 2.4f;
        [Tooltip("Above jog speed the jog plays faster, up to this factor (until run/sprint clips are added)")]
        public float maxAnimSpeedup = 1.6f;
        [Tooltip("Extra lean on animated players, as a fraction of the capsule lean (the clips already lean)")]
        [Range(0f, 1f)] public float animatedLeanScale = 0.35f;
        [Tooltip("Damping (s) on the blend-tree inputs, so gait changes are smooth")]
        public float blendDamping = 0.12f;
    }

    /// <summary>
    /// Visual body for one player: root (ground position, facing, fade) -> lean pivot (tilt, bob) -> capsule.
    /// A chest stripe shows which way the player faces; a shorts band breaks up the capsule like a kit.
    /// </summary>
    public class PlayerBody
    {
        public Transform root;
        public Transform lean;
        public TextMeshPro label;
        public Material bodyMat, trimMat;
        Renderer bodyRenderer, stripeRenderer, shortsRenderer;
        public static Material SharedShortsMaterial => shortsMat;

        Animator animator;
        public bool Animated => animator != null;
        static readonly int MoveXId = Animator.StringToHash("MoveX");
        static readonly int MoveYId = Animator.StringToHash("MoveY");
        static readonly int SpeedId = Animator.StringToHash("Speed");

        /// <summary>Swap materials (e.g. solid when tracked, see-through when estimated). Null shorts keeps the shared solid one.</summary>
        public void SetLook(Material body, Material trim, Material shorts = null)
        {
            if (bodyRenderer.sharedMaterial != body) bodyRenderer.sharedMaterial = body;
            if (stripeRenderer != null && stripeRenderer.sharedMaterial != trim) stripeRenderer.sharedMaterial = trim;
            if (shortsRenderer == null) return;
            var s = shorts != null ? shorts : shortsMat;
            if (shortsRenderer.sharedMaterial != s) shortsRenderer.sharedMaterial = s;
        }

        float speed, speedVel, yaw, phase;
        float leanPitch, leanRoll;
        bool hasYaw;

        public float Speed => speed;

        const float Height = 1.8f;
        static Material shortsMat;

        /// <summary>
        /// Creates a body. With a humanoid model and animator controller it's an animated character
        /// (body tinted with bodyMat, joints with trimMat); otherwise the capsule body.
        /// </summary>
        public static PlayerBody Create(string name, Transform parent, Material bodyMat, Material trimMat, string labelText, bool showLabel,
                                        GameObject model = null, RuntimeAnimatorController controller = null)
        {
            var b = new PlayerBody { bodyMat = bodyMat, trimMat = trimMat };
            b.root = new GameObject(name).transform;
            b.root.SetParent(parent, false);

            b.lean = new GameObject("Lean").transform;
            b.lean.SetParent(b.root, false);

            if (model != null && controller != null)
            {
                var inst = Object.Instantiate(model, b.lean, false);
                inst.name = "Model";
                SkinnedMeshRenderer surface = null, joints = null;
                foreach (var smr in inst.GetComponentsInChildren<SkinnedMeshRenderer>())
                {
                    if (smr.name.ToLowerInvariant().Contains("joint")) joints = smr; else if (surface == null) surface = smr;
                }
                if (surface == null) surface = joints;
                b.bodyRenderer = surface;
                b.stripeRenderer = joints != null && joints != surface ? joints : null;
                if (b.bodyRenderer != null) b.bodyRenderer.sharedMaterial = bodyMat;
                if (b.stripeRenderer != null) b.stripeRenderer.sharedMaterial = trimMat;

                b.animator = inst.GetComponent<Animator>();
                if (b.animator == null) b.animator = inst.AddComponent<Animator>();
                b.animator.runtimeAnimatorController = controller;
                b.animator.applyRootMotion = false;          // position comes from the tracking data
                b.animator.cullingMode = AnimatorCullingMode.CullUpdateTransforms;
                b.animator.Play(0, 0, Random.value);        // desync so players don't step in unison
                AddLabel(b, labelText, showLabel);
                return b;
            }

            var body = PitchBuilder.Primitive(PrimitiveType.Capsule, "Body", b.lean, bodyMat);
            body.transform.localPosition = new Vector3(0f, Height / 2f, 0f);
            body.transform.localScale = new Vector3(0.7f, Height / 2f, 0.7f);
            b.bodyRenderer = body.GetComponent<Renderer>();

            // Chest stripe on the front: makes facing direction readable from any camera
            var stripe = PitchBuilder.Primitive(PrimitiveType.Cube, "Chest", b.lean, trimMat);
            stripe.transform.localPosition = new Vector3(0f, 1.32f, 0.31f);
            stripe.transform.localScale = new Vector3(0.36f, 0.11f, 0.12f);
            b.stripeRenderer = stripe.GetComponent<Renderer>();

            // Shorts band around the hips
            if (shortsMat == null) shortsMat = PitchBuilder.MakeMaterial(new Color(0.1f, 0.1f, 0.14f));
            var shorts = PitchBuilder.Primitive(PrimitiveType.Cylinder, "Shorts", b.lean, shortsMat);
            shorts.transform.localPosition = new Vector3(0f, 0.78f, 0f);
            shorts.transform.localScale = new Vector3(0.72f, 0.13f, 0.72f);
            b.shortsRenderer = shorts.GetComponent<Renderer>();

            AddLabel(b, labelText, showLabel);
            return b;
        }

        static void AddLabel(PlayerBody b, string labelText, bool showLabel)
        {
            var lgo = new GameObject("Label");
            lgo.transform.SetParent(b.root, false);
            lgo.transform.localPosition = new Vector3(0f, Height + 0.35f, 0f);
            b.label = lgo.AddComponent<TextMeshPro>();
            b.label.text = labelText;
            b.label.fontSize = 8;
            b.label.alignment = TextAlignmentOptions.Center;
            b.label.color = Color.white;
            b.label.outlineWidth = 0.25f;
            b.label.outlineColor = Color.black;
            b.label.rectTransform.sizeDelta = new Vector2(6f, 2f);
            lgo.SetActive(showLabel);
        }

        /// <summary>
        /// Moves and animates the body. dt is replay time elapsed (0 while paused); snap jumps straight
        /// to the target state (after a seek or when the player first appears).
        /// </summary>
        public void UpdateMotion(Vector2 pos, Vector2 vel, Vector2 lookAt, float fade, float dt, bool snap, BodyMotionSettings s)
        {
            root.localPosition = new Vector3(pos.x, 0f, pos.y);
            root.localScale = Vector3.one * Mathf.Max(0.001f, fade);

            // Speed: clamp glitches, then smooth so gait doesn't flicker with detection noise
            float raw = Mathf.Min(vel.magnitude, s.maxSpeed);
            if (snap) { speed = raw; speedVel = 0f; }
            else if (dt > 0f) speed = Mathf.SmoothDamp(speed, raw, ref speedVel, s.speedSmooth, Mathf.Infinity, dt);

            // Facing: direction of travel when moving, the ball when nearly still
            Vector2 faceDir = speed > s.faceMoveAbove ? vel : lookAt - pos;
            if (faceDir.sqrMagnitude > 0.01f)
            {
                float target = Mathf.Atan2(faceDir.x, faceDir.y) * Mathf.Rad2Deg;
                if (snap || !hasYaw) yaw = target;
                else if (dt > 0f) yaw = Mathf.MoveTowardsAngle(yaw, target, s.turnRate * dt);
                hasYaw = true;
            }
            root.localRotation = Quaternion.Euler(0f, yaw, 0f);

            // Lean into the movement relative to where the body faces (forward when running, a little sideways when cutting)
            float yr = yaw * Mathf.Deg2Rad;
            var forward = new Vector2(Mathf.Sin(yr), Mathf.Cos(yr));
            var right = new Vector2(forward.y, -forward.x);
            float vScale = vel.magnitude > 0.001f ? Mathf.Min(1f, speed / vel.magnitude) : 0f; // use the smoothed speed
            float fwd = Vector2.Dot(vel, forward) * vScale;
            float side = Vector2.Dot(vel, right) * vScale;
            float leanScale = Animated ? s.animatedLeanScale : 1f;
            float targetPitch = Mathf.Clamp(fwd * s.leanPerMs * leanScale, -s.maxLean * 0.4f, s.maxLean);
            float targetRoll = Mathf.Clamp(-side * s.leanPerMs * 0.8f * leanScale, -s.maxLean * 0.5f, s.maxLean * 0.5f);

            if (Animated)
            {
                // Blend tree input: velocity in the body's own frame (x = right, y = forward), so moving
                // one way while facing another gives backpedals and sidesteps. The jog clips top out at
                // jogClipSpeed; beyond that the clip plays faster instead (foot speed matches ground speed).
                var local = new Vector2(side, fwd);
                float mag = local.magnitude;
                float gait = 1f;
                if (mag > s.jogClipSpeed)
                {
                    gait = Mathf.Min(mag / s.jogClipSpeed, s.maxAnimSpeedup);
                    local *= s.jogClipSpeed / mag;
                }
                if (snap)
                {
                    animator.SetFloat(MoveXId, local.x);
                    animator.SetFloat(MoveYId, local.y);
                }
                else if (dt > 0f)
                {
                    animator.SetFloat(MoveXId, local.x, s.blendDamping, dt);
                    animator.SetFloat(MoveYId, local.y, s.blendDamping, dt);
                }
                animator.SetFloat(SpeedId, speed);
                // Follow replay time: frozen while paused, faster/slower with playback speed
                float playbackRate = Time.deltaTime > 0f ? dt / Time.deltaTime : 0f;
                animator.speed = playbackRate * gait;
            }
            float k = snap ? 1f : 1f - Mathf.Exp(-8f * dt);
            leanPitch = Mathf.Lerp(leanPitch, targetPitch, k);
            leanRoll = Mathf.Lerp(leanRoll, targetRoll, k);

            // Running bob: step rate and height grow with speed (walk ~1.8 steps/s up to ~3.2 at a sprint)
            if (speed > 0.3f)
            {
                float stepsPerSecond = Mathf.Lerp(1.8f, 3.2f, Mathf.Clamp01(speed / 8f));
                phase += Mathf.PI * stepsPerSecond * dt; // |sin| peaks once per step
            }
            float bob = Animated ? 0f : Mathf.Abs(Mathf.Sin(phase)) * s.bobHeight * Mathf.Clamp01(speed / 4f); // clips have their own bob

            lean.localPosition = new Vector3(0f, bob, 0f);
            lean.localRotation = Quaternion.Euler(leanPitch, 0f, leanRoll);
        }

        public void SetActive(bool on)
        {
            if (root.gameObject.activeSelf != on) root.gameObject.SetActive(on);
        }

        public bool Active => root.gameObject.activeSelf;
    }
}
