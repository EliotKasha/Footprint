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

        float speed, speedVel, yaw, phase;
        float leanPitch, leanRoll;
        bool hasYaw;

        public float Speed => speed;

        const float Height = 1.8f;
        static Material shortsMat;

        public static PlayerBody Create(string name, Transform parent, Material bodyMat, Material trimMat, string labelText, bool showLabel)
        {
            var b = new PlayerBody { bodyMat = bodyMat, trimMat = trimMat };
            b.root = new GameObject(name).transform;
            b.root.SetParent(parent, false);

            b.lean = new GameObject("Lean").transform;
            b.lean.SetParent(b.root, false);

            var body = PitchBuilder.Primitive(PrimitiveType.Capsule, "Body", b.lean, bodyMat);
            body.transform.localPosition = new Vector3(0f, Height / 2f, 0f);
            body.transform.localScale = new Vector3(0.7f, Height / 2f, 0.7f);

            // Chest stripe on the front: makes facing direction readable from any camera
            var stripe = PitchBuilder.Primitive(PrimitiveType.Cube, "Chest", b.lean, trimMat);
            stripe.transform.localPosition = new Vector3(0f, 1.32f, 0.31f);
            stripe.transform.localScale = new Vector3(0.36f, 0.11f, 0.12f);

            // Shorts band around the hips
            if (shortsMat == null) shortsMat = PitchBuilder.MakeMaterial(new Color(0.1f, 0.1f, 0.14f));
            var shorts = PitchBuilder.Primitive(PrimitiveType.Cylinder, "Shorts", b.lean, shortsMat);
            shorts.transform.localPosition = new Vector3(0f, 0.78f, 0f);
            shorts.transform.localScale = new Vector3(0.72f, 0.13f, 0.72f);

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
            return b;
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
            float targetPitch = Mathf.Clamp(fwd * s.leanPerMs, -s.maxLean * 0.4f, s.maxLean);
            float targetRoll = Mathf.Clamp(-side * s.leanPerMs * 0.8f, -s.maxLean * 0.5f, s.maxLean * 0.5f);
            float k = snap ? 1f : 1f - Mathf.Exp(-8f * dt);
            leanPitch = Mathf.Lerp(leanPitch, targetPitch, k);
            leanRoll = Mathf.Lerp(leanRoll, targetRoll, k);

            // Running bob: step rate and height grow with speed (walk ~1.8 steps/s up to ~3.2 at a sprint)
            if (speed > 0.3f)
            {
                float stepsPerSecond = Mathf.Lerp(1.8f, 3.2f, Mathf.Clamp01(speed / 8f));
                phase += Mathf.PI * stepsPerSecond * dt; // |sin| peaks once per step
            }
            float bob = Mathf.Abs(Mathf.Sin(phase)) * s.bobHeight * Mathf.Clamp01(speed / 4f);

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
