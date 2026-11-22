// FormationLinkRenderer.cs
// Draws the pink bars on the pitch between neighbouring players of the same line (back / middle / front).
// The bars follow the RENDERED 3D players (so they stay glued to the smoothed models), fade in and out
// when a player leaves or enters the camera, and get a little thicker the more the two players pass to
// each other (recent completed passes, from the events list).
using System.Collections.Generic;
using UnityEngine;

namespace PitchOverlay
{
    public class FormationLinkRenderer : MonoBehaviour
    {
        [Header("Look")]
        public Color linkColor = new Color(1f, 0.24f, 0.55f, 1f);   // hot pink
        [Range(0f, 1f)] public float maxOpacity = 0.85f;
        public float baseWidth = 0.35f;            // metres
        public float widthPerPass = 0.15f;         // extra width per recent pass between the pair
        public float maxWidth = 0.9f;
        public float groundY = 0.06f;              // slightly above the grass so it doesn't flicker
        public Material linkMaterial;              // optional: any transparent Unlit material

        [Header("Behaviour")]
        public float rebuildInterval = 0.2f;       // how often lines are re-worked out (seconds)
        public float fadeSpeed = 5f;               // higher = snappier fade
        public float passWindowSeconds = 45f;      // passes older than this stop thickening a bar
        public float maxLateral = 24f;             // players further apart sideways than this aren't linked
        public float maxDepthDifference = 10f;     // ...or at very different depths

        class Link
        {
            public LineRenderer lr;
            public int a, b;
            public float alpha, target, weight, idle;
        }

        readonly Dictionary<long, Link> links = new Dictionary<long, Link>();
        readonly List<long> toRemove = new List<long>();
        MatchAnalytics analytics;
        Material mat;
        float timer;

        public void Bind(MatchAnalytics a) { analytics = a; }

        void Awake()
        {
            mat = linkMaterial != null ? linkMaterial : new Material(Shader.Find("Sprites/Default"));
        }

        static long Key(int a, int b) { return a < b ? ((long)a << 32) | (uint)b : ((long)b << 32) | (uint)a; }

        void LateUpdate()
        {
            if (analytics == null) return;
            timer += Time.deltaTime;
            if (timer >= rebuildInterval) { timer = 0f; Rebuild(); }
            Animate();
        }

        void Rebuild()
        {
            foreach (var kv in links) kv.Value.target = 0f;       // anything not re-confirmed fades out

            for (int team = 0; team < 2; team++)
            {
                var pts = new List<Pt>();
                foreach (var d in PlayerMotionDriver.ById.Values)
                {
                    if (d.Team != team || d.IsKeeper || !d.IsActive) continue;
                    pts.Add(new Pt { id = d.Id, p = new Vector2(d.transform.position.x, d.transform.position.z) });
                }
                if (pts.Count < 2) continue;

                float sign = analytics.AttackSign(team);
                var lines = FormationShape.BuildLines(pts, sign);
                var pairs = FormationShape.BuildLinks(lines, sign, maxLateral, maxDepthDifference);
                foreach (var pair in pairs)
                {
                    long key = Key(pair.a.id, pair.b.id);
                    Link l;
                    if (!links.TryGetValue(key, out l))
                    {
                        l = new Link { a = pair.a.id, b = pair.b.id, lr = MakeLine() };
                        links[key] = l;
                    }
                    l.target = 1f;
                    l.idle = 0f;
                    l.weight = analytics.PassesBetween(pair.a.id, pair.b.id, analytics.time, passWindowSeconds);
                }
            }
        }

        void Animate()
        {
            toRemove.Clear();
            foreach (var kv in links)
            {
                var l = kv.Value;
                PlayerMotionDriver da = null, dbb = null;
                bool ok = PlayerMotionDriver.ById.TryGetValue(l.a, out da) &&
                          PlayerMotionDriver.ById.TryGetValue(l.b, out dbb) &&
                          da.IsActive && dbb.IsActive;

                float target = ok ? l.target : 0f;
                l.alpha = Mathf.MoveTowards(l.alpha, target, fadeSpeed * Time.deltaTime);
                l.idle = (l.alpha <= 0f && target <= 0f) ? l.idle + Time.deltaTime : 0f;

                bool visible = l.alpha > 0.01f && ok;
                l.lr.enabled = visible;
                if (visible)
                {
                    Vector3 pa = da.transform.position; pa.y = groundY;
                    Vector3 pb = dbb.transform.position; pb.y = groundY;
                    l.lr.SetPosition(0, pa);
                    l.lr.SetPosition(1, pb);
                    float w = Mathf.Min(maxWidth, baseWidth + widthPerPass * l.weight);
                    l.lr.startWidth = l.lr.endWidth = w;
                    var c = linkColor; c.a = linkColor.a * maxOpacity * l.alpha;
                    l.lr.startColor = l.lr.endColor = c;
                }
                if (l.idle > 3f) toRemove.Add(kv.Key);           // forget links that stayed gone
            }
            foreach (var k in toRemove) { Destroy(links[k].lr.gameObject); links.Remove(k); }
        }

        LineRenderer MakeLine()
        {
            var go = new GameObject("Link");
            go.transform.SetParent(transform, false);
            go.transform.rotation = Quaternion.Euler(90f, 0f, 0f);      // lie flat on the pitch
            var lr = go.AddComponent<LineRenderer>();
            lr.useWorldSpace = true;
            lr.alignment = LineAlignment.TransformZ;
            lr.positionCount = 2;
            lr.numCapVertices = 6;                                      // rounded ends
            lr.sharedMaterial = mat;
            lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            lr.receiveShadows = false;
            lr.enabled = false;
            return lr;
        }
    }
}
