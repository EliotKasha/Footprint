using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace SoccerReplay
{
    /// <summary>
    /// Builds a stylised stadium around the pitch at runtime: a two-tier bowl with rounded corners,
    /// roof, ad boards, an instanced crowd, plus sun, fog and post-processing.
    /// Pitch x -> Unity X, pitch y -> Unity Z (same as PitchBuilder).
    /// </summary>
    public class StadiumBuilder : MonoBehaviour
    {
        [Header("Layout (meters)")]
        public float runoffSide = 8f;
        public float runoffEnd = 10f;
        public float cornerRadius = 10f;

        [Header("Stands")]
        public int lowerRows = 15;
        public int upperRows = 18;
        public float stepDepth = 0.8f;
        public float stepHeight = 0.42f;
        public float lowerBase = 1.2f;
        public float concourseGap = 2.5f;
        public float concourseHeight = 3f;
        public float aisleEvery = 9f;
        public float aisleWidth = 1.2f;

        [Header("Colors")]
        public Color homeSeat = new Color(0.72f, 0.12f, 0.13f);
        public Color awaySeat = new Color(0.16f, 0.30f, 0.78f);
        public Color concrete = new Color(0.58f, 0.58f, 0.60f);
        public Color darkConcrete = new Color(0.30f, 0.30f, 0.32f);
        public Color roofColor = new Color(0.62f, 0.64f, 0.67f);
        public Color exteriorColor = new Color(0.40f, 0.41f, 0.44f);
        public Color apronGrass = new Color(0.15f, 0.40f, 0.17f);
        public Color boardColor = new Color(0.06f, 0.09f, 0.20f);

        [Header("Crowd")]
        public bool crowd = true;
        [Range(0f, 1f)] public float crowdFill = 0.85f;
        public float seatSpacing = 0.9f;
        [Range(0f, 1f)] public float bobbingFans = 0.6f;

        [Header("Atmosphere")]
        public Material skybox;
        public bool postProcessing = true;
        public bool fog = true;
        public float sunElevation = 32f;
        public float sunAzimuth = -20f;

        [Header("Ad boards")]
        public string[] boardTexts = { "CODE JAM 2026", "SOCCER VISION AI", "TRACKED BY CV", "LIVE TACTICS", "3D MATCH REPLAY", "HOME  vs  AWAY" };
        [Tooltip("Crest shown on the boards (leave empty for text-only boards)")]
        public Texture2D boardLogo;
        [Tooltip("Text written next to the crest")]
        public string logoText = "SHEFFIELD THURSDAY";
        [Tooltip("Show the crest on every Nth board (1 = every board)")]
        [Min(1)] public int logoEvery = 1;
        [Tooltip("Cut the crest out as a circle (for screenshots with a square background)")]
        public bool circularLogo = true;
        [Tooltip("Sponsor ads; when set, the boards rotate through these instead of the single logo/texts above")]
        public List<BoardAd> boardAds = new List<BoardAd>();
        [Tooltip("Give every ad the same board color (below) instead of each ad's own background")]
        public bool uniformAdBackground = true;
        public Color adBoardColor = Color.white;
        [Tooltip("Like LED boards: each board flips to the next sponsor every few seconds")]
        public bool rotateAds = true;
        [Min(1f)] public float adRotateSeconds = 6f;
        [Min(0.05f)] public float adFlipSeconds = 0.4f;

        // Rotating boards: one content group per ad on each board
        readonly List<(GameObject[] groups, int offset)> rotatingBoards = new List<(GameObject[], int)>();
        int lastShownCycle = int.MinValue;

        [System.Serializable]
        public class BoardAd
        {
            public string name = "Sponsor";
            public Texture2D logo;
            [Tooltip("Text next to the logo (leave empty for a logo-only board)")]
            public string text;
            public Color background = new Color(0.06f, 0.09f, 0.20f);
            public Color textColor = Color.white;
            [Tooltip("Cut the logo out as a circle (for round crests on a square background)")]
            public bool circularCrop;
            [Tooltip("Logo height as a fraction of the board height")]
            [Range(0.2f, 1f)] public float logoHeight = 0.8f;
        }

        // Half extents of the stand front edge
        float A, B;

        struct PathPoint
        {
            public Vector2 p, n;
            public bool aisle, away;
        }
        readonly List<PathPoint> path = new List<PathPoint>();

        // Instanced crowd: one batch per color
        class CrowdGroup
        {
            public Material mat;
            public RenderParams rp;
            public readonly List<Matrix4x4> build = new List<Matrix4x4>();
            public readonly List<float> phase = new List<float>();
            public readonly List<float> amp = new List<float>();
            public Matrix4x4[] matrices;
            public float[] baseY;
        }
        readonly List<CrowdGroup> crowdGroups = new List<CrowdGroup>();
        Mesh crowdMesh;

        // Tier geometry summary (distances outward from the front edge)
        float lowerDepth, lowerTop, upperStart, upperBase, upperDepth, upperTop, roofInner, roofOuter, roofInnerH, roofOuterH;

        public float FrontHalfLength => A;
        public float FrontHalfWidth => B;

        public void Build(float pitchLength, float pitchWidth, Camera cam, Color homeKit, Color awayKit)
        {
            A = pitchLength / 2f + runoffEnd;
            B = pitchWidth / 2f + runoffSide;

            lowerDepth = lowerRows * stepDepth;
            lowerTop = lowerBase + lowerRows * stepHeight;
            upperStart = lowerDepth + concourseGap;
            upperBase = lowerTop + concourseHeight;
            upperDepth = upperRows * stepDepth;
            upperTop = upperBase + upperRows * stepHeight;
            roofInner = lowerDepth + 1f;
            roofOuter = upperStart + upperDepth + 1.5f;
            roofInnerH = upperTop + 5f;
            roofOuterH = upperTop + 3f;

            var root = new GameObject("Stadium").transform;
            root.SetParent(transform, false);

            BuildGround(root);
            BuildPath();
            BuildBowl(root);
            BuildRoof(root);
            BuildBoards(root, pitchLength, pitchWidth);
            BuildCornerFlags(root, pitchLength, pitchWidth);
            if (crowd) BuildCrowd(homeKit, awayKit);
            SetupAtmosphere(cam);
        }

        // ---------------- Ground ----------------

        void BuildGround(Transform root)
        {
            var apron = PitchBuilder.Primitive(PrimitiveType.Plane, "Apron", root, PitchBuilder.MakeMaterial(apronGrass));
            apron.transform.localPosition = new Vector3(0f, -0.03f, 0f);
            apron.transform.localScale = new Vector3(2f * A / 10f + 0.2f, 1f, 2f * B / 10f + 0.2f);

            var outside = PitchBuilder.Primitive(PrimitiveType.Plane, "Surroundings", root, PitchBuilder.MakeMaterial(new Color(0.42f, 0.43f, 0.44f)));
            outside.transform.localPosition = new Vector3(0f, -0.06f, 0f);
            outside.transform.localScale = new Vector3(80f, 1f, 80f);
        }

        // ---------------- Path around the pitch ----------------

        void BuildPath()
        {
            path.Clear();
            float rc = Mathf.Clamp(cornerRadius, 1f, Mathf.Min(A, B) - 1f);
            AddStraight(new Vector2(A, -(B - rc)), new Vector2(A, B - rc), new Vector2(1f, 0f));
            AddArc(new Vector2(A - rc, B - rc), rc, 0f, 90f);
            AddStraight(new Vector2(A - rc, B), new Vector2(-(A - rc), B), new Vector2(0f, 1f));
            AddArc(new Vector2(-(A - rc), B - rc), rc, 90f, 180f);
            AddStraight(new Vector2(-A, B - rc), new Vector2(-A, -(B - rc)), new Vector2(-1f, 0f));
            AddArc(new Vector2(-(A - rc), -(B - rc)), rc, 180f, 270f);
            AddStraight(new Vector2(-(A - rc), -B), new Vector2(A - rc, -B), new Vector2(0f, -1f));
            AddArc(new Vector2(A - rc, -(B - rc)), rc, 270f, 360f);

            // Away supporters in the north-east corner
            for (int i = 0; i < path.Count; i++)
            {
                var pp = path[i];
                pp.away = pp.p.x > A - 32f && pp.p.y > B - 26f;
                path[i] = pp;
            }
        }

        void AddStraight(Vector2 s, Vector2 e, Vector2 n)
        {
            float len = Vector2.Distance(s, e);
            Vector2 dir = (e - s) / len;
            float t = 0f;
            bool aisle = false;
            // Start with half a seat block so aisles fall symmetrically
            float first = (len % (aisleEvery + aisleWidth)) * 0.5f + aisleEvery * 0.5f;
            path.Add(new PathPoint { p = s, n = n, aisle = false });
            t = first;
            aisle = true;
            while (t < len - 0.05f)
            {
                path.Add(new PathPoint { p = s + dir * t, n = n, aisle = aisle });
                t += aisle ? aisleWidth : aisleEvery;
                aisle = !aisle;
            }
        }

        void AddArc(Vector2 c, float r, float a0, float a1)
        {
            const int segs = 10;
            for (int i = 0; i < segs; i++)
            {
                float a = Mathf.Lerp(a0, a1, i / (float)segs) * Mathf.Deg2Rad;
                var n = new Vector2(Mathf.Cos(a), Mathf.Sin(a));
                path.Add(new PathPoint { p = c + n * r, n = n, aisle = false });
            }
        }

        static Vector3 At(PathPoint pp, float outward, float y) =>
            new Vector3(pp.p.x + pp.n.x * outward, y, pp.p.y + pp.n.y * outward);

        static Vector3 Out(PathPoint pp) => new Vector3(pp.n.x, 0f, pp.n.y);

        // ---------------- Mesh helpers ----------------

        class MeshBuilder
        {
            public readonly List<Vector3> v = new List<Vector3>();
            public readonly List<int>[] tris;
            public MeshBuilder(int submeshes)
            {
                tris = new List<int>[submeshes];
                for (int i = 0; i < submeshes; i++) tris[i] = new List<int>();
            }

            /// <summary>Adds a quad and makes sure it faces the given direction.</summary>
            public void Quad(int sub, Vector3 a, Vector3 b, Vector3 c, Vector3 d, Vector3 facing)
            {
                if (Vector3.Dot(Vector3.Cross(b - a, c - a), facing) < 0f) { var t = b; b = d; d = t; }
                int i = v.Count;
                v.Add(a); v.Add(b); v.Add(c); v.Add(d);
                var l = tris[sub];
                l.Add(i); l.Add(i + 1); l.Add(i + 2);
                l.Add(i); l.Add(i + 2); l.Add(i + 3);
            }

            public Mesh ToMesh(string name)
            {
                var m = new Mesh { name = name, indexFormat = IndexFormat.UInt32 };
                m.SetVertices(v);
                m.subMeshCount = tris.Length;
                for (int i = 0; i < tris.Length; i++) m.SetTriangles(tris[i], i);
                m.RecalculateNormals();
                m.RecalculateBounds();
                return m;
            }
        }

        GameObject MeshObject(string name, Transform parent, Mesh mesh, Material[] mats, bool castShadows = true)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterials = mats;
            mr.shadowCastingMode = castShadows ? ShadowCastingMode.On : ShadowCastingMode.Off;
            return go;
        }

        // ---------------- Bowl ----------------

        // Submeshes: 0 home seats, 1 home seats (alt row), 2 away seats, 3 away seats (alt row), 4 concrete, 5 dark concrete
        void BuildBowl(Transform root)
        {
            var mb = new MeshBuilder(7);
            Tier(mb, 0f, lowerBase, lowerRows, 0f);
            Tier(mb, upperStart, upperBase, upperRows, 0f);

            int n = path.Count;
            for (int i = 0; i < n; i++)
            {
                var p0 = path[i]; var p1 = path[(i + 1) % n];
                Vector3 inward = -(Out(p0) + Out(p1)).normalized;

                // Concourse floor between the tiers
                mb.Quad(4, At(p0, lowerDepth, lowerTop), At(p1, lowerDepth, lowerTop), At(p1, upperStart, lowerTop), At(p0, upperStart, lowerTop), Vector3.up);
                // Upper tier front wall (visible above the concourse)
                mb.Quad(5, At(p0, upperStart, 0f), At(p1, upperStart, 0f), At(p1, upperStart, upperBase), At(p0, upperStart, upperBase), inward);
                // Lower tier front wall (behind the ad boards)
                mb.Quad(5, At(p0, 0f, 0f), At(p1, 0f, 0f), At(p1, 0f, lowerBase), At(p0, 0f, lowerBase), inward);
                // Lower tier back wall
                mb.Quad(4, At(p0, lowerDepth, 0f), At(p1, lowerDepth, 0f), At(p1, lowerDepth, lowerTop), At(p0, lowerDepth, lowerTop), -inward);
                // Outer wall up to the roof, both faces
                float back = upperStart + upperDepth;
                mb.Quad(6, At(p0, back, 0f), At(p1, back, 0f), At(p1, back, roofOuterH), At(p0, back, roofOuterH), -inward);
                mb.Quad(5, At(p0, back, upperTop), At(p1, back, upperTop), At(p1, back, roofOuterH), At(p0, back, roofOuterH), inward);
            }

            var mats = new[]
            {
                PitchBuilder.MakeMaterial(homeSeat),
                PitchBuilder.MakeMaterial(homeSeat * 0.85f + Color.black * 0.15f),
                PitchBuilder.MakeMaterial(awaySeat),
                PitchBuilder.MakeMaterial(awaySeat * 0.85f + Color.black * 0.15f),
                PitchBuilder.MakeMaterial(concrete),
                PitchBuilder.MakeMaterial(darkConcrete),
                PitchBuilder.MakeMaterial(exteriorColor),
            };
            MeshObject("Stands", root, mb.ToMesh("Stands"), mats);
        }

        void Tier(MeshBuilder mb, float start, float baseH, int rows, float unused)
        {
            int n = path.Count;
            for (int i = 0; i < n; i++)
            {
                var p0 = path[i]; var p1 = path[(i + 1) % n];
                Vector3 inward = -(Out(p0) + Out(p1)).normalized;
                for (int r = 0; r < rows; r++)
                {
                    float z0 = start + r * stepDepth, z1 = z0 + stepDepth;
                    float y0 = baseH + r * stepHeight, y1 = y0 + stepHeight;
                    // Riser
                    mb.Quad(4, At(p0, z0, y0), At(p1, z0, y0), At(p1, z0, y1), At(p0, z0, y1), inward);
                    // Tread (seats), aisles are bare concrete
                    int sub = p0.aisle ? 4 : (p0.away ? 2 : 0) + (r % 2);
                    mb.Quad(sub, At(p0, z0, y1), At(p1, z0, y1), At(p1, z1, y1), At(p0, z1, y1), Vector3.up);
                }
            }
        }

        // ---------------- Roof ----------------

        void BuildRoof(Transform root)
        {
            var mb = new MeshBuilder(2);
            int n = path.Count;
            for (int i = 0; i < n; i++)
            {
                var p0 = path[i]; var p1 = path[(i + 1) % n];
                Vector3 inward = -(Out(p0) + Out(p1)).normalized;
                Vector3 a = At(p0, roofInner, roofInnerH), b = At(p1, roofInner, roofInnerH);
                Vector3 c = At(p1, roofOuter, roofOuterH), d = At(p0, roofOuter, roofOuterH);
                mb.Quad(0, a, b, c, d, Vector3.up);
                mb.Quad(0, a, b, c, d, Vector3.down);
                // Dark fascia along the inner edge
                const float fascia = 1.4f;
                Vector3 a2 = a + Vector3.down * fascia, b2 = b + Vector3.down * fascia;
                mb.Quad(1, a2, b2, b, a, inward);
                mb.Quad(1, a2, b2, b, a, -inward);
            }
            var mats = new[] { PitchBuilder.MakeMaterial(roofColor), PitchBuilder.MakeMaterial(darkConcrete) };
            MeshObject("Roof", root, mb.ToMesh("Roof"), mats);
        }

        // ---------------- Pitchside ----------------

        void BuildBoards(Transform root, float L, float W)
        {
            var boards = new GameObject("Ad Boards").transform;
            boards.SetParent(root, false);
            var boardMat = PitchBuilder.MakeMaterial(boardColor);
            var logoMat = boardLogo != null ? MakeLogoMaterial(boardLogo, circularLogo) : null;
            int textIdx = 0, boardIdx = 0;
            const float panel = 8f, height = 0.95f;

            // Sponsor ads: boards rotate through the list (each with its own logo, text and board color)
            var ads = new List<(BoardAd ad, Material board, Material logo, float aspect)>();
            if (boardAds != null)
                foreach (var ad in boardAds)
                {
                    if (ad == null || (ad.logo == null && string.IsNullOrEmpty(ad.text))) continue;
                    Material lm = null; float aspect = 1f;
                    if (ad.logo != null)
                    {
                        lm = MakeLogoMaterial(ad.logo, ad.circularCrop);
                        var t = lm.GetTexture("_BaseMap");
                        aspect = t != null && t.height > 0 ? (float)t.width / t.height : 1f;
                    }
                    ads.Add((ad, PitchBuilder.MakeMaterial(uniformAdBackground ? adBoardColor : ad.background), lm, aspect));
                }
            rotatingBoards.Clear();

            void Row(Vector2 from, Vector2 to, Vector2 faceDir)
            {
                float len = Vector2.Distance(from, to);
                int count = Mathf.Max(1, Mathf.RoundToInt(len / panel));
                Vector2 step = (to - from) / count;
                var face = new Vector3(faceDir.x, 0f, faceDir.y);
                for (int i = 0; i < count; i++)
                {
                    Vector2 c = from + step * (i + 0.5f);
                    if (ads.Count > 0)
                    {
                        int idx = boardIdx++;
                        if (rotateAds && ads.Count > 1)
                        {
                            // One panel, with every sponsor's content stacked on it; Update() shows one at a time
                            var groups = new GameObject[ads.Count];
                            for (int k = 0; k < ads.Count; k++)
                            {
                                var a = ads[k];
                                groups[k] = BuildAdBoard(boards, c, step.magnitude - 0.1f, height, face, a.ad, a.board, a.logo, a.aspect, panel: k == 0);
                                groups[k].SetActive(false);
                            }
                            rotatingBoards.Add((groups, idx));
                        }
                        else
                        {
                            var a = ads[idx % ads.Count];
                            BuildAdBoard(boards, c, step.magnitude - 0.1f, height, face, a.ad, a.board, a.logo, a.aspect, panel: true);
                        }
                        continue;
                    }
                    var board = PitchBuilder.Primitive(PrimitiveType.Cube, "Board", boards, boardMat);
                    board.transform.localPosition = new Vector3(c.x, height / 2f, c.y);
                    board.transform.localRotation = Quaternion.LookRotation(face);
                    board.transform.localScale = new Vector3(step.magnitude - 0.1f, height, 0.15f);

                    var faceRot = Quaternion.LookRotation(-face);       // readable from the pitch side
                    Vector3 readRight = faceRot * Vector3.right;          // the viewer's "right" along the board
                    Vector3 center = new Vector3(c.x, height / 2f, c.y) + face * 0.09f;
                    float boardLen = step.magnitude - 0.1f;
                    bool withLogo = logoMat != null && boardIdx++ % logoEvery == 0;

                    var tgo = new GameObject("Text");
                    tgo.transform.SetParent(boards, false);
                    tgo.transform.localRotation = faceRot;
                    var tmp = tgo.AddComponent<TextMeshPro>();
                    tmp.fontStyle = FontStyles.Bold;
                    tmp.enableAutoSizing = true;
                    tmp.fontSizeMin = 2f;
                    tmp.fontSizeMax = 6f;

                    if (withLogo)
                    {
                        // Crest on the left, club name to its right, centered as a group
                        float logoSize = height * 0.88f;
                        float textWidth = Mathf.Min(boardLen - logoSize - 0.6f, 5.2f);
                        float groupWidth = logoSize + 0.25f + textWidth;
                        float left = -groupWidth / 2f;

                        var logo = GameObject.CreatePrimitive(PrimitiveType.Quad);
                        logo.name = "Crest";
                        Object.Destroy(logo.GetComponent<Collider>());
                        logo.transform.SetParent(boards, false);
                        logo.transform.localPosition = center + readRight * (left + logoSize / 2f) + face * 0.005f;
                        logo.transform.localRotation = faceRot;
                        logo.transform.localScale = new Vector3(logoSize, logoSize, 1f);
                        var lr = logo.GetComponent<MeshRenderer>();
                        lr.sharedMaterial = logoMat;
                        lr.shadowCastingMode = ShadowCastingMode.Off;

                        tgo.transform.localPosition = center + readRight * (left + logoSize + 0.25f + textWidth / 2f);
                        tmp.text = logoText;
                        tmp.alignment = TextAlignmentOptions.Left;
                        tmp.color = Color.white;
                        tmp.rectTransform.sizeDelta = new Vector2(textWidth, height * 0.8f);
                    }
                    else
                    {
                        tgo.transform.localPosition = center;
                        tmp.text = boardTexts.Length > 0 ? boardTexts[textIdx++ % boardTexts.Length] : "";
                        tmp.alignment = TextAlignmentOptions.Center;
                        tmp.color = (textIdx % 2 == 0) ? new Color(1f, 0.85f, 0.2f) : Color.white;
                        tmp.rectTransform.sizeDelta = new Vector2(step.magnitude - 0.4f, height);
                    }
                }
            }

            float hl = L / 2f, hw = W / 2f;
            float sideZ = hw + 4f, endX = hl + 5f;
            Row(new Vector2(-hl - 2f, -sideZ), new Vector2(hl + 2f, -sideZ), Vector2.up);    // near touchline, faces pitch
            Row(new Vector2(-hl - 2f, sideZ), new Vector2(hl + 2f, sideZ), Vector2.down);    // far touchline
            Row(new Vector2(-endX, -hw), new Vector2(-endX, hw), Vector2.right);             // west goal line
            Row(new Vector2(endX, -hw), new Vector2(endX, hw), Vector2.left);                // east goal line
        }

        /// <summary>
        /// One sponsor's content on a board: the logo and (optionally) text, centered as a group. Returns the
        /// group so rotating boards can flip between sponsors. With panel=true it also builds the board itself.
        /// </summary>
        GameObject BuildAdBoard(Transform parent, Vector2 c, float boardLen, float height, Vector3 face,
                                BoardAd ad, Material boardMat, Material logoMat, float aspect, bool panel)
        {
            if (panel)
            {
                var board = PitchBuilder.Primitive(PrimitiveType.Cube, $"Board ({ad.name})", parent, boardMat);
                board.transform.localPosition = new Vector3(c.x, height / 2f, c.y);
                board.transform.localRotation = Quaternion.LookRotation(face);
                board.transform.localScale = new Vector3(boardLen, height, 0.15f);
            }

            var faceRot = Quaternion.LookRotation(-face);   // readable from the pitch side
            // Content group sits on the board face; its local x runs along the board as the viewer reads it
            var groupGo = new GameObject($"Ad ({ad.name})");
            groupGo.transform.SetParent(parent, false);
            groupGo.transform.localPosition = new Vector3(c.x, height / 2f, c.y) + face * 0.09f;
            groupGo.transform.localRotation = faceRot;
            var groupT = groupGo.transform;
            Vector3 readRight = Vector3.right;
            Vector3 center = Vector3.zero;
            face = Vector3.back;                              // "toward the pitch" in the group's local space
            parent = groupT;
            faceRot = Quaternion.identity;
            Color textColor = uniformAdBackground ? (adBoardColor.grayscale > 0.5f ? new Color(0.08f, 0.12f, 0.24f) : Color.white) : ad.textColor;

            bool hasText = !string.IsNullOrEmpty(ad.text);
            float logoH = height * Mathf.Clamp(ad.logoHeight, 0.2f, 1f);
            float logoW = logoMat != null ? Mathf.Min(logoH * aspect, boardLen * (hasText ? 0.45f : 0.85f)) : 0f;
            if (logoMat != null) logoH = logoW / aspect;
            const float gap = 0.3f;
            float textW = hasText ? Mathf.Min(boardLen - logoW - gap - 0.6f, 5.2f) : 0f;
            float group = logoW + (logoMat != null && hasText ? gap : 0f) + textW;
            float left = -group / 2f;

            if (logoMat != null)
            {
                var logo = GameObject.CreatePrimitive(PrimitiveType.Quad);
                logo.name = $"Logo ({ad.name})";
                Object.Destroy(logo.GetComponent<Collider>());
                logo.transform.SetParent(parent, false);
                logo.transform.localPosition = center + readRight * (left + logoW / 2f) + face * 0.005f;
                logo.transform.localRotation = faceRot;
                logo.transform.localScale = new Vector3(logoW, logoH, 1f);
                var lr = logo.GetComponent<MeshRenderer>();
                lr.sharedMaterial = logoMat;
                lr.shadowCastingMode = ShadowCastingMode.Off;
            }
            if (hasText)
            {
                var tgo = new GameObject("Text");
                tgo.transform.SetParent(parent, false);
                tgo.transform.localRotation = faceRot;
                tgo.transform.localPosition = center + readRight * (left + group - textW / 2f);
                var tmp = tgo.AddComponent<TextMeshPro>();
                tmp.text = ad.text;
                tmp.fontStyle = FontStyles.Bold;
                tmp.enableAutoSizing = true;
                tmp.fontSizeMin = 2f;
                tmp.fontSizeMax = 6f;
                tmp.alignment = logoMat != null ? TextAlignmentOptions.Left : TextAlignmentOptions.Center;
                tmp.color = textColor;
                tmp.rectTransform.sizeDelta = new Vector2(textW, height * 0.8f);
            }
            return groupGo;
        }

        /// <summary>
        /// LED-style rotation: every adRotateSeconds each board flips (squash and stretch vertically) to the next
        /// sponsor. Boards are offset so neighbours show different sponsors, and all change together.
        /// </summary>
        void UpdateRotatingAds()
        {
            if (rotatingBoards.Count == 0) return;
            float period = Mathf.Max(1f, adRotateSeconds);
            float t = Time.time;
            int cycle = Mathf.FloorToInt(t / period);
            float into = t - cycle * period;
            float flip = Mathf.Min(adFlipSeconds, period * 0.5f);
            bool flipping = into < flip;
            // First half of the flip: previous sponsor squashes away; second half: next one stretches in
            int shownCycle = flipping && into < flip * 0.5f ? cycle - 1 : cycle;
            float scaleY = flipping ? Mathf.Abs(Mathf.Cos(Mathf.PI * into / flip)) : 1f;

            bool changed = shownCycle != lastShownCycle;
            lastShownCycle = shownCycle;
            foreach (var (groups, offset) in rotatingBoards)
            {
                int n = groups.Length;
                int show = ((shownCycle + offset) % n + n) % n;
                for (int k = 0; k < n; k++)
                {
                    bool on = k == show;
                    if (changed && groups[k].activeSelf != on) groups[k].SetActive(on);
                    if (on) groups[k].transform.localScale = new Vector3(1f, Mathf.Max(0.001f, scaleY), 1f);
                }
            }
        }

        /// <summary>Unlit, alpha-blended material for a logo so it shows its true colors on the board.</summary>
        Material MakeLogoMaterial(Texture2D src, bool circular)
        {
            Texture2D tex = circular ? CutOutCircle(src) : src;
            var shader = Shader.Find("Universal Render Pipeline/Unlit");
            var m = new Material(shader);
            m.SetTexture("_BaseMap", tex);
            m.SetColor("_BaseColor", Color.white);
            m.SetFloat("_Surface", 1f);
            m.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
            m.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
            m.SetFloat("_ZWrite", 0f);
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            m.renderQueue = (int)RenderQueue.Transparent;
            return m;
        }

        /// <summary>
        /// Makes everything outside the crest's circular white rim transparent. The rim is found by scanning
        /// from the image center outwards for the last bright pixel, so it adapts to how the screenshot was cropped.
        /// </summary>
        static Texture2D CutOutCircle(Texture2D src)
        {
            if (!src.isReadable) { Debug.LogWarning("[Stadium] Logo texture isn't readable; using it uncropped."); return src; }
            int w = src.width, h = src.height;
            var px = src.GetPixels32();
            int cx = w / 2, cy = h / 2;
            bool Bright(int x, int y) { var p = px[y * w + x]; return p.r > 200 && p.g > 200 && p.b > 200; }

            int left = 0, right = w - 1, bottom = 0, top = h - 1;
            for (int x = 0; x < cx; x++) if (Bright(x, cy)) { left = x; break; }
            for (int x = w - 1; x > cx; x--) if (Bright(x, cy)) { right = x; break; }
            for (int y = 0; y < cy; y++) if (Bright(cx, y)) { bottom = y; break; }
            for (int y = h - 1; y > cy; y--) if (Bright(cx, y)) { top = y; break; }
            float ccx = (left + right) / 2f, ccy = (bottom + top) / 2f;
            float r = Mathf.Min(right - left, top - bottom) / 2f;

            // Crop to the circle's bounding square and feather the edge slightly
            int size = Mathf.Max(8, Mathf.CeilToInt(r * 2f) + 2);
            var outTex = new Texture2D(size, size, TextureFormat.RGBA32, true);
            var outPx = new Color32[size * size];
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    float sx = ccx - size / 2f + x, sy = ccy - size / 2f + y;
                    int ix = Mathf.Clamp(Mathf.RoundToInt(sx), 0, w - 1), iy = Mathf.Clamp(Mathf.RoundToInt(sy), 0, h - 1);
                    var p = px[iy * w + ix];
                    float d = Mathf.Sqrt((sx - ccx) * (sx - ccx) + (sy - ccy) * (sy - ccy));
                    p.a = (byte)(255f * Mathf.Clamp01(r + 0.5f - d));
                    outPx[y * size + x] = p;
                }
            outTex.SetPixels32(outPx);
            outTex.wrapMode = TextureWrapMode.Clamp;
            outTex.Apply(true);
            return outTex;
        }

        void BuildCornerFlags(Transform root, float L, float W)
        {
            var poleMat = PitchBuilder.MakeMaterial(new Color(0.95f, 0.95f, 0.95f));
            var flagMat = PitchBuilder.MakeMaterial(new Color(1f, 0.85f, 0.1f));
            foreach (int sx in new[] { -1, 1 })
                foreach (int sz in new[] { -1, 1 })
                {
                    var pole = PitchBuilder.Primitive(PrimitiveType.Cylinder, "Corner Flag", root, poleMat);
                    pole.transform.localPosition = new Vector3(sx * L / 2f, 0.75f, sz * W / 2f);
                    pole.transform.localScale = new Vector3(0.05f, 0.75f, 0.05f);
                    var flag = PitchBuilder.Primitive(PrimitiveType.Cube, "Flag", pole.transform, flagMat);
                    flag.transform.localPosition = new Vector3(0f, 0.8f, 0f);
                    flag.transform.localScale = new Vector3(0.4f / 0.05f, 0.3f / 0.75f, 0.01f / 0.05f);
                    flag.transform.localPosition = new Vector3(0.2f / 0.05f, 0.8f, 0f);
                }
        }

        // ---------------- Crowd ----------------

        void BuildCrowd(Color homeKit, Color awayKit)
        {
            var tmp = GameObject.CreatePrimitive(PrimitiveType.Cube);
            crowdMesh = tmp.GetComponent<MeshFilter>().sharedMesh;
            Destroy(tmp);

            Color[] palette =
            {
                homeSeat, homeKit, Color.Lerp(homeKit, Color.white, 0.35f), // 0-2 home colors
                new Color(0.92f, 0.92f, 0.92f), new Color(0.12f, 0.12f, 0.13f), new Color(0.45f, 0.45f, 0.48f), // 3-5 neutral clothing
                awayKit, Color.Lerp(awayKit, Color.white, 0.35f), // 6-7 away colors
            };
            foreach (var c in palette)
            {
                var m = PitchBuilder.MakeMaterial(c);
                m.enableInstancing = true;
                crowdGroups.Add(new CrowdGroup
                {
                    mat = m,
                    rp = new RenderParams(m) { shadowCastingMode = ShadowCastingMode.Off, receiveShadows = true, worldBounds = new Bounds(Vector3.zero, Vector3.one * 600f) }
                });
            }

            var rng = new System.Random(1234);
            var scale = new Vector3(0.45f, 0.8f, 0.3f);
            void Fill(float start, float baseH, int rows)
            {
                int n = path.Count;
                for (int i = 0; i < n; i++)
                {
                    var p0 = path[i]; var p1 = path[(i + 1) % n];
                    if (p0.aisle) continue;
                    var rot = Quaternion.LookRotation(-(Out(p0) + Out(p1)).normalized);
                    for (int r = 0; r < rows; r++)
                    {
                        float z = start + (r + 0.45f) * stepDepth;
                        float y = baseH + (r + 1) * stepHeight + scale.y / 2f;
                        Vector3 a = At(p0, z, y), b = At(p1, z, y);
                        int seats = Mathf.FloorToInt(Vector3.Distance(a, b) / seatSpacing);
                        for (int s = 0; s < seats; s++)
                        {
                            if (rng.NextDouble() > crowdFill) continue;
                            int g;
                            double roll = rng.NextDouble();
                            if (p0.away) g = roll < 0.75 ? 6 + rng.Next(2) : 3 + rng.Next(3);
                            else g = roll < 0.7 ? rng.Next(3) : 3 + rng.Next(3);
                            var pos = Vector3.Lerp(a, b, (s + 0.5f) / seats) + (Vector3.up * (float)(rng.NextDouble() * 0.08));
                            var grp = crowdGroups[g];
                            grp.build.Add(transform.localToWorldMatrix * Matrix4x4.TRS(pos, rot, scale));
                            grp.phase.Add((float)(rng.NextDouble() * Mathf.PI * 2f));
                            grp.amp.Add(rng.NextDouble() < bobbingFans ? 0.03f + (float)rng.NextDouble() * 0.04f : 0f);
                        }
                    }
                }
            }
            Fill(0f, lowerBase, lowerRows);
            Fill(upperStart, upperBase, upperRows);

            int total = 0;
            foreach (var g in crowdGroups)
            {
                g.matrices = g.build.ToArray();
                g.baseY = new float[g.matrices.Length];
                for (int i = 0; i < g.matrices.Length; i++) g.baseY[i] = g.matrices[i].m13;
                total += g.matrices.Length;
            }
            Debug.Log($"[Stadium] Built bowl ({path.Count} path segments), crowd of {total}");
        }

        void Update()
        {
            UpdateRotatingAds();
            if (crowdGroups.Count == 0 || crowdMesh == null) return;
            float t = Time.time;
            foreach (var g in crowdGroups)
            {
                var m = g.matrices;
                for (int i = 0; i < m.Length; i++)
                {
                    float a = g.amp[i];
                    if (a > 0f) m[i].m13 = g.baseY[i] + a * Mathf.Sin(t * 2.4f + g.phase[i]);
                }
                const int batch = 1000;
                for (int start = 0; start < m.Length; start += batch)
                    Graphics.RenderMeshInstanced(g.rp, crowdMesh, 0, m, Mathf.Min(batch, m.Length - start), start);
            }
        }

        // ---------------- Lighting & atmosphere ----------------

        void SetupAtmosphere(Camera cam)
        {
            var sun = RenderSettings.sun != null ? RenderSettings.sun : FindAnyObjectByType<Light>();
            if (sun != null && sun.type == LightType.Directional)
            {
                sun.transform.rotation = Quaternion.Euler(sunElevation, sunAzimuth, 0f);
                sun.intensity = 1.5f;
                sun.color = new Color(1f, 0.95f, 0.86f);
                sun.shadows = LightShadows.Soft;
                sun.shadowStrength = 0.8f;
            }

            if (GraphicsSettings.currentRenderPipeline is UniversalRenderPipelineAsset urp)
            {
                urp.shadowDistance = Mathf.Max(urp.shadowDistance, 280f);
                urp.shadowCascadeCount = 4;
                urp.msaaSampleCount = 4; // smooths thin distant pitch lines
            }

            if (skybox != null)
            {
                RenderSettings.skybox = skybox;
            }
            else
            {
                // Physically based sky that matches the sun direction and the haze
                var sky = Shader.Find("Skybox/Procedural");
                if (sky != null)
                {
                    var m = new Material(sky);
                    m.SetFloat("_SunSize", 0.035f);
                    m.SetFloat("_SunSizeConvergence", 6f);
                    m.SetFloat("_AtmosphereThickness", 0.9f);
                    m.SetColor("_SkyTint", new Color(0.52f, 0.60f, 0.72f));
                    m.SetColor("_GroundColor", new Color(0.40f, 0.41f, 0.43f));
                    m.SetFloat("_Exposure", 1.25f);
                    RenderSettings.skybox = m;
                }
            }
            if (sun != null) RenderSettings.sun = sun;
            DynamicGI.UpdateEnvironment();

            if (fog)
            {
                RenderSettings.fog = true;
                RenderSettings.fogMode = FogMode.Linear;
                RenderSettings.fogColor = new Color(0.76f, 0.81f, 0.87f);
                RenderSettings.fogStartDistance = 110f;
                RenderSettings.fogEndDistance = 650f;
            }

            if (postProcessing && cam != null)
            {
                var data = cam.GetUniversalAdditionalCameraData();
                data.renderPostProcessing = true;
                data.antialiasing = AntialiasingMode.SubpixelMorphologicalAntiAliasing;

                var go = new GameObject("Stadium Post FX");
                go.transform.SetParent(transform, false);
                var vol = go.AddComponent<Volume>();
                vol.isGlobal = true;
                vol.priority = 10f;
                var prof = ScriptableObject.CreateInstance<VolumeProfile>();
                var tm = prof.Add<Tonemapping>(true); tm.mode.Override(TonemappingMode.ACES);
                var ca = prof.Add<ColorAdjustments>(true);
                ca.postExposure.Override(0.35f);
                ca.contrast.Override(10f);
                ca.saturation.Override(8f);
                var bloom = prof.Add<Bloom>(true); bloom.intensity.Override(0.35f); bloom.threshold.Override(1.1f);
                var vig = prof.Add<Vignette>(true); vig.intensity.Override(0.2f); vig.smoothness.Override(0.45f);
                vol.sharedProfile = prof;
            }
        }
    }
}
