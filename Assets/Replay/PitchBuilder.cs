using UnityEngine;

namespace SoccerReplay
{
    /// <summary>Builds a regulation-marked pitch out of primitives. Pitch x -> Unity X, pitch y -> Unity Z.</summary>
    public static class PitchBuilder
    {
        const float LineWidth = 0.12f;
        const float LineHeight = 0.02f;

        public static Material MakeMaterial(Color color)
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) shader = Shader.Find("Standard");
            var m = new Material(shader) { color = color };
            if (m.HasProperty("_Smoothness")) m.SetFloat("_Smoothness", 0.1f);
            return m;
        }

        /// <summary>URP Lit material switched to alpha-blended transparency, so its color alpha can fade it.</summary>
        public static Material MakeTransparentMaterial(Color color) => MakeTransparent(MakeMaterial(color));

        /// <summary>Switch an existing URP Lit material (e.g. a textured character material) to alpha-blended transparency.</summary>
        public static Material MakeTransparent(Material m)
        {
            m.SetFloat("_Surface", 1f); // Transparent
            m.SetFloat("_Blend", 0f);   // Alpha
            m.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
            m.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            m.SetFloat("_SrcBlendAlpha", (float)UnityEngine.Rendering.BlendMode.One);
            m.SetFloat("_DstBlendAlpha", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            m.SetFloat("_ZWrite", 0f);
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            m.SetOverrideTag("RenderType", "Transparent");
            m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            return m;
        }

        public static GameObject Primitive(PrimitiveType type, string name, Transform parent, Material mat)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            go.transform.SetParent(parent, false);
            var col = go.GetComponent<Collider>();
            if (col != null) Object.Destroy(col);
            go.GetComponent<Renderer>().sharedMaterial = mat;
            return go;
        }

        public static Transform Build(Transform parent, float length, float width)
        {
            var root = new GameObject("Pitch").transform;
            root.SetParent(parent, false);

            float hl = length / 2f, hw = width / 2f;

            // Grass with mowing stripes
            var grassDark = MakeMaterial(new Color(0.16f, 0.45f, 0.18f));
            var grassLight = MakeMaterial(new Color(0.20f, 0.53f, 0.22f));
            var surround = Primitive(PrimitiveType.Plane, "Surround", root, grassDark);
            surround.transform.localScale = new Vector3((length + 16f) / 10f, 1f, (width + 12f) / 10f);

            int stripes = 14;
            float sw = length / stripes;
            for (int i = 0; i < stripes; i += 2)
            {
                var s = Primitive(PrimitiveType.Plane, "Stripe", root, grassLight);
                s.transform.localPosition = new Vector3(-hl + sw * (i + 0.5f), 0.002f, 0f);
                s.transform.localScale = new Vector3(sw / 10f, 1f, width / 10f);
            }

            // Lines
            var white = MakeMaterial(Color.white);
            var lines = new GameObject("Lines").transform;
            lines.SetParent(root, false);

            Rect(lines, white, -hl, hl, -hw, hw);                // touchlines + goal lines
            Line(lines, white, new Vector2(0, -hw), new Vector2(0, hw)); // halfway line
            Circle(lines, white, Vector2.zero, 9.15f, 64, null);
            Spot(lines, white, Vector2.zero);

            foreach (int side in new[] { -1, 1 })
            {
                float gx = side * hl;
                // Penalty area (16.5 x 40.32) and goal area (5.5 x 18.32)
                Rect(lines, white, Mathf.Min(gx, gx - side * 16.5f), Mathf.Max(gx, gx - side * 16.5f), -20.16f, 20.16f);
                Rect(lines, white, Mathf.Min(gx, gx - side * 5.5f), Mathf.Max(gx, gx - side * 5.5f), -9.16f, 9.16f);
                var spot = new Vector2(gx - side * 11f, 0f);
                Spot(lines, white, spot);
                // Penalty arc: the part of the 9.15m circle outside the box
                float boxEdge = gx - side * 16.5f;
                Circle(lines, white, spot, 9.15f, 64, p => side > 0 ? p.x < boxEdge : p.x > boxEdge);
                Goal(root, white, gx, side);
            }

            return root;
        }

        static void Rect(Transform parent, Material mat, float x0, float x1, float y0, float y1)
        {
            Line(parent, mat, new Vector2(x0, y0), new Vector2(x1, y0));
            Line(parent, mat, new Vector2(x0, y1), new Vector2(x1, y1));
            Line(parent, mat, new Vector2(x0, y0), new Vector2(x0, y1));
            Line(parent, mat, new Vector2(x1, y0), new Vector2(x1, y1));
        }

        static void Line(Transform parent, Material mat, Vector2 a, Vector2 b)
        {
            var go = Primitive(PrimitiveType.Cube, "Line", parent, mat);
            Vector2 d = b - a;
            // Sit clearly above the grass stripes so distant lines don't flicker (z-fighting)
            go.transform.localPosition = new Vector3((a.x + b.x) / 2f, 0.03f, (a.y + b.y) / 2f);
            go.transform.localRotation = Quaternion.Euler(0f, -Mathf.Atan2(d.y, d.x) * Mathf.Rad2Deg, 0f);
            go.transform.localScale = new Vector3(d.magnitude + LineWidth, LineHeight, LineWidth);
        }

        static void Circle(Transform parent, Material mat, Vector2 c, float r, int segments, System.Func<Vector2, bool> keep)
        {
            for (int i = 0; i < segments; i++)
            {
                float a0 = i * Mathf.PI * 2f / segments, a1 = (i + 1) * Mathf.PI * 2f / segments;
                var p0 = c + new Vector2(Mathf.Cos(a0), Mathf.Sin(a0)) * r;
                var p1 = c + new Vector2(Mathf.Cos(a1), Mathf.Sin(a1)) * r;
                if (keep != null && !(keep(p0) && keep(p1))) continue;
                Line(parent, mat, p0, p1);
            }
        }

        static void Spot(Transform parent, Material mat, Vector2 p)
        {
            var go = Primitive(PrimitiveType.Cylinder, "Spot", parent, mat);
            go.transform.localPosition = new Vector3(p.x, 0.01f, p.y);
            go.transform.localScale = new Vector3(0.4f, 0.01f, 0.4f);
        }

        static void Goal(Transform parent, Material mat, float gx, int side)
        {
            const float halfMouth = 3.66f, height = 2.44f, post = 0.12f, depth = 2f;
            var goal = new GameObject(side > 0 ? "Goal East" : "Goal West").transform;
            goal.SetParent(parent, false);
            foreach (float z in new[] { -halfMouth, halfMouth })
            {
                var p = Primitive(PrimitiveType.Cube, "Post", goal, mat);
                p.transform.localPosition = new Vector3(gx, height / 2f, z);
                p.transform.localScale = new Vector3(post, height, post);
                var back = Primitive(PrimitiveType.Cube, "Stanchion", goal, mat);
                back.transform.localPosition = new Vector3(gx + side * depth / 2f, height, z);
                back.transform.localScale = new Vector3(depth, post / 2f, post / 2f);
            }
            var bar = Primitive(PrimitiveType.Cube, "Crossbar", goal, mat);
            bar.transform.localPosition = new Vector3(gx, height, 0f);
            bar.transform.localScale = new Vector3(post, post, halfMouth * 2f + post);

            // Back frame: rear posts and ground bars that hold the net
            float backX = gx + side * depth;
            float thin = post / 2f;
            foreach (float z in new[] { -halfMouth, halfMouth })
            {
                var rear = Primitive(PrimitiveType.Cube, "Rear Post", goal, mat);
                rear.transform.localPosition = new Vector3(backX, height / 2f, z);
                rear.transform.localScale = new Vector3(thin, height, thin);
                var ground = Primitive(PrimitiveType.Cube, "Ground Bar", goal, mat);
                ground.transform.localPosition = new Vector3(gx + side * depth / 2f, thin / 2f, z);
                ground.transform.localScale = new Vector3(depth, thin, thin);
            }
            var backGround = Primitive(PrimitiveType.Cube, "Ground Bar", goal, mat);
            backGround.transform.localPosition = new Vector3(backX, thin / 2f, 0f);
            backGround.transform.localScale = new Vector3(thin, thin, halfMouth * 2f);

            // Net on every surface: back, roof and both sides
            var netBack = NetPanel(goal, "Net Back", halfMouth * 2f, height);
            netBack.localPosition = new Vector3(backX, height / 2f, 0f);
            netBack.localRotation = Quaternion.Euler(0f, 90f, 0f);

            var roof = NetPanel(goal, "Net Roof", halfMouth * 2f, depth);
            roof.localPosition = new Vector3(gx + side * depth / 2f, height, 0f);
            roof.localRotation = Quaternion.Euler(90f, 90f, 0f);

            foreach (float z in new[] { -halfMouth, halfMouth })
            {
                var sidePanel = NetPanel(goal, "Net Side", depth, height);
                sidePanel.localPosition = new Vector3(gx + side * depth / 2f, height / 2f, z);
            }
        }

        // ---------------- Goal net ----------------

        const float NetCell = 0.12f;       // mesh hole size (m), like a real 12 cm goal net
        const int NetCellsPerTile = 4;     // cells drawn in one repeat of the texture
        static Texture2D netTexture;

        /// <summary>White cords on transparent cells; mipmaps fade it to a light haze at broadcast distance.</summary>
        static Texture2D NetTexture()
        {
            if (netTexture != null) return netTexture;
            const int size = 64, cell = size / NetCellsPerTile, cord = 2;
            netTexture = new Texture2D(size, size, TextureFormat.RGBA32, true) { name = "Goal Net", wrapMode = TextureWrapMode.Repeat, filterMode = FilterMode.Trilinear, anisoLevel = 8 };
            var px = new Color32[size * size];
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    bool line = x % cell < cord || y % cell < cord;
                    px[y * size + x] = line ? new Color32(255, 255, 255, 255) : new Color32(255, 255, 255, 0);
                }
            netTexture.SetPixels32(px);
            netTexture.Apply(true);
            return netTexture;
        }

        /// <summary>A double-sided see-through net panel of the given size (m), as a quad in its local XY plane.</summary>
        static Transform NetPanel(Transform parent, string name, float width, float height)
        {
            var m = MakeTransparent(MakeMaterial(new Color(1f, 1f, 1f, 0.95f)));
            m.SetTexture("_BaseMap", NetTexture());
            m.mainTextureScale = new Vector2(width / (NetCell * NetCellsPerTile), height / (NetCell * NetCellsPerTile));
            m.SetFloat("_Cull", 0f);   // visible from inside and outside the goal
            var q = Primitive(PrimitiveType.Quad, name, parent, m);
            q.transform.localScale = new Vector3(width, height, 1f);
            var r = q.GetComponent<Renderer>();
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            return q.transform;
        }
    }
}
