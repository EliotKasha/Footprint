using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;

namespace SoccerReplay.EditorTools
{
    /// <summary>
    /// The replay creates its materials in code (Shader.Find + keywords). A build only contains shaders and
    /// shader variants that some saved asset uses, so without help the standalone app is missing them
    /// (e.g. the sponsor-board Unlit shader, the sky, see-through players and nets).
    /// This writes one saved material per variant the game makes into Assets/Replay/Resources/BuildShaderVariants,
    /// using the same code paths as the game, so the build always includes them. It runs before every build,
    /// and from the menu Soccer Replay → Prepare Build Shaders.
    /// </summary>
    public class BuildShaderVariants : IPreprocessBuildWithReport
    {
        const string Folder = "Assets/Replay/Resources/BuildShaderVariants";
        public int callbackOrder => 0;

        public void OnPreprocessBuild(BuildReport report) => Generate();

        [MenuItem("Soccer Replay/Prepare Build Shaders")]
        public static void Generate()
        {
            Directory.CreateDirectory(Folder);
            int n = 0;
            void Save(string name, Material m)
            {
                if (m == null || m.shader == null) { Debug.LogWarning($"[Build shaders] skipped {name}: shader not found"); return; }
                string path = $"{Folder}/{name}.mat";
                AssetDatabase.DeleteAsset(path);
                AssetDatabase.CreateAsset(m, path);
                n++;
            }

            // URP Lit: solid, see-through, GPU-instanced (crowd), double-sided see-through (goal net)
            Save("Lit_Opaque", PitchBuilder.MakeMaterial(Color.white));
            var instanced = PitchBuilder.MakeMaterial(Color.white); instanced.enableInstancing = true;
            Save("Lit_Opaque_Instanced", instanced);
            Save("Lit_Transparent", PitchBuilder.MakeTransparentMaterial(new Color(1f, 1f, 1f, 0.5f)));
            var net = PitchBuilder.MakeTransparentMaterial(Color.white); net.SetFloat("_Cull", 0f);
            net.SetTexture("_BaseMap", Texture2D.whiteTexture);
            Save("Lit_Transparent_DoubleSided", net);

            // URP Unlit see-through with a texture (sponsor logos, pass line)
            var unlitShader = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlitShader != null)
            {
                var unlit = new Material(unlitShader);
                unlit.SetTexture("_BaseMap", Texture2D.whiteTexture);
                unlit.SetFloat("_Surface", 1f);
                unlit.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
                unlit.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
                unlit.SetFloat("_ZWrite", 0f);
                unlit.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                unlit.renderQueue = (int)RenderQueue.Transparent;
                Save("Unlit_Transparent", unlit);
            }

            // Sprites/Default: the stats overlay's formation lines (FormationLinkRenderer)
            var sprites = Shader.Find("Sprites/Default");
            if (sprites != null) Save("Sprites_Default", new Material(sprites));

            // Procedural sky
            var sky = Shader.Find("Skybox/Procedural");
            if (sky != null) Save("Skybox_Procedural", new Material(sky));

            // Textured characters: their own materials are in the build already (the scene references the model);
            // the see-through copies made for off-camera players are not, so save one of each
            var done = new System.Collections.Generic.HashSet<Material>();
            foreach (var guid in AssetDatabase.FindAssets("t:Model", new[] { "Assets/Characters" }))
            {
                var model = AssetDatabase.LoadAssetAtPath<GameObject>(AssetDatabase.GUIDToAssetPath(guid));
                if (model == null) continue;
                var renderers = model.GetComponentsInChildren<SkinnedMeshRenderer>(true);
                // Only dressed characters (shirt / shorts meshes) get see-through copies of their own materials
                if (!System.Array.Exists(renderers, r => r.name.ToLowerInvariant().Contains("shirt"))) continue;
                foreach (var r in renderers)
                {
                    var src = r.sharedMaterial;
                    if (src == null || !done.Add(src) || !src.shader.name.StartsWith("Universal Render Pipeline")) continue;
                    Save($"Ghost_{model.name}_{src.name}", PitchBuilder.MakeTransparent(new Material(src)));
                }
            }

            AssetDatabase.SaveAssets();
            Debug.Log($"[Build shaders] {n} variant materials saved in {Folder}");
        }
    }
}
