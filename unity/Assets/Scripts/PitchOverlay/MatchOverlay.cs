// MatchOverlay.cs
// The one script you add to the scene. It loads match_events.json, keeps the analytics in step with the
// 3D replay, and feeds the pink bars and the stats panel.
// [FORMATION-ML] also loads the measured team shapes (analysis.json), and has LoadMatchFile(path) so the
// project's "Load match..." button can load a match file at runtime (+ its .analysis.json automatically).
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace PitchOverlay
{
    public class MatchOverlay : MonoBehaviour
    {
        public TextAsset matchJson;                     // match_events.json (same file MatchReplay uses) - optional if
                                                        // you call LoadMatchFile from your own loader instead
        public TextAsset shapeAnalysisJson;             // [FORMATION-ML] the matching *.analysis.json (optional)
        public StatsPanel panel;
        public FormationLinkRenderer links;
        public float renderDelay = 0.25f;               // must equal PlayerMotionDriver.renderDelay

        // [REPLAY] Formation chips measured live in Unity (LiveShapes = C# port of formation-ml's shape.py),
        // so only match_events.json is needed. Votes over the last `shapeWindow` seconds of the replay.
        [Header("Live team shape")]
        public bool liveShapes = true;
        public float shapeWindow = 20f;                 // seconds of play the shape is measured over (0 = whole clip so far)
        [Tooltip("Usable moments (3 per second) needed before a shape is shown. formation-ml uses 20 for a whole clip; " +
                 "fewer shows shapes sooner, marked low confidence (faded chip) until 8 s have been seen.")]
        public int shapeMinMoments = 9;
        public bool preferAnalysisFile = false;         // true: use a matching .analysis.json (whole-clip shapes) when present
        public LiveShapes Shapes { get; private set; }

        public MatchAnalytics Analytics { get; private set; }

        void Start()
        {
            if (matchJson) Load(matchJson.text, shapeAnalysisJson ? shapeAnalysisJson.text : null);
        }

        // [FORMATION-ML] Call this from the project's "Load match..." code with the file the user picked.
        // If "<same name>.analysis.json" sits next to it (made by formation-ml's export_for_unity.py),
        // the measured team shapes are loaded too. Example: match_events_v6.json -> match_events_v6.analysis.json
        public void LoadMatchFile(string path)
        {
            string analysisPath = Path.ChangeExtension(path, null) + ".analysis.json";
            string analysis = File.Exists(analysisPath) ? File.ReadAllText(analysisPath) : null;
            if (analysis == null) Debug.Log("[MatchOverlay] No team-shape file at " + analysisPath + " (team shapes are measured live).");
            // [REPLAY] only use the shapes if they were measured from this same clip (the file records its source)
            if (analysis != null)
            {
                string source = null;
                try { var info = JsonConvert.DeserializeObject<ShapeAnalysisFile>(analysis); if (info != null) source = info.source_clip; }
                catch (JsonException) { }
                if (!string.IsNullOrEmpty(source) &&
                    !string.Equals(Path.GetFileNameWithoutExtension(source), Path.GetFileNameWithoutExtension(path), System.StringComparison.OrdinalIgnoreCase))
                {
                    Debug.LogWarning($"[MatchOverlay] {Path.GetFileName(analysisPath)} was made from '{source}', not '{Path.GetFileName(path)}': " +
                                     "ignoring it (shapes will show '-'). Re-run export_for_unity.py on this file.");
                    analysis = null;
                }
            }
            Load(File.ReadAllText(path), analysis);
        }

        void Load(string matchText, string analysisText)
        {
            var match = JsonConvert.DeserializeObject<MatchFile>(matchText);
            Analytics = new MatchAnalytics(match);
            bool useFile = analysisText != null && (preferAnalysisFile || !liveShapes);
            Shapes = liveShapes && !useFile ? new LiveShapes(match) : null;             // [REPLAY]
            if (analysisText != null)                                                   // [FORMATION-ML]
            {
                var shapes = useFile ? ShapeAnalysis.Apply(analysisText, Analytics)
                                     : JsonConvert.DeserializeObject<ShapeAnalysisFile>(analysisText);   // names only
                ShapeTeam a, b;
                if (panel && panel.HasDefaultNames && shapes != null && shapes.teams != null &&
                    shapes.teams.TryGetValue("0", out a) && shapes.teams.TryGetValue("1", out b))
                    panel.SetTeamNames(a.name, b.name);                                 // use names from the file
            }
            if (panel) panel.Bind(Analytics);
            if (links) links.Bind(Analytics);
        }

        void Update()
        {
            if (Analytics == null) return;
            // Same moment in time the 3D players are being drawn at (they render renderDelay in the past).
            Analytics.AdvanceTo(PlayerMotionDriver.Clock - renderDelay);
            if (Shapes != null) Shapes.Apply(Analytics, Analytics.time, shapeWindow, shapeMinMoments);   // [REPLAY]
        }

        // --- hooks for the rest of the project ---
        public void SetScore(int? a, int? b) { if (panel) panel.SetScore(a, b); }
        public void SetFormation(int team, string label, float confidence)
        {
            if (Analytics != null) Analytics.SetFormation(team, label, confidence);
        }
        public void ToggleStats() { if (panel) panel.TogglePanel(); }     // [FORMATION-ML] for a "Stats" button
    }
}
