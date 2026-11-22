// ShapeAnalysis.cs  [FORMATION-ML]  (new file)
// Reads the analysis.json written by formation-ml's export_for_unity.py and hands each team's MEASURED
// shape (e.g. "defends 4-5-1, medium confidence") to MatchAnalytics, so StatsPanel can show it.
//
// "Measured shape" = the shape the team actually took on the pitch, found by counting players per line
// in moments when the team was on camera. It is NOT a guess of the team-sheet formation. Checked on
// real matches: right about 2 times in 3 when confidence is "high"; "low" is not trustworthy.
using System.Collections.Generic;
using Newtonsoft.Json;

namespace PitchOverlay
{
    public class MeasuredShape
    {
        public string shape;            // e.g. "4-5-1"; null = not enough footage to measure
        public string confidence;       // "low" / "medium" / "high"; null if shape is null
        public bool rough;              // true for attacking shapes (players move around a lot more)
        public float agreement;         // share of moments that showed this shape (0..1)
        public float seconds_seen;
        public List<string> runners_up = new List<string>();

        public bool Known { get { return !string.IsNullOrEmpty(shape); } }

        // e.g. "4-5-1 (medium)" or "3-5-2 (rough, low)" or "not enough seen"
        public string Describe()
        {
            if (!Known) return "not enough seen";
            return shape + " (" + (rough ? "rough, " : "") + confidence + ")";
        }
    }

    public class ShapeTeam
    {
        public string name;
        public string kit_hex;
        public MeasuredShape defending;
        public MeasuredShape attacking;
    }

    public class ShapeAnalysisFile
    {
        public string source_clip;
        public float clip_seconds;
        public Dictionary<string, ShapeTeam> teams = new Dictionary<string, ShapeTeam>();
    }

    public static class ShapeAnalysis
    {
        // Parse analysis.json and copy each team's measured shapes into the analytics.
        // Team "0" / "1" are the same numbers as the "team" field in match_events.json.
        public static ShapeAnalysisFile Apply(string json, MatchAnalytics analytics)
        {
            var file = JsonConvert.DeserializeObject<ShapeAnalysisFile>(json);
            if (file == null || file.teams == null) return file;
            for (int t = 0; t < 2; t++)
            {
                ShapeTeam team;
                if (!file.teams.TryGetValue(t.ToString(), out team)) continue;
                analytics.team[t].defending = team.defending;
                analytics.team[t].attacking = team.attacking;
            }
            return file;
        }
    }
}
