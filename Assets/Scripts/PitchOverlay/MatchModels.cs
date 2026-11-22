// MatchModels.cs
// Plain data classes that mirror match_events.json. Newtonsoft.Json fills them in.
// Fields that may be missing are nullable, so a frame without a ball / owner / team never crashes.
using System.Collections.Generic;
using Newtonsoft.Json;

namespace PitchOverlay
{
    public class MatchFile
    {
        public double fps;
        public List<FrameData> frames = new List<FrameData>();
        public List<MatchEvent> events = new List<MatchEvent>();
    }

    public class FrameData
    {
        public float t;                    // seconds since clip start
        public float[] ball;               // [x, y, height]  (metres, pitch centre = 0,0)
        public int? owner;                 // player id the CV system thinks has the ball (often null)
        public float[][] view_polygon;     // [REPLAY] what the camera shows: [[x, y], ...] (metres, centre = 0,0); used by LiveShapes
        public List<PlayerSample> players = new List<PlayerSample>();
    }

    public class PlayerSample
    {
        public int id;
        public float x, y;                 // metres, pitch centre = 0,0 (x along the length)
        public int? team;                  // 0 or 1
        public string role;                // "player" or "goalkeeper"
    }

    public class MatchEvent
    {
        public string type;                // "touch" or "pass"
        public float? t;                   // touch time
        public float? t_start, t_end;      // pass start / end
        public int? player;                // touch: who
        public int? team;
        [JsonProperty("from")] public int? passer;
        [JsonProperty("to")] public int? receiver;
        public float? length;              // metres
        public bool? completed, aerial;
    }
}
