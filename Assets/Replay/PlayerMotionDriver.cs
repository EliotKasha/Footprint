using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// ADAPTER for the PitchOverlay scripts (scoreboard bar, stats panel, formation links).
///
/// Those scripts were written against an earlier prototype whose players were driven by a
/// "PlayerMotionDriver" with a shared Clock and an id -> player lookup. This project's replay
/// (ReplayController + PlayerBody) works differently, so this small stand-in keeps the same names:
/// ReplayController sets <see cref="Clock"/> to the replay time and refills <see cref="ById"/> with the
/// players on screen every frame. The PitchOverlay scripts then work unchanged, and follow pause,
/// scrubbing and playback speed exactly like the 3D replay.
/// </summary>
public class PlayerMotionDriver : MonoBehaviour
{
    /// <summary>Tracked (on-camera) players by tracking id; refilled every frame by ReplayController.</summary>
    public static readonly Dictionary<int, PlayerMotionDriver> ById = new Dictionary<int, PlayerMotionDriver>();

    /// <summary>Replay time in seconds (match file time), set every frame by ReplayController.</summary>
    public static float Clock;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetStatics() { ById.Clear(); Clock = 0f; }

    public int Id { get; private set; }
    public int Team { get; private set; }
    public bool IsKeeper;
    public bool IsActive { get; private set; }

    /// <summary>Called by ReplayController for each player shown this frame.</summary>
    public void Set(int id, int team, bool keeper, bool active)
    {
        Id = id; Team = team; IsKeeper = keeper; IsActive = active;
    }
}
