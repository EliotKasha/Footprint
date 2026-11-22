// StatsPanel.cs
// [FORMATION-ML] LAYOUT REWORK of Zain's panel, so it doesn't cover the pitch:
//   1. A slim TV-style SCOREBOARD BAR across the top of the screen (over the stands, which carry no
//      information in the Broadcast camera): team names, score, clock, possession, and each team's
//      CURRENT shape ("ATTACKING 4-2-4" if it has the ball, else "DEFENDING 4-5-1").
//   2. A "STATS" button on the bar opens/closes the FULL panel (goal threat, passes, team shape details).
//      Closed by default. Also: call TogglePanel() from any other button (e.g. the bottom control bar),
//      or press Tab.
// The numbers, colours and helpers are Zain's; only the layout changed. Built in code with UI Toolkit,
// needs a UIDocument on the same GameObject (see the README).
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

namespace PitchOverlay
{
    [RequireComponent(typeof(UIDocument))]
    public class StatsPanel : MonoBehaviour
    {
        [Header("Teams")]
        public string teamAName = "Team A";
        public string teamBName = "Team B";
        public string teamAShort = "";              // [FORMATION-ML] e.g. "ARG"; empty = first 3 letters of the name
        public string teamBShort = "";
        public Color teamAColor = new Color(0.23f, 0.53f, 1f);
        public Color teamBColor = new Color(1f, 0.75f, 0.04f);

        [Header("Layout")]                          // [FORMATION-ML]
        public float margin = 12f;
        public float barHeight = 46f;
        public float panelWidth = 300f;
        public bool panelOpenAtStart = false;
        public bool showEventTicker = false;        // off: the scene already has a "Match events" log
        public float refreshHz = 10f;

        static readonly Color PanelBg = new Color(0.06f, 0.07f, 0.10f, 0.82f);
        static readonly Color TextMain = new Color(0.92f, 0.94f, 0.98f);
        static readonly Color TextMuted = new Color(0.55f, 0.60f, 0.70f);
        static readonly Color Track = new Color(1f, 1f, 1f, 0.10f);
        static readonly Color Divider = new Color(1f, 1f, 1f, 0.12f);

        MatchAnalytics analytics;
        bool built, panelOpen;
        float timer;
        int? scoreA, scoreB;

        // top bar
        Label clock, score, possA, possB, barChipA, barChipB;
        VisualElement possFillA, possFillB;
        Button toggleButton;
        // full panel
        VisualElement panel;
        Label shapeA, shapeB, passA, passB, threatNumA, threatNumB, ticker, measuredA, measuredB;
        VisualElement threatFillA, threatFillB;
        readonly List<Label> nameLabelsA = new List<Label>(), nameLabelsB = new List<Label>();

        float dPossA = 50f, dThreatA, dThreatB;

        public void Bind(MatchAnalytics a) { analytics = a; }
        public void SetScore(int? a, int? b) { scoreA = a; scoreB = b; }
        public bool IsPanelOpen { get { return panelOpen; } }

        // [FORMATION-ML] open/close the full stats panel - wire this to any button you like
        public void TogglePanel() { SetPanelOpen(!panelOpen); }
        public void SetPanelOpen(bool open)
        {
            panelOpen = open;
            if (panel != null) panel.style.display = open ? DisplayStyle.Flex : DisplayStyle.None;
            if (toggleButton != null) toggleButton.text = open ? "STATS ▴" : "STATS ▾";
        }

        // [FORMATION-ML] change team names after a match is loaded (e.g. from the analysis file)
        public void SetTeamNames(string a, string b)
        {
            if (!string.IsNullOrEmpty(a)) teamAName = a;
            if (!string.IsNullOrEmpty(b)) teamBName = b;
            foreach (var l in nameLabelsA) l.text = (string)l.userData == "short" ? Short(teamAName, teamAShort) : teamAName;
            foreach (var l in nameLabelsB) l.text = (string)l.userData == "short" ? Short(teamBName, teamBShort) : teamBName;
        }
        public bool HasDefaultNames { get { return teamAName == "Team A" && teamBName == "Team B"; } }

        void Start()
        {
            Build(GetComponent<UIDocument>().rootVisualElement);
            SetPanelOpen(panelOpenAtStart);
        }

        void Update()
        {
            if (TabPressed()) TogglePanel();
            if (!built || analytics == null) return;
            timer += Time.deltaTime;
            if (timer < 1f / Mathf.Max(1f, refreshHz)) return;
            float dt = timer; timer = 0f;
            Refresh(dt);
        }

        static bool TabPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            if (kb != null && kb.tabKey.wasPressedThisFrame) return true;
#endif
#if ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKeyDown(KeyCode.Tab)) return true;
#endif
            return false;
        }

        // ------------------------------------------------------------------ build
        void Build(VisualElement root)
        {
            root.Clear();
            root.pickingMode = PickingMode.Ignore;
            BuildBar(root);
            BuildPanel(root);
            built = true;
        }

        // [FORMATION-ML] the slim scoreboard bar, centred at the top of the screen
        void BuildBar(VisualElement root)
        {
            var row = new VisualElement();                      // full-width strip, only used to centre the bar
            row.pickingMode = PickingMode.Ignore;
            row.style.position = Position.Absolute;
            row.style.left = 0; row.style.right = 0; row.style.top = margin;
            row.style.flexDirection = FlexDirection.Row;
            row.style.justifyContent = Justify.Center;
            root.Add(row);

            var bar = new VisualElement();
            bar.pickingMode = PickingMode.Ignore;
            bar.style.flexDirection = FlexDirection.Row;
            bar.style.alignItems = Align.Center;
            bar.style.height = barHeight;
            bar.style.backgroundColor = PanelBg;
            Pad(bar, 14, 0);
            Round(bar, 10);
            Border(bar, 1, new Color(1f, 1f, 1f, 0.08f));
            row.Add(bar);

            // teams + score
            bar.Add(BarTeam(teamAName, teamAShort, teamAColor, false, nameLabelsA));
            score = Text("– : –", 18, TextMain, true);
            score.style.marginLeft = 10; score.style.marginRight = 10;
            bar.Add(score);
            bar.Add(BarTeam(teamBName, teamBShort, teamBColor, true, nameLabelsB));
            bar.Add(VDivider());

            clock = Text("0:00", 14, TextMuted, true);
            clock.style.minWidth = 40; clock.style.unityTextAlign = TextAnchor.MiddleCenter;
            bar.Add(clock);
            bar.Add(VDivider());

            // possession
            var possLabel = Text("POSS", 10, TextMuted, true); possLabel.style.marginRight = 8;
            bar.Add(possLabel);
            possA = Text("50%", 13, teamAColor, true); possA.style.minWidth = 34;
            bar.Add(possA);
            var possBar = new VisualElement();
            possBar.pickingMode = PickingMode.Ignore;
            possBar.style.flexDirection = FlexDirection.Row;
            possBar.style.width = 110; possBar.style.height = 7;
            possBar.style.marginLeft = 6; possBar.style.marginRight = 6;
            possBar.style.overflow = Overflow.Hidden; Round(possBar, 4);
            possFillA = new VisualElement(); possFillA.style.backgroundColor = teamAColor;
            possFillB = new VisualElement(); possFillB.style.backgroundColor = teamBColor;
            possFillA.style.flexBasis = 0; possFillB.style.flexBasis = 0;
            possBar.Add(possFillA); possBar.Add(possFillB);
            bar.Add(possBar);
            possB = Text("50%", 13, teamBColor, true); possB.style.minWidth = 34;
            possB.style.unityTextAlign = TextAnchor.MiddleRight;
            bar.Add(possB);
            bar.Add(VDivider());

            // current shape of each team (measured shape from formation-ml)
            barChipA = Chip(); bar.Add(barChipA);
            barChipB = Chip(); barChipB.style.marginLeft = 6; bar.Add(barChipB);
            bar.Add(VDivider());

            // open / close the full panel
            toggleButton = new Button(TogglePanel) { text = "STATS ▾" };
            toggleButton.style.fontSize = 11;
            toggleButton.style.unityFontStyleAndWeight = FontStyle.Bold;
            toggleButton.style.color = TextMain;
            toggleButton.style.backgroundColor = new Color(1f, 1f, 1f, 0.08f);
            Border(toggleButton, 1, new Color(1f, 1f, 1f, 0.18f));
            Round(toggleButton, 6);
            Pad(toggleButton, 10, 4);
            toggleButton.style.marginLeft = 0; toggleButton.style.marginRight = 0;
            bar.Add(toggleButton);
        }

        // the full panel (Zain's sections), on the right, just below the bar. Hidden until opened.
        void BuildPanel(VisualElement root)
        {
            panel = new VisualElement();
            panel.pickingMode = PickingMode.Ignore;
            panel.style.position = Position.Absolute;
            panel.style.right = margin; panel.style.top = margin + barHeight + 8;
            panel.style.width = panelWidth;
            panel.style.backgroundColor = PanelBg;
            Pad(panel, 16, 14);
            Round(panel, 14);
            Border(panel, 1, new Color(1f, 1f, 1f, 0.08f));
            root.Add(panel);

            panel.Add(SectionTitle("GOAL THREAT  (placeholder estimate)"));
            panel.Add(GaugeRow(teamAName, teamAColor, out threatFillA, out threatNumA, nameLabelsA));
            panel.Add(GaugeRow(teamBName, teamBColor, out threatFillB, out threatNumB, nameLabelsB));
            Spacer(panel, 14);

            panel.Add(SectionTitle("PASSES"));
            panel.Add(StatRow(teamAName, teamAColor, out passA, nameLabelsA));
            panel.Add(StatRow(teamBName, teamBColor, out passB, nameLabelsB));
            Spacer(panel, 14);

            panel.Add(SectionTitle("TEAM SHAPE"));
            panel.Add(ShapeRow(teamAName, teamAColor, out shapeA, out measuredA, nameLabelsA));
            panel.Add(ShapeRow(teamBName, teamBColor, out shapeB, out measuredB, nameLabelsB));

            ticker = Text("", 11, TextMuted, false);
            ticker.style.opacity = 0f;
            ticker.style.marginTop = 10;
            ticker.style.display = showEventTicker ? DisplayStyle.Flex : DisplayStyle.None;
            panel.Add(ticker);
        }

        // ------------------------------------------------------------------ refresh
        void Refresh(float dt)
        {
            var a = analytics.team[0]; var b = analytics.team[1];
            float k = 1f - Mathf.Exp(-dt / 0.35f);                    // gentle easing, no jumpy bars

            int m = (int)(analytics.time / 60f), s = (int)(analytics.time % 60f);
            clock.text = m + ":" + s.ToString("00");

            score.text = (scoreA.HasValue ? scoreA.Value.ToString() : "–") + " : " +
                         (scoreB.HasValue ? scoreB.Value.ToString() : "–");

            dPossA = Mathf.Lerp(dPossA, a.possessionPct, k);
            possA.text = Mathf.RoundToInt(dPossA) + "%";
            possB.text = Mathf.RoundToInt(100f - dPossA) + "%";
            possFillA.style.flexGrow = Mathf.Max(0.001f, dPossA);
            possFillB.style.flexGrow = Mathf.Max(0.001f, 100f - dPossA);

            ShowBarChip(barChipA, a, 0, teamAColor, Short(teamAName, teamAShort));
            ShowBarChip(barChipB, b, 1, teamBColor, Short(teamBName, teamBShort));

            if (!panelOpen) return;                                    // nothing else is visible

            dThreatA = Mathf.Lerp(dThreatA, a.threat, k);
            dThreatB = Mathf.Lerp(dThreatB, b.threat, k);
            threatFillA.style.width = Length.Percent(dThreatA);
            threatFillB.style.width = Length.Percent(dThreatB);
            threatNumA.text = Mathf.RoundToInt(dThreatA).ToString();
            threatNumB.text = Mathf.RoundToInt(dThreatB).ToString();

            passA.text = PassText(a);
            passB.text = PassText(b);

            shapeA.text = ShapeText(a);
            shapeB.text = ShapeText(b);
            ShowMeasuredLine(measuredA, a);
            ShowMeasuredLine(measuredB, b);

            if (showEventTicker)
            {
                float age = analytics.time - analytics.lastEventTime;
                if (analytics.lastEventText.Length > 0 && age >= 0f && age < 5f)
                {
                    ticker.text = analytics.lastEventText;
                    ticker.style.opacity = Mathf.Clamp01((5f - age) / 1.2f);
                    ticker.style.color = analytics.lastEventTeam == 0 ? teamAColor :
                                         analytics.lastEventTeam == 1 ? teamBColor : TextMuted;
                }
                else ticker.style.opacity = 0f;
            }
        }

        // [FORMATION-ML] bar chip: the shape the team is in RIGHT NOW (attacking if it has the ball, else
        // defending). Stronger colour = higher confidence. "–" if not measured. If no analysis file was
        // loaded, falls back to Zain's formation label (only shown at >= 40% confidence).
        void ShowBarChip(Label chip, TeamLive t, int tm, Color c, string shortName)
        {
            if (t.defending == null && t.attacking == null)
            {
                bool show = !string.IsNullOrEmpty(t.formation) && t.formationConf >= 0.40f;
                chip.text = shortName + "  " + (show ? t.formation + " " + Mathf.RoundToInt(t.formationConf * 100f) + "%" : "–");
                chip.style.backgroundColor = new Color(c.r, c.g, c.b, show ? 0.35f : 0.10f);
                return;
            }
            bool hasBall = analytics.holder == tm;
            var now = hasBall ? t.attacking : t.defending;
            if (now != null && now.Known)
            {
                chip.text = shortName + "  " + (hasBall ? "ATTACKING " : "DEFENDING ") + now.shape;
                chip.style.backgroundColor = new Color(c.r, c.g, c.b, ConfidenceAlpha(now.confidence));
            }
            else
            {
                chip.text = shortName + "  " + (hasBall ? "ATTACKING" : "DEFENDING") + " –";
                chip.style.backgroundColor = new Color(c.r, c.g, c.b, 0.10f);
            }
        }

        static void ShowMeasuredLine(Label line, TeamLive t)
        {
            if (t.defending == null && t.attacking == null) { line.style.display = DisplayStyle.None; return; }
            line.style.display = DisplayStyle.Flex;
            line.text = "Defends " + Describe(t.defending) + "\nAttacks " + Describe(t.attacking);
        }

        static string Describe(MeasuredShape m) { return m != null ? m.Describe() : "not enough seen"; }

        static float ConfidenceAlpha(string confidence)
        {
            return confidence == "high" ? 0.55f : confidence == "medium" ? 0.35f : 0.15f;
        }

        static string Short(string name, string shortName)
        {
            if (!string.IsNullOrEmpty(shortName)) return shortName;
            if (string.IsNullOrEmpty(name)) return "?";
            string n = name.Replace(" ", "");
            return (n.Length <= 3 ? n : n.Substring(0, 3)).ToUpperInvariant();
        }

        static string PassText(TeamLive t)
        {
            if (t.passes == 0) return "–";
            return t.passesCompleted + " / " + t.passes + "   " + Mathf.RoundToInt(t.PassAccuracy) + "%";
        }

        static string ShapeText(TeamLive t)
        {
            if (t.width <= 0f) return "–";
            return "W " + Mathf.RoundToInt(t.width) + " m   D " + Mathf.RoundToInt(t.depth) +
                   " m   Line " + Mathf.RoundToInt(t.lineHeight) + " m";
        }

        // ------------------------------------------------------------------ small UI helpers
        static Label Text(string text, int size, Color color, bool bold)
        {
            var l = new Label(text);
            l.style.fontSize = size; l.style.color = color;
            l.style.unityFontStyleAndWeight = bold ? FontStyle.Bold : FontStyle.Normal;
            l.style.marginLeft = 0; l.style.marginRight = 0; l.style.marginTop = 0; l.style.marginBottom = 0;
            l.style.paddingLeft = 0; l.style.paddingRight = 0;
            l.pickingMode = PickingMode.Ignore;
            return l;
        }

        static Label NameLabel(string name, int size, Color color, bool bold, List<Label> registry, bool isShort)
        {
            var l = Text(name, size, color, bold);
            l.userData = isShort ? "short" : "full";
            registry.Add(l);
            return l;
        }

        static VisualElement Row()
        {
            var r = new VisualElement();
            r.style.flexDirection = FlexDirection.Row;
            r.style.justifyContent = Justify.SpaceBetween;
            r.style.alignItems = Align.Center;
            r.pickingMode = PickingMode.Ignore;
            return r;
        }

        static VisualElement VDivider()
        {
            var d = new VisualElement();
            d.pickingMode = PickingMode.Ignore;
            d.style.width = 1; d.style.height = 22;
            d.style.backgroundColor = Divider;
            d.style.marginLeft = 12; d.style.marginRight = 12;
            return d;
        }

        static Label Chip()
        {
            var chip = Text("–", 11, TextMain, true);
            Round(chip, 8); Pad(chip, 8, 3);
            chip.style.minWidth = 150;
            chip.style.unityTextAlign = TextAnchor.MiddleCenter;
            return chip;
        }

        static Label SectionTitle(string t)
        {
            var l = Text(t, 10, TextMuted, true);
            l.style.marginBottom = 6;
            return l;
        }

        static VisualElement Dot(Color c)
        {
            var d = new VisualElement();
            d.style.width = 8; d.style.height = 8; d.style.backgroundColor = c;
            Round(d, 4); d.style.marginRight = 6; d.style.flexShrink = 0;
            d.pickingMode = PickingMode.Ignore;
            return d;
        }

        VisualElement BarTeam(string name, string shortName, Color c, bool alignRight, List<Label> registry)
        {
            var v = new VisualElement();
            v.pickingMode = PickingMode.Ignore;
            v.style.flexDirection = alignRight ? FlexDirection.RowReverse : FlexDirection.Row;
            v.style.alignItems = Align.Center;
            var d = Dot(c);
            if (alignRight) { d.style.marginRight = 0; d.style.marginLeft = 6; }
            v.Add(d);
            v.Add(NameLabel(Short(name, shortName), 14, TextMain, true, registry, true));
            return v;
        }

        static VisualElement GaugeRow(string name, Color c, out VisualElement fill, out Label num, List<Label> registry)
        {
            var row = new VisualElement();
            row.pickingMode = PickingMode.Ignore;
            row.style.flexDirection = FlexDirection.Row; row.style.alignItems = Align.Center;
            row.style.marginBottom = 6;
            var n = NameLabel(name, 12, TextMain, false, registry, false); n.style.width = 80; row.Add(n);
            var track = new VisualElement();
            track.pickingMode = PickingMode.Ignore;
            track.style.flexGrow = 1; track.style.height = 6; track.style.backgroundColor = Track;
            Round(track, 3); track.style.overflow = Overflow.Hidden;
            fill = new VisualElement();
            fill.style.height = Length.Percent(100); fill.style.width = Length.Percent(0);
            fill.style.backgroundColor = c;
            track.Add(fill); row.Add(track);
            num = Text("0", 12, TextMain, true); num.style.width = 30; num.style.unityTextAlign = TextAnchor.MiddleRight;
            row.Add(num);
            return row;
        }

        static VisualElement StatRow(string name, Color c, out Label value, List<Label> registry)
        {
            var row = Row(); row.style.marginBottom = 4;
            var left = new VisualElement(); left.style.flexDirection = FlexDirection.Row; left.style.alignItems = Align.Center;
            left.pickingMode = PickingMode.Ignore;
            left.Add(Dot(c)); left.Add(NameLabel(name, 12, TextMain, false, registry, false));
            row.Add(left);
            value = Text("–", 12, TextMain, true);
            row.Add(value);
            return row;
        }

        static VisualElement ShapeRow(string name, Color c, out Label value, out Label measured, List<Label> registry)
        {
            var box = new VisualElement(); box.style.marginBottom = 8;
            box.pickingMode = PickingMode.Ignore;
            var left = new VisualElement(); left.style.flexDirection = FlexDirection.Row; left.style.alignItems = Align.Center;
            left.pickingMode = PickingMode.Ignore;
            left.Add(Dot(c)); left.Add(NameLabel(name, 12, TextMain, false, registry, false));
            box.Add(left);
            value = Text("–", 11, TextMuted, false);
            value.style.marginTop = 2; value.style.marginLeft = 14;
            box.Add(value);
            measured = Text("", 11, TextMain, false);                  // "Defends 4-5-1 (medium)\nAttacks ..."
            measured.style.marginTop = 2; measured.style.marginLeft = 14;
            measured.style.whiteSpace = WhiteSpace.Normal;
            measured.style.display = DisplayStyle.None;
            box.Add(measured);
            return box;
        }

        static void Spacer(VisualElement parent, float h) { var s = new VisualElement(); s.style.height = h; parent.Add(s); }
        static void Pad(VisualElement e, float h, float v)
        {
            e.style.paddingLeft = h; e.style.paddingRight = h; e.style.paddingTop = v; e.style.paddingBottom = v;
        }
        static void Round(VisualElement e, float r)
        {
            e.style.borderTopLeftRadius = r; e.style.borderTopRightRadius = r;
            e.style.borderBottomLeftRadius = r; e.style.borderBottomRightRadius = r;
        }
        static void Border(VisualElement e, float w, Color c)
        {
            e.style.borderLeftWidth = w; e.style.borderRightWidth = w; e.style.borderTopWidth = w; e.style.borderBottomWidth = w;
            e.style.borderLeftColor = c; e.style.borderRightColor = c; e.style.borderTopColor = c; e.style.borderBottomColor = c;
        }
    }
}
