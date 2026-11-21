#!/usr/bin/env bash
# Run the whole pipeline on one video: tracking -> events/ball height -> visual check.
#
#   ./run_pipeline.sh test1.mp4
#   ./run_pipeline.sh test1.mp4 --stride 4 --no-viz
#   ./run_pipeline.sh test1.mp4 --out results/game1 -- --ball-tiles 2 --player-tiles 2
#
# Results go to out/<video name>/ (or --out DIR):
#   match.json         raw tracking (players, ball, calibration)
#   match_events.json  + possession, passes, ball height, stats   <- load this one in Unity
#   viz.mp4            minimap next to the video, for checking
#   pipeline.log       everything printed by the three steps
#
# Options:
#   --stride N      process every Nth frame (default 2; 1 = every frame, slower)
#   --out DIR       output folder (default out/<video name>)
#   --no-viz        skip the visual check video
#   --show          also open a live viz window (q = quit, space = pause)
#   --skip-track    reuse an existing match.json in the output folder (re-run events + viz only)
#   --              everything after this is passed straight to track_match.py
#
# Model files (override with environment variables):
#   PLAYERS_MODEL=football-player-detection.pt
#   PITCH_MODEL=football-pitch-detection.pt
#   BALL_MODEL=football-ball-detection.pt      (set BALL_MODEL= to run without a ball model)
# Core ML exports (much faster on a Mac) work too, e.g.
#   PLAYERS_MODEL=football-player-detection.mlpackage ./run_pipeline.sh test1.mp4

set -euo pipefail

# ---------- settings ----------
PLAYERS_MODEL="${PLAYERS_MODEL-football-player-detection.mlpackage}"
BALL_MODEL="${BALL_MODEL-football-ball-detection.mlpackage}"
PITCH_MODEL="${PITCH_MODEL-football-pitch-detection.mlpackage}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src"
[ -d "$SRC_DIR" ] || SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # script placed inside src/

usage() { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# ---------- arguments ----------
[ $# -ge 1 ] || usage 1
case "$1" in -h|--help) usage 0 ;; esac
VIDEO="$1"; shift
STRIDE=2; OUT=""; VIZ=1; SHOW=0; SKIP_TRACK=0; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --stride)     STRIDE="${2:?--stride needs a number}"; shift 2 ;;
    --out)        OUT="${2:?--out needs a folder}"; shift 2 ;;
    --no-viz)     VIZ=0; shift ;;
    --show)       SHOW=1; shift ;;
    --skip-track) SKIP_TRACK=1; shift ;;
    -h|--help)    usage 0 ;;
    --)           shift; EXTRA=("$@"); break ;;
    *)            echo "unknown option: $1 (use -- to pass options to track_match.py)" >&2; usage 1 ;;
  esac
done

# ---------- checks ----------
die() { echo "error: $*" >&2; exit 1; }
[ -f "$VIDEO" ] || die "video not found: $VIDEO"
for f in track_match.py match_events.py viz_match.py; do
  [ -f "$SRC_DIR/$f" ] || die "missing $SRC_DIR/$f"
done

# Python: an active virtualenv wins, else ./.venv, else whatever python3 is on PATH
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=python
elif [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"
else PY=python3; fi
command -v "$PY" >/dev/null 2>&1 || die "python not found ($PY)"

NAME="$(basename "${VIDEO%.*}")"
OUT="${OUT:-out/$NAME}"
mkdir -p "$OUT"
LOG="$OUT/pipeline.log"
: > "$LOG"

step() {  # bold in a terminal, plain text when piped and in the log
  local line; line="[$(date +%H:%M:%S)] $*"
  if [ -t 1 ]; then printf '\n\033[1m%s\033[0m\n' "$line"; else printf '\n%s\n' "$line"; fi
  printf '\n%s\n' "$line" >> "$LOG"
}
# Progress filter: "frame N/TOTAL  (X processed fps)" lines become one live progress bar with an ETA
# (only when the output is a terminal); deprecation noise is dropped; everything else is printed.
PROGRESS_AWK='
/^frame [0-9]+\/[0-9]+/ {
  if (!tty) next
  split($2, a, "/"); n = a[1] + 0; tot = a[2] + 0
  fps = 0
  if (match($0, /\([0-9.]+ processed fps\)/)) { split(substr($0, RSTART + 1, RLENGTH - 2), b, " "); fps = b[1] + 0 }
  pct = (tot > 0) ? n * 100 / tot : 0; if (pct > 100) pct = 100
  fill = int(pct * 30 / 100); bar = ""
  for (i = 0; i < 30; i++) bar = bar ((i < fill) ? "#" : "-")
  eta = ""
  if (fps > 0) { sec = int((tot - n) / stride / fps); eta = sprintf("  ETA %dm%02ds", int(sec / 60), sec % 60) }
  printf "\r  [%s] %3d%%  %d/%d  %.1f fps%s\033[K", bar, pct, n, tot, fps, eta
  shown = 1; fflush(); next
}
/FutureWarning|self\.t = sv\.ByteTrack/ { next }
{ if (shown) { printf "\r\033[K"; shown = 0 } print; fflush() }
END { if (shown) printf "\r\033[K" }'
TTY=0; [ -t 1 ] && TTY=1

run() {  # run STRIDE CMD...: all output goes to the log; the screen gets a progress bar + useful lines
  local stride="$1"; shift
  local t0; t0=$(date +%s)
  set +e
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "$LOG" | awk -v stride="$stride" -v tty="$TTY" "$PROGRESS_AWK"
  local status=${PIPESTATUS[0]}
  set -e
  [ "$status" -eq 0 ] || die "step failed (exit $status), see $LOG"
  local dt=$(( $(date +%s) - t0 ))
  printf '  (took %dm %02ds)\n' $((dt / 60)) $((dt % 60))
}

START=$(date +%s)

# ---------- 1) tracking ----------
if [ "$SKIP_TRACK" -eq 1 ]; then
  [ -f "$OUT/match.json" ] || die "--skip-track: no $OUT/match.json yet"
  step "1/3 tracking: skipped, reusing $OUT/match.json"
else
  [ -e "$PLAYERS_MODEL" ] || die "player model not found: $PLAYERS_MODEL (set PLAYERS_MODEL=...)"
  [ -e "$PITCH_MODEL" ]   || die "pitch model not found: $PITCH_MODEL (set PITCH_MODEL=...)"
  BALL_ARGS=()
  if [ -n "$BALL_MODEL" ]; then
    [ -e "$BALL_MODEL" ] || die "ball model not found: $BALL_MODEL (set BALL_MODEL= to run without one)"
    BALL_ARGS=(--ball "$BALL_MODEL")
  fi
  step "1/3 tracking $VIDEO (stride $STRIDE) - the slow part"
  run "$STRIDE" "$PY" "$SRC_DIR/track_match.py" "$VIDEO" \
      --players "$PLAYERS_MODEL" --pitch "$PITCH_MODEL" "${BALL_ARGS[@]+"${BALL_ARGS[@]}"}" \
      --stride "$STRIDE" --save-h --out "$OUT/match.json" "${EXTRA[@]+"${EXTRA[@]}"}"
fi

# ---------- 2) events + ball height ----------
step "2/3 possession, passes, ball height"
run 1 "$PY" "$SRC_DIR/match_events.py" "$OUT/match.json" --out "$OUT/match_events.json"

# ---------- 3) visual check ----------
if [ "$VIZ" -eq 1 ]; then
  step "3/3 visual check video"
  VIZ_ARGS=(--video "$VIDEO" --out "$OUT/viz.mp4")
  [ "$SHOW" -eq 1 ] && VIZ_ARGS+=(--show)
  run 1 "$PY" "$SRC_DIR/viz_match.py" "$OUT/match_events.json" "${VIZ_ARGS[@]}"
else
  step "3/3 visual check: skipped (--no-viz)"
fi

ELAPSED=$(( $(date +%s) - START ))
step "done in $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "  Unity file:   $OUT/match_events.json"
[ "$VIZ" -eq 1 ] && echo "  check video:  $OUT/viz.mp4"
echo "  full log:     $LOG"