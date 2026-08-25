# Athlete Pose & Form Analyzer

A computer vision tool that analyzes athletic movement from a single uploaded video.
It tracks 2D and 3D body pose, computes sport-specific biomechanical metrics, and
flags common form issues using coaching heuristics — for squats, sprinting, and
golf swings.

## Demo

```
streamlit run app.py
```

Upload a video, pick a movement type, and get:
- An annotated video with the skeleton overlay + a live metric readout
- Joint-angle time-series charts with rep boundaries and flagged issues marked
- An interactive 3D skeleton viewer for any frame
- A rule-based form report (e.g. "possible knee valgus", "excessive forward lean")

## How it works

```
video file
   │
   ▼
PoseEstimator (MediaPipe Pose)  ──►  per-frame 2D pixel landmarks
   │                                  + monocular 3D world landmarks (33 joints)
   ▼
SportAnalyzer (squat / sprint / golf)
   │   - joint-angle time series (hip, knee, spine, etc.)
   │   - phase/rep segmentation (e.g. squat bottoms, foot strikes, swing top/impact)
   │   - rule-based flags against coaching thresholds
   ▼
FormReport  ──►  Streamlit UI (annotated video, charts, 3D viewer, feedback list)
```

**Why this structure:** `PoseEstimator` is the only module that touches MediaPipe.
Everything downstream works with plain `FrameLandmarks`/`VideoLandmarks` objects,
so swapping in a different pose backend (e.g. MoveNet, or a proper multi-view 3D
triangulation setup) only requires rewriting one file. Adding a new sport means
subclassing `SportAnalyzer` and implementing `analyze()` — no changes needed
elsewhere.

### Per-sport metrics

| Sport | Metrics tracked | Flags |
|---|---|---|
| **Squat** | knee angle, hip angle, trunk lean, knee valgus ratio, L/R symmetry | didn't reach depth, excessive forward lean, knee valgus, asymmetry |
| **Sprint** | trunk lean, hip flexion (knee drive), foot-strike overstride ratio, cadence | overstride, limited knee drive, excessive lean |
| **Golf** | spine angle, hip lateral position, swing phases (address/top/impact) | early extension, hip sway |
| **Football (injury screen)** | knee flexion at landing, knee valgus ratio, trunk flexion, L/R timing & angle symmetry | stiff landing, knee valgus, upright trunk, asymmetric/non-simultaneous landing |
| **Cricket — bowling** | front knee angle at front-foot-contact, trunk lateral flexion, shoulder-hip separation | limited front-leg bracing, high lateral flexion (lumbar stress-injury marker), mixed-action separation |
| **Cricket — batting** | head lateral deviation from base, hip sway, trunk lean at completion | head off the ball, excessive weight shift, loss of balance |

Football and cricket bowling film a specific event (a jump-landing/cut, or one delivery) rather than a repeated movement — the analyzer auto-locates that event (e.g. jump apex → landing, or front-foot-contact → release) instead of relying on rep counting.

Rep/phase segmentation uses simple signal-processing on the angle time series
(e.g. local minima of knee angle = bottom of each squat) rather than a trained
model — this keeps it fast, dependency-light, and easy to explain, at the cost
of being less robust to noisy or partial-body video than a learned approach.

## Project structure

```
athlete-pose-tracker/
├── app.py                     # Streamlit UI
├── theme.py                    # visual identity: custom CSS + render helpers
├── pipeline.py                 # orchestrates estimator -> analyzer
├── pose_estimation/
│   └── estimator.py            # MediaPipe Pose wrapper -> FrameLandmarks/VideoLandmarks
├── analyzers/
│   ├── base.py                 # SportAnalyzer ABC, FormReport, FormFlag
│   ├── squat.py
│   ├── sprint.py
│   ├── golf.py
│   ├── football.py             # injury-risk screen (jump-landing / cutting)
│   ├── cricket_bowling.py
│   └── cricket_batting.py
├── utils/
│   ├── angles.py                # joint angle math, smoothing, minima-finding
│   └── visualization.py         # skeleton overlay video, charts, 3D plot
├── tests/                       # pytest suite (see Testing below)
├── .github/workflows/tests.yml  # CI: tests + lint on every push
├── .streamlit/config.toml       # base theme
├── requirements.txt             # pinned runtime dependencies
├── requirements-dev.txt         # pytest, ruff
└── LICENSE
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**First run downloads a pose model file** (a few MB, cached under
`~/.cache/athlete_pose_tracker/`) the first time you click Analyze, so an
internet connection is needed once. After that it works offline.

**On MediaPipe's API:** this project uses MediaPipe's current Tasks API
(`PoseLandmarker`), not the older `mp.solutions.pose` API you'll see in a lot
of older tutorials — that older API has been removed from recent MediaPipe
releases ([tracked here](https://github.com/google-ai-edge/mediapipe/issues/6200)),
so code built on it will throw `AttributeError: module 'mediapipe' has no
attribute 'solutions'` on a current `pip install mediapipe`. If you ever see
that error elsewhere, it means whatever you're running predates that removal.
`requirements.txt` pins exact versions that are confirmed to work together
(verified via a clean-venv install + full test run + app boot check) rather
than open-ended `>=` ranges, specifically to avoid this kind of breakage.

Filming tips baked into the UI: squats work from side or front angle; sprints
need a side-on (sagittal) view; golf swings and cricket batting need a
face-on view; cricket bowling needs a side-on view with the full delivery
stride in frame; football needs a front-on/3-4 view of a jump-landing or
cutting movement.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

34 tests covering the geometry utilities (`utils/angles.py`) with hand-checkable
cases (e.g. a straight line is exactly 180°, a right angle is exactly 90°) and
the analyzers against synthetically generated pose sequences — a squat cycle
built from real trigonometry (not just arbitrary numbers) so the rep-counting
and depth-flagging logic is checked against known-correct angles, plus a
crash-safety sweep of every analyzer against zero-detection video. CI
(`.github/workflows/tests.yml`) runs this suite plus a lint pass on Python
3.10/3.11/3.12 on every push.

## Grounding in the literature

The injury-risk thresholds (knee valgus, stiff landings, trunk lateral
flexion in bowling) are motivated by specific findings in the sports-
biomechanics literature — see **[docs/REFERENCES.md](docs/REFERENCES.md)**
for the citations, what each one actually established, and — just as
importantly — the gap between what those studies measured (typically 3D
marker-based motion capture, sometimes with force plates) and what this
project measures (2D/monocular-3D proxies from a single RGB camera). That
gap is not fully closed here; see Limitations below.

## Limitations & honest caveats

- **Not validated against ground truth.** This pipeline's output has not
  been compared to a marker-based motion-capture system or any labeled
  dataset. Published validation studies of MediaPipe generally report joint-
  angle errors in the ~5-13° range against marker-based references under
  favorable single- or two-camera conditions (see docs/REFERENCES.md) — that
  establishes the *approach* is usable for research, not that this specific
  pipeline's numbers are accurate. A natural next step is running this
  project's outputs against a marker-based or multi-camera reference on a
  small sample to establish actual error bounds.
- **Monocular 3D is an estimate, not a measurement.** MediaPipe's 3D world
  landmarks come from a single RGB camera, so depth (Z) is noisier than X/Y.
  Relative joint angles are reliable; absolute distances (e.g. "your stride
  was exactly 1.8m") are not — this project doesn't claim that precision.
- **2D proxies stand in for 3D/kinetic quantities the literature actually
  validated.** E.g. the knee-valgus metrics are frontal-plane pixel-position
  deviations, not the knee-abduction *moment* (a kinetic quantity requiring
  force plates) that Hewett et al. (2005) validated as an ACL-injury
  predictor. The thresholds approximate the same visually-screenable
  pattern a coach would look for, not the validated kinetic predictor
  itself. Full detail per-analyzer in docs/REFERENCES.md.
- **Thresholds are heuristic defaults, not fit to labeled data.** They're
  motivated by the literature's *direction and typical magnitude* of effect
  (e.g. "greater lateral flexion associated with injury" and roughly what
  "greater" meant in that sample), not fit via regression to a labeled
  outcome using this project's own measurement method. Tuning them against
  real, ideally labeled, footage is the logical next step (see Testing).
- **Single camera angle per movement.** Real biomechanics analysis (especially
  for golf swing plane or true shoulder rotation) benefits from multiple
  synced camera views. This is intentionally a single-camera tool.
- **Occlusion and fast motion hurt detection.** Sprint videos especially can
  have motion blur; the UI surfaces the pose-detection rate so users know when
  to trust the results less.

## Possible extensions

- **Validate against ground truth**: compare this pipeline's joint angles to
  a marker-based or multi-camera reference system, on even a small sample,
  and report actual error bounds the way the studies in docs/REFERENCES.md
  do — the single highest-value next step for research use.
- Compare a rep/swing against the athlete's own historical baseline to track
  progress over time
- Add more movements (deadlift, vertical jump, throwing motion)
- Fit thresholds to labeled outcome data (injury history, coach ratings)
  instead of literature-motivated defaults
- Multi-camera triangulation for genuinely accurate 3D
