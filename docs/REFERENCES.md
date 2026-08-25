# Literature grounding

This project's rule-based thresholds are motivated by findings in the sports-
biomechanics literature, not invented numbers. This document lists what each
threshold is based on, how directly the citation supports it, and where the
gap is between the citation's method (typically 3D marker-based motion
capture) and this project's method (monocular 2D/3D pose estimation from a
single RGB video). Being explicit about that gap is the point of this
document -- overstating how well-grounded a heuristic is would be worse than
not citing anything.

## Knee valgus and ACL injury risk (squat, football analyzers)

Hewett et al. (2005), a prospective cohort study of 205 female athletes in
soccer, basketball, and volleyball, found that dynamic knee valgus and high
knee-abduction moment during a landing task predicted future non-contact ACL
injury, with high sensitivity and specificity in the follow-up laboratory
validation.
> Hewett TE, Myer GD, Ford KR, Heidt RS Jr, Colosimo AJ, McLean SG, van den
> Bogert AJ, Paterno MV, Succop P. Biomechanical measures of neuromuscular
> control and valgus loading of the knee predict anterior cruciate ligament
> injury risk in female athletes: a prospective study. *Am J Sports Med*.
> 2005;33(4):492-501.

**Gap:** Hewett et al. measured true 3D knee abduction *moment* (a kinetic
quantity, requiring force plates) via marker-based motion capture. This
project's `knee_valgus_deviation` / `knee_valgus_ratio` metrics are 2D
kinematic *proxies* -- frontal-plane pixel-position deviation, not a moment
-- computed from a single camera. They approximate the same visual pattern
clinicians screen for by eye, not the validated kinetic predictor itself.

## Stiff (shallow-flexion) landings and ACL loading (football analyzer)

Referenced via a systematic review of landing biomechanics: in a study of
171 female basketball and floorball players, stiffer landings (lower knee
flexion, higher peak vertical ground reaction force) were associated with
higher ACL injury risk.
> Leppänen M, Pasanen K, Kujala UM, et al. Stiff landings are associated
> with increased ACL injury risk in young female basketball and floorball
> players. *Am J Sports Med*. 2017;45(2):386-393.

**Confidence note:** I have not personally read the full text of this paper
-- the citation above is reconstructed from a secondary source (an arXiv
review paper) that summarized its findings and attributed them to
"Leppänen et al., 2016." The 2017 *AJSM* citation is the commonly-referenced
published version of this study; **please verify the exact year/volume/page
against Google Scholar or PubMed before citing it to anyone**, since I am
reconstructing bibliographic detail I did not directly verify against the
primary source.

## Trunk lateral flexion and lumbar stress injury in fast bowlers (cricket_bowling analyzer)

> Elliott B. Back injuries and the fast bowler in cricket. *J Sports Sci*.
> 2000;18(12):983-991.

This review reports that shoulder counter-rotations of 12-40 degrees during
the delivery stride predicted increased incidence of lumbar spondylolysis,
disc abnormality, and muscle injury, and that the "mixed" bowling action
(front-on lower body, side-on upper body) shows more lateral flexion and
hyperextension of the lumbar spine at front-foot impact than pure side-on or
front-on actions.

> Bayne H, Elliott B, Campbell A, Alderson J. Lumbar load in adolescent fast
> bowlers: a prospective injury study. *J Sci Med Sport*. 2016;19(2):117-122.

A prospective study of 25 adolescent fast bowlers: the 12 who sustained a
low-back injury showed significantly greater thorax lateral flexion at ball
release (50±6° vs 40±8° in uninjured bowlers) and higher peak lumbar lateral
flexion moments, alongside greater pelvis rotation and reduced front hip
flexion.

**Gap:** Both studies used 3D marker-based motion capture and, in Bayne et
al.'s case, inverse-dynamics moment calculations. This project's
`trunk_lean_from_vertical` at front-foot-contact is a single-camera 2D
proxy for thorax lateral flexion, and `shoulder_hip_separation` is a crude
2D proxy for shoulder counter-rotation (reliable mainly from a front-on or
3/4 camera angle, not pure side-on). A separate methodological study,
Bayne et al.'s field-based follow-up, found that 2D thorax lateral flexion
and pelvis rotation *at ball release* correlated well with their 3D
equivalents -- which is some support for the general approach of using 2D
proxies for these two specific variables, though this project measures at
front-foot-contact rather than ball release and has not been validated
against 3D ground truth itself.

## Monocular pose estimation vs. marker-based motion capture (general validity)

Several independent validation studies comparing MediaPipe-derived joint
angles to marker-based (Vicon/Qualisys) ground truth report mean absolute
errors in the roughly 5-13 degree range and moderate-to-good agreement
(ICC ~0.82-0.85) depending on joint, movement, and camera placement:

> Comparison of computational pose estimation models for joint angles with
> 3D motion capture. *J Bodyw Mov Ther*. 2024. (MediaPipe and HRNet vs.
> marker-based reference for knee/elbow kinematics; coefficient of
> variation under 10% across five tested activities.)

> Accuracy evaluation of 3D pose estimation with MediaPipe Pose for
> physical exercises. (Optimal two-camera configuration: ICC 0.85 static /
> 0.82 dynamic, MAE ~9-13 degrees vs. marker-based ground truth;
> *this project uses a single camera*, which these results suggest is
> less accurate than the two-camera configurations tested.)

**This is the single most important limitation for a research audience**:
this project has not been validated against any ground-truth motion-capture
system. The above studies establish that MediaPipe-derived angles are
*usable* for research and clinical purposes under favorable conditions, not
that this specific pipeline's numbers are accurate. A natural next step
(see README's "Possible extensions") is running this project's outputs
against a marker-based or multi-camera reference on a small sample to
establish actual error bounds, the way the papers above do.

## What this document is and isn't

This is an honest attempt to connect the project's design choices to real
literature, compiled via targeted searches rather than from memory, with
explicit notes on confidence and on where a citation's method diverges from
this project's method. It is not a substitute for reading the primary
sources yourself before citing them further, and the one flagged citation
above (Leppänen et al.) should be independently verified before being
relied upon.
