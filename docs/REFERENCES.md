**Literature grounding**

This project's rule-based thresholds are motivated by findings in the sports-
biomechanics literature. This document lists what each
threshold is based on, how directly the citation supports it.

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


## Stiff (shallow-flexion) landings and ACL loading (football analyzer)

Referenced via a systematic review of landing biomechanics: in a study of
171 female basketball and floorball players, stiffer landings (lower knee
flexion, higher peak vertical ground reaction force) were associated with
higher ACL injury risk.
> Leppänen M, Pasanen K, Kujala UM, et al. Stiff landings are associated
> with increased ACL injury risk in young female basketball and floorball
> players. *Am J Sports Med*. 2017;45(2):386-393.


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


