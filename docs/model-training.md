# Guitar Tab Studio model training

## Scope

The trainable transcription backend predicts two synchronized targets:

1. Fret activity: one of silence or fret 0-24 for each of six strings.
2. Re-articulation onset: one independent attack probability for each string.

The separate onset head is required for repeated attacks on the same string and
fret. A frame-only pitch model cannot distinguish those attacks while the note
remains active.

Technique classification remains a separate stage. A trained string/fret model
must not have its positions rewritten by the legacy fingering heuristics.

`tabcnn-gru-v2` keeps the pooled CQT frequency positions before the recurrent
layers. The retired v1 encoder averaged the complete frequency axis and could
learn attack timing but discarded much of the information required to identify
pitch, string and fret. Checkpoint loading remains backward compatible with v1.

## Data contract

Training manifests use JSON Lines schema version 1. Every record contains:

- `track_id`: unique excerpt identifier.
- `group_id`: schema v1 compatibility identifier. It must conservatively group
  every record connected by performer, original work, recording session or
  derived-audio lineage; it cannot represent only one of those dimensions.
- `audio`: absolute path or path relative to the manifest.
- `split`: `train`, `validation`, or `test`.
- `tuning`: open-string MIDI pitches ordered string 1 through 6.
- `capo`: physical capo fret.
- `events`: onset, offset, physical string, absolute physical fret, technique
  and confidence. The UI subtracts the display capo when rendering TAB.
- `beats`, `bpm`, and `time_signature`.
- `provenance`: dataset and annotation source.

No `group_id` may appear in more than one split.

Community schema v2 stores `performer_id`, `work_id`, `recording_id`,
`session_id` and `lineage_id` separately. Its splitter builds connected
components across those relationships and audio fingerprints before assigning
a split.

## Community-governed data and experiments

The current repository-local review commands remain the authoritative workflow
until the community control plane is implemented. Future community
contributions must preserve the same event schema and hash guarantees while
adding stricter separation:

- A personal project correction is not a training label.
- A consented contribution first enters quarantine for rights, format, safety
  and duplicate checks.
- Raw submissions remain immutable and separate from consensus and adjudicated
  gold labels.
- Training datasets are frozen releases with provenance, license, group split,
  manifest hash and dataset card.
- Sealed evaluation records are never sent to annotation or training services.

Community experiments may select only approved dataset releases, recipes and
bounded parameters in the initial release. Each run must record the code
revision, container digest, dataset and split hashes, parameters, random seed,
environment lock, artifact hashes, metrics and model card. A community run can
register a candidate but cannot update the production model alias.

The role model, consensus rules, rights policy, DVC/MLflow integration and
delivery phases are specified in the
[community annotation and open training roadmap](community-ml-roadmap.md).

## Dataset policy

### GuitarSet

The importer reads the six per-string `note_midi` JAMS annotations without
requiring the `jams` Python package.

- Players `00-03`: train.
- Player `04`: validation.
- Player `05`: legacy held-out benchmark. It is no longer the final sealed test
  because multiple gated evaluations and later error analysis have used it.
- Microphone audio is preferred over hex-pickup mixes for real-world transfer.

Download the two required archives from the official Zenodo record with
verified, resumable HTTP range requests:

```bash
.venv/bin/stringtrace-download-guitarset \
  --output-directory apps/api/data/training/guitarset/downloads \
  --workers 8 \
  --chunk-mib 1
```

The command accepts only the official files and checks their published size
and MD5 before replacing the destination:

- `annotation.zip`: `b39b78e63d3446f2e54ddb7a54df9b10`
- `audio_mono-mic.zip`: `275966d6610ac34999b58426beb119c3`

Interrupted downloads resume from validated range parts. Do not extract an
archive until this command reports its MD5 as verified.

After extraction, convert the JAMS annotations and microphone recordings:

```bash
.venv/bin/stringtrace-import-guitarset \
  --root apps/api/data/training/guitarset \
  --audio-kind mic \
  --output apps/api/data/training/guitarset/manifest.jsonl
```

The full dataset is not downloaded automatically. Local free space must be
checked before importing GuitarSet or SynthTab.

### Licensed technique position transfer

The technique corpus can be converted into auxiliary standard-tuning
string/fret supervision without exposing its validation or test partitions:

```bash
.venv/bin/stringtrace-convert-technique-positions \
  --manifest apps/api/data/training/technique-combined-v2.jsonl \
  --output apps/api/data/training/technique-position-train-v1.jsonl \
  --audit-output apps/api/data/training/reports/technique-position-train-v1-audit.json
```

The converter applies a dataset-level license allowlist, exports only the
original `train` split, preserves `group_id`, writes relative audio paths and
drops positions outside frets 0-24. The current output contains 365 tracks,
10 groups and 18,050 events from Guitar-TECHS and AG-PT-set. Both releases are
CC BY 4.0. IDMT-SMT-GUITAR is CC BY-NC-ND 4.0, so it is excluded from
adaptation training even if an older generated manifest declared a less
restrictive license.

For transfer experiments, training and validation manifests and candidate
caches can be selected independently:

```bash
.venv/bin/stringtrace-train-onset-position \
  --manifest apps/api/data/training/technique-position-train-v1.jsonl \
  --validation-manifest apps/api/data/training/guitarset/manifest.jsonl \
  --validation-candidate-cache apps/api/data/training/guitarset/.basic-pitch-cache \
  --initialize-from apps/api/data/training/checkpoints/guitarset-onset-position-confidence-v5.pt \
  --output apps/api/data/training/checkpoints/technique-transfer.pt
```

Only the GuitarSet validation split may select a transfer checkpoint. The
GuitarSet test split remains locked unless the validation result materially
clears the existing candidate.

### Source-verified project excerpts

Source-score measures may be exported directly only when every note in the
selected excerpt has a `positionSource` starting with `source-score`:

```bash
.venv/bin/stringtrace-prepare-reference \
  --project apps/api/data/state/projects/<project-id>.json \
  --audio apps/api/data/jobs/<job-id>/source.m4a \
  --output apps/api/data/training/reference \
  --measure 1 \
  --capo 1
```

Model-generated project notes are candidates, not labels. They cannot be
exported directly. Rank difficult, non-overlapping excerpts before creating
review packages:

```bash
.venv/bin/stringtrace-reference-queue \
  --project apps/api/data/state/projects/<project-id>.json \
  --coverage apps/api/data/training/reports/reference-coverage.json \
  --output apps/api/data/training/reports/reference-review-queue.json \
  --excerpt-measures 2 \
  --limit 12
```

The queue prioritizes same-string intervals within 120 ms, frets 12 and above,
strings missing from every supplied real-reference condition, and continuous
position uncertainty. The default weights are `3`, `2`, `2`, and `1`
respectively. It excludes excerpts containing source-score notes and greedily
selects non-overlapping windows. The score only schedules review work; it is
not a label, evaluation metric or training weight.

Create a review package for one queued contiguous excerpt:

```bash
.venv/bin/stringtrace-reference-review init \
  --project apps/api/data/state/projects/<project-id>.json \
  --audio apps/api/data/jobs/<job-id>/source.m4a \
  --review apps/api/data/training/reviews/<review-id>.json \
  --measure 10 \
  --measure 11
```

The command writes a JSON candidate list and an adjacent WAV excerpt. Listen to
the complete excerpt, edit the event list to add missed notes and remove false
positives, and verify onset/offset, string and absolute physical fret. Candidate
techniques are reset to `unknown`; they remain unknown unless technique review
is explicitly approved.

Approve only after the entire excerpt has been checked:

```bash
.venv/bin/stringtrace-reference-review approve \
  --review apps/api/data/training/reviews/<review-id>.json \
  --reviewer-alias reviewer-1 \
  --timing-reviewed \
  --string-fret-reviewed \
  --completeness-reviewed

.venv/bin/stringtrace-prepare-reference \
  --project apps/api/data/state/projects/<project-id>.json \
  --audio apps/api/data/jobs/<job-id>/source.m4a \
  --review apps/api/data/training/reviews/<review-id>.json \
  --output apps/api/data/training/reference
```

Use a non-identifying reviewer alias. Approval binds the event content, project
snapshot, source audio and review WAV with SHA-256. Export fails if any bound
input changes. The exported record always uses `split=test` and the project ID
as `group_id`; source URLs and local paths are not copied into its provenance.
Reviewed and source-score excerpts are never used to fit or calibrate the
model.

When production transcribes a synchronized derivative such as the Demucs
guitar stem, keep `--audio` bound to the reviewed source and render the
secondary condition explicitly:

```bash
.venv/bin/stringtrace-prepare-reference \
  --project apps/api/data/state/projects/<project-id>.json \
  --audio apps/api/data/jobs/<job-id>/source.m4a \
  --review apps/api/data/training/reviews/<review-id>.json \
  --render-audio apps/api/data/jobs/<job-id>/media/guitar.m4a \
  --render-offset 0.04644 \
  --audio-condition guitar-stem \
  --output apps/api/data/training/reference-guitar-stem-approved
```

The offset must be measured independently, not tuned against evaluation
scores. Provenance records the derivative hash, condition and compensation;
the derivative is marked as transferred from an approved review rather than
as directly human-reviewed audio.

Report coverage separately for each real audio condition:

```bash
.venv/bin/stringtrace-reference-coverage \
  --manifest source-mix=apps/api/data/training/reference/manifest.jsonl \
  --manifest guitar-stem-approved=apps/api/data/training/reference-guitar-stem-approved/manifest.jsonl \
  --output apps/api/data/training/reports/reference-coverage.json
```

The report includes events by string and technique, missing strings, fret
buckets, frets 12 and above, same-string intervals within 120 ms, capo values,
tempo range, source type and annotation provenance. It rejects any real
reference manifest containing a non-test split.

### Synthetic smoke data

Synthetic plucks only verify that feature extraction, labels, training,
checkpoint loading and decoding work together. Synthetic scores are excluded
from accuracy claims.

```bash
.venv/bin/stringtrace-generate-synthetic \
  --output apps/api/data/training/synthetic-smoke
```

## Training and evaluation

```bash
.venv/bin/stringtrace-train \
  --manifest apps/api/data/training/guitarset/manifest.jsonl \
  --output apps/api/data/training/checkpoints/string-fret.pt \
  --epochs 50

.venv/bin/stringtrace-train \
  --manifest apps/api/data/training/guitarset/manifest.jsonl \
  --output apps/api/data/training/checkpoints/string-fret.pt \
  --resume apps/api/data/training/checkpoints/string-fret.pt \
  --epochs 50 \
  --attack-fret-weight 2 \
  --attack-fret-frames 6

.venv/bin/stringtrace-train \
  --manifest apps/api/data/training/guitarset/manifest.jsonl \
  --output apps/api/data/training/checkpoints/string-fret-hard-sampled.pt \
  --resume apps/api/data/training/checkpoints/string-fret.pt \
  --epochs 52 \
  --hard-sample-fast-bonus 0.5 \
  --hard-sample-high-fret-bonus 0.5
```

Train the onset-window candidate model against cached Basic Pitch detections:

```bash
.venv/bin/stringtrace-train-onset-position \
  --manifest apps/api/data/training/guitarset/manifest.jsonl \
  --initialize-from apps/api/data/training/checkpoints/guitarset-onset-position-candidate-v4.pt \
  --output apps/api/data/training/checkpoints/guitarset-onset-position-confidence-v5.pt \
  --epochs 6 \
  --silence-weight 0.5 \
  --candidate-cache apps/api/data/training/guitarset/.basic-pitch-cache \
  --candidate-confidence \
  --candidate-onset-threshold 0.35 \
  --candidate-frame-threshold 0.25 \
  --candidate-minimum-note-length 70

.venv/bin/stringtrace-evaluate-onset-position \
  --manifest apps/api/data/training/guitarset/manifest.jsonl \
  --checkpoint apps/api/data/training/checkpoints/guitarset-onset-position-confidence-v5.pt \
  --split validation \
  --mode direct \
  --minimum-direct-probability 0.50 \
  --basic-pitch-cache apps/api/data/training/guitarset/.basic-pitch-cache \
  --pitch-onset-threshold 0.35 \
  --pitch-frame-threshold 0.25 \
  --pitch-minimum-note-length 70 \
  --output apps/api/data/training/reports/guitarset-onset-position-confidence-v5-validation.json
```

Evaluate legacy frame models and the real-reference gate separately:

```bash
.venv/bin/stringtrace-evaluate \
  --manifest apps/api/data/training/reference/manifest.jsonl \
  --checkpoint apps/api/data/training/checkpoints/string-fret.pt \
  --split test \
  --output apps/api/data/training/reports/reference.json

.venv/bin/stringtrace-calibrate-decoder \
  --manifest apps/api/data/training/guitarset/manifest.jsonl \
  --checkpoint apps/api/data/training/checkpoints/string-fret.pt \
  --split validation \
  --activity-contexts 0.023,0.035,0.046,0.058,0.070 \
  --activity-delays 0,0.012,0.023,0.035,0.046 \
  --minimum-event-confidences 0.38,0.40,0.42,0.44 \
  --output apps/api/data/training/reports/decoder-calibration.json

.venv/bin/stringtrace-evaluate-worker \
  --manifest apps/api/data/training/reference/manifest.jsonl \
  --analysis apps/api/data/jobs/<job-id>/analysis.json \
  --output apps/api/data/training/reports/basic-pitch-reference.json

.venv/bin/stringtrace-model-gate \
  --report apps/api/data/training/reports/reference.json \
  --baseline apps/api/data/training/reports/basic-pitch-reference.json
```

Training creates a persistent `.feature-cache` next to the manifest by default.
Each source recording is transformed once, then split into cached CQT windows.
The cache key includes the audio file metadata, feature configuration and window
layout, so stale entries are not reused after those inputs change. Use
`--no-feature-cache` only for diagnostics.

Metrics are intentionally separated:

- Onset F1: attack timing only.
- Pitch F1: sounding pitch, regardless of string.
- Tablature F1: onset, string and fret must all match.
- Repeated-tablature recall: repeated attacks on the same string/fret.
- Beat-relative onset: timing recall and error normalized by the local annotated
  beat duration.
- Decoder errors: exact TAB, same-pitch wrong-string, same-string wrong-fret,
  wrong-string-and-pitch, spurious-onset and missed-onset counts.

Pitch and tablature matching compare the absolute physical frets stored by the
model, Worker and project. The annotated capo is display metadata and is not
added again during evaluation.

Decoder calibration extracts features and runs the checkpoint once per track,
then evaluates the complete parameter grid against the validation split. Do
not calibrate on the test split or source-verified project excerpts.

Hard sampling raises the probability of windows containing same-string attacks
within 140 ms or frets 12 and above. It preserves the number of samples per
epoch and stores the sampling configuration in checkpoint metadata. It is an
experimental training option, not a production decoder rule.

Pitch-shift augmentation moves cached CQT bins and absolute fret labels
together by one to three semitones. It runs in memory and does not duplicate
the feature cache:

```bash
.venv/bin/stringtrace-train \
  --manifest apps/api/data/training/guitarset/manifest.jsonl \
  --output apps/api/data/training/checkpoints/string-fret-shifted.pt \
  --resume apps/api/data/training/checkpoints/string-fret.pt \
  --epochs 52 \
  --pitch-shift-probability 0.5 \
  --pitch-shift-min 1 \
  --pitch-shift-max 3
```

## Production gate v2 (authoritative)

A checkpoint must not become the default backend until all conditions hold:

1. A new sealed set contains at least 100 songs that have never participated in
   model design and covers multiple players, guitars, pickups, microphone
   recordings, tempos, capo positions and techniques. GuitarSet player 05 is
   excluded from this final gate because it is now a legacy benchmark.
2. Tablature F1 is at least 0.75 and at least 0.15 above the Basic Pitch baseline.
3. Repeated same-string/same-fret onset recall is at least 0.85.
4. No split leakage or invalid string/fret events are reported.
5. Error review includes audio/TAB playback, not only loss curves.
6. All conditions are conjunctive; passing a candidate-unlock or validation
   threshold does not waive any production requirement.

A failed gate does not stop research training or explicit local evaluation. It
only blocks changing the default Worker backend, so an experimental checkpoint
cannot silently replace the more reliable baseline for new projects.

For a community-produced checkpoint, the metric gate above is necessary but not
sufficient. The platform must also reproduce the run in a clean environment,
verify complete data and model lineage, pass the sealed evaluation and
robustness suites, then complete shadow and canary operation. In the current
single-maintainer phase, the owner may approve after every automated gate
passes and must record `approval_mode=single-maintainer`. Once another
maintainer is available, the submitter cannot independently approve promotion.

Enable an approved checkpoint explicitly:

```bash
export STRINGTRACE_TAB_MODEL=/absolute/path/to/string-fret.pt
export STRINGTRACE_MODEL_DEVICE=cpu
```

The worker reports `transcription_model.kind = "trained"` when the checkpoint
backend is active. Without a checkpoint, it reports the current Basic Pitch
baseline and preserves the review warning.

## Current measured status

The local GuitarSet import contains 360 microphone recordings and 62,476
string-level note events. All 360 JAMS files and all 360 WAV files match the
official mirdata 1.1.0 per-file MD5 index. The performer split contains 240
training, 60 validation and 60 test tracks.

The validation and legacy benchmark performers remain isolated. Decoder
thresholds are selected on player 04 only. Player 05 was first read after the
v5 candidate and thresholds were frozen, but later gated evaluations and error
analysis mean it now serves only as a comparable legacy benchmark. It must not
be described or used as the final unbiased production test. Final promotion
requires the new, never-inspected, at-least-100-song `sealed_evaluation` set.

| Candidate | Validation TAB F1 | Test TAB F1 | Test repeated recall | Result |
| --- | ---: | ---: | ---: | --- |
| E17 distance-balanced | 0.5148 | 0.5049 | 0.5343 | Rejected |
| E18 pitch consistency | 0.5284 | 0.5076 | 0.5459 | Rejected |
| E18 pitch shift | 0.5156 | 0.5024 | 0.5488 | Rejected |
| E19 Basic Pitch hybrid | 0.5790 | 0.5922 | 0.5958 | Superseded |
| Onset-window v2 | 0.6499 | 0.6877 | 0.6589 | Superseded |
| Candidate-distribution v4 | 0.6907 | Not read | Not read | Superseded |
| Confidence-input v5 | **0.7042** | **0.7367** | **0.7187** | Gate failed |
| Group-activity v6 | 0.7033 | Not read | Not read | Rejected |
| Technique transfer v8 | 0.7048 | Not read | Not read | Rejected |
| Mixed-domain auxiliary v11 | 0.7040 | Not read | Not read | Rejected |
| Candidate calibrator | 0.7092 | Not read | Not read | Rejected |

The v5 legacy player-05 result uses Basic Pitch `onset=0.35`, `frame=0.25`,
`minimum_note_length=70 ms`, direct assignment, and a fixed direct probability
of `0.50`. Its complete test metrics are:

- Onset F1: `0.8533`.
- Pitch F1: `0.8171`.
- Tablature F1: `0.7367`.
- Repeated-tablature recall: `0.7187`.
- Exact TAB matches: `6,268`.
- Same-pitch wrong-string errors: `684`.
- Spurious onsets: `1,040`.
- Missed onsets: `1,454`.

This is a material improvement over E19 (`+0.1445` test TAB F1 and `+0.1229`
repeated-note recall), but it fails all three numeric promotion requirements:
TAB F1 is below `0.75`, improvement over E19 is below `0.15`, and repeated
recall is below `0.85`. Basic Pitch therefore remains the default backend.

The low-threshold candidate frontend is not the immediate ceiling. With oracle
string/fret labels, its validation TAB F1 is `0.9751` and repeated-note recall
is `0.9563`. The remaining gap is in onset rejection, missing-note recovery and
same-pitch string selection. Candidate-distribution training, confidence input,
joint output confidence, a group-activity head, constrained supplementation and
repeat-aware thresholds were all evaluated. Only candidate-distribution
training and confidence input improved the held-out result.

Licensed technique-position transfer was also evaluated without re-reading the
legacy player-05 benchmark. Full two-stage transfer reached validation TAB F1 `0.7048` and
repeated recall `0.6964`; mixed-domain auxiliary training reached `0.7040` and
`0.6959`. A train-only candidate calibrator improved validation onset/pitch F1
to `0.8283`/`0.7807`, but TAB F1 stopped at `0.7092`; supplementation raised
repeated recall only to `0.7250`. These branches are rejected and their
checkpoints are not retained. The compact evidence record is
`technique-position-transfer-experiments-v1.json`.

The source-mix reference now contains eight approved human-reviewed excerpts
with 246 evaluation-eligible events across all six strings. Two confirmed
events are at fret 12 or above. One 0.3546-second
inaudible passage is explicitly excluded from evaluation. The earlier
1.997-second, 18-event source-score record remains quarantined because its
independent acoustic alignment audit failed; the guitar-stem condition still
contains only its matching quarantined record. Quarantined events are retained
for diagnostics and excluded from accuracy and promotion claims.

The coverage-driven review queue contains 12 non-overlapping two-measure
excerpts and 483 candidate events. `real-reference-01` through
`real-reference-08` are approved and exported, so the 200-event data-volume
gate now passes. The current set still represents one project and one source
recording; independent-source, performer, guitar and capo diversity remain
open requirements. Two independent-song candidates are retained under the
deferred review archive and are not shown in the active review queue.

Evaluation on all eight approved excerpts confirms that the production gate
must remain closed. On the production-consistent, offset-compensated guitar
stem, the onset-position candidate reaches onset F1 `0.2314`, TAB F1 `0.1488`
and repeated-attack recall `0.0891`. The existing Worker baseline reaches onset
F1 `0.1698`, TAB F1 `0.0589` and repeated-attack recall `0.0891`; the
candidate's TAB improvement is `0.0898`, below the required `0.15`.
Validation-selected direct threshold `0.40` changes real TAB F1 only to
`0.1489`, while low-threshold supplemental candidates reduce it to `0.1471`;
neither experiment is promoted. These same-source results do not satisfy the
accuracy or independent-source diversity gates.

Repeat-aware group sampling was then evaluated using only GuitarSet training
and validation splits. It raised validation repeated-attack recall from
`0.7040` to `0.7047`, but reduced end-to-end TAB F1 from `0.7042` to `0.7035`.
The `guitarset-onset-position-repeat-v7.pt` checkpoint is retained as rejected
research evidence; the legacy player-05 benchmark was not accessed and v5 remains the
candidate.

Frame-model onset fusion was also evaluated without reading the legacy
player-05 benchmark.
The experimental evaluator takes onset groups proposed by
`guitarset-position-aware-v3-e19-pitch-conditioned.pt`, removes every group
already covered by Basic Pitch within 50 ms, and sends only the remaining
groups through the v5 onset-window model for string/fret reassignment. The
high-precision validation configuration retained 339 of 16,096 frame-model
candidate events and accepted 21 final events. Validation TAB F1 changed only
from `0.704161` to `0.704814`, while repeated-attack recall changed from
`0.703999` to `0.705693`. This is not a material gain, so the fusion is
rejected, the legacy player-05 benchmark was not accessed, and the production candidate
remains unchanged. The complete evidence is
`guitarset-onset-position-fusion-v1-validation.json`.

Candidate-corruption training now applies pitch dropout consistently to cached
Basic Pitch candidates. Earlier, `--pitch-drop-probability` affected only
reference-derived candidates and was silently bypassed in the production-like
cached-candidate training path. Positive groups retain one reference pitch when
all candidates are dropped, while negative candidate groups may remain empty.

Two validation-only robustness trials were trained from v5. The stronger v8
trial used 12 ms onset jitter, 10% candidate-pitch dropout and 10% false-pitch
injection. At the balanced direct threshold `0.50`, validation TAB F1 improved
from `0.704161` to `0.707190` and repeated-attack recall improved from
`0.703999` to `0.716706`. A threshold sweep from `0.45` through `0.70` did not
produce a materially larger joint improvement. Repeated-attack recall improved
in all five GuitarSet validation styles, but TAB F1 gains were uneven and
effectively zero for the singer-songwriter subset. A milder v9 trial was worse.
Both trials are rejected, the legacy player-05 benchmark was not accessed, and v5 remains the
production candidate. The retained research checkpoint is
`guitarset-onset-position-robust-v8.pt`; compact evidence is recorded in
`guitarset-onset-position-robustness-v1.json`.

A production-path domain benchmark now covers 60 stratified GuitarSet training
tracks and all 60 validation tracks after `htdemucs_6s` guitar separation.
The derivative spans five styles, accompaniment and solo roles, and all five
non-test performers. Every stem retains the original note labels plus source
and derived-audio hashes. The validation candidate frontend preserved 94.63%
of source pitch/onset candidates and 98.09% of source onsets. With v5, TAB F1
changed only from `0.704161` on clean audio to `0.703077` on the corresponding
stems, so separation of isolated guitar is not the main real-domain failure.

A v10 model jointly fine-tuned on source/stem views was calibrated from direct
probability `0.30` through `0.50`. No threshold beat v5 on either domain; its
best clean and stem TAB F1 values were `0.702149` and `0.700403`. v10 is
rejected without accessing the legacy player-05 benchmark. The reusable builder is
`stringtrace_ml.prepare_guitarset_stems`, and the audit is
`guitarset-demucs-domain-v1.json`. This result motivated a separate
polyphonic-mixture residual benchmark instead of more processing of already
isolated guitar recordings.

The mixture residual builder now creates a deterministic accompaniment with
drums, bass and keyboard, mixes it with stratified GuitarSet recordings at
`-3`, `0` or `+3` dB SNR, runs the production `htdemucs_6s` guitar separator
and retains only the resulting guitar stem. The current derivative contains
40 training and 20 validation stems (261,570,832 audio bytes), with source,
mixture and stem hashes recorded per track. Its combined source/stem manifest
contains 80 training and 40 validation views. This is an auditable separator
residual surrogate, not a substitute for licensed real multitrack recordings.

The unchanged v5 model reaches TAB F1 `0.676500` on the 20-track mixture-stem
validation split, compared with `0.704161` on clean audio and `0.703077` on
isolated-guitar Demucs stems. A v11 model fine-tuned on the combined
source/mixture-stem views reached only `0.675412` at the production threshold
`0.50`. Validation-only calibration selected `0.45`, where mixture TAB F1
rose by just `0.001444` to `0.677944`, while clean and isolated-stem TAB F1
fell by `0.004181` and `0.003859`. Mixture repeated-attack recall was
unchanged. v11 is rejected, its checkpoint is not retained, and the locked
test was not accessed. The reusable builder is
`stringtrace_ml.prepare_mixture_stems`; the compact audit is
`guitarset-mixture-demucs-domain-v1.json`.

The v2 residual domain expands this procedure to every non-test GuitarSet
recording: 240 training stems and 60 validation stems, or 480/120
source-plus-stem views. It uses a new deterministic seed and five SNR tiers
from `-6` through `+6` dB. The 300 derived stems occupy 1,612,373,320 bytes;
every stem and metadata record passed source, mixture, output hash, duration
and realized-SNR checks. No player-05 test recording is present in either
training manifest.

The conservatively fine-tuned v12 checkpoint improves v2 mixture-stem
validation TAB F1 from `0.664238` to `0.678890` and repeated-attack recall from
`0.657574` to `0.671806`. All five SNR strata improve. Clean validation also
changes from `0.704161` to `0.704753`, while isolated-stem validation changes
from `0.703077` to `0.705337`, so the no-regression gate permits one locked
evaluation. On the GuitarSet player-05 test split, TAB F1 changes from
`0.736718` to `0.739437`. On the eight approved real guitar-stem excerpts, TAB
F1 changes from `0.148760` to `0.158470` and repeated-attack recall from
`0.089109` to `0.099010`.

These gains are consistent but still below production thresholds. Real-stem
TAB improvement over the Worker baseline is `0.099544`, below the required
`0.15`, and the real reference still contains only one source recording. v12
is retained as a research checkpoint, v5 remains the stable trained candidate,
Basic Pitch remains the default Worker backend, and no threshold was tuned
after the additional legacy benchmark evaluation. The complete audit is
`guitarset-mixture-demucs-domain-v2.json`.

A licensed cross-source residual domain now adds 48 AG-PT-set and 12
Guitar-TECHS training recordings, covering 10 source groups and 2.04 hours of
audio. All selected sources are CC BY 4.0. Their deterministic mixture stems
use the same five `-6` through `+6` dB SNR tiers and `htdemucs_6s` production
separator. The derivative contains 60 stems (1,292,623,308 bytes) and 120
source/stem training views. `stringtrace-compose-training-manifest` combines
these with the frozen v2 domain while rejecting test records, duplicate track
IDs and group-level split leakage.

Naive joint fine-tuning in v13 reduced clean, isolated-stem and mixture-stem
TAB F1 by `0.000862`, `0.002123` and `0.002980` relative to v12. Diagnostics
showed that sparsely annotated Guitar-TECHS mixture stems contained 5.52
unmatched candidate groups per positive group, compared with 0.55 for complete
GuitarSet mixture annotations. The training CLI therefore supports repeated
`--positive-only-dataset` options, which remove unmatched candidates only from
explicitly named sparse auxiliary datasets and retain all GuitarSet negatives.
This removed 12,027 auxiliary negative groups in v14, but the three TAB F1
deltas remained `-0.000779`, `-0.000171` and `-0.002562`.

A zero-initialized 32-dimensional residual adapter was then added above the
frozen v12 classifier. Its initial output is exactly identical to v12 and only
17,468 adapter parameters are trainable. The v15 adapter still reduced clean,
isolated-stem and mixture-stem TAB F1 by `0.002438`, `0.003740` and `0.004302`.
This rules out shared-output adapter capacity as a sufficient fix for the
auxiliary position-distribution conflict.

The v16 auxiliary head was trained exclusively on the 5,569 matched
cross-source groups while the v12 base remained frozen. Validation-only
adapter-scale calibration found `0.05` to be the least disruptive setting.
It improved repeated-attack recall by `0.002203`, `0.001864` and `0.003219`
across clean, isolated-stem and mixture-stem validation, but TAB F1 still
changed by `-0.000084`, `-0.000628` and `-0.000006`. Larger scales traded
progressively more TAB F1 for repeated recall.

All four cross-source checkpoints are rejected without re-reading the legacy
player-05 benchmark.
The licensed data domain, safe manifest composer, sparse-label sampling and
scaled adapter capability are retained for a future dedicated onset or
repeated-attack objective. v12 remains the strongest research checkpoint. Full
evidence is in
`cross-source-mixture-domain-v3.json`.

A dedicated v17 candidate-activity head now reuses the frozen v12 encoder and
predicts only whether a Basic Pitch candidate group is active and whether it
represents a repeated attack. It cannot change string/fret logits. Training
uses complete GuitarSet positives and negatives plus positive-only AG-PT-set
and Guitar-TECHS groups. The selected head reaches candidate-level activity F1
`0.9046` and repeat F1 `0.6904`.

Validation selected activity probability `0.20`, repeat probability `0.50`
and repeat-only direct threshold `0.45`. Clean, isolated-stem and mixture-stem
TAB F1 changed by `+0.000157`, `+0.000361` and `+0.000085`; repeated-attack
recall changed by `+0.007286`, `+0.007963` and `+0.010335`. This passed the
validation gate and allowed one additional legacy benchmark evaluation.
GuitarSet player-05 TAB
F1 then changed by `-0.000655`, although repeated recall improved by
`+0.008352`. The approved real guitar-stem TAB F1 improved by `+0.001958` and
repeated recall by `+0.009901`.

v17 is retained as a small research-only head bound to the v12 checkpoint, but
is not enabled in production because the locked GuitarSet TAB no-regression
gate and all absolute real-domain thresholds remain unmet. No thresholds were
changed after locked evaluation. Full evidence is in
`candidate-activity-domain-v1.json`.

The v18 activity head adds six deterministic temporal features: previous-group
gap, same-pitch gap, same-pitch repeat-window membership, previous-group pitch
overlap, group size and mean candidate confidence. Candidate-level repeat F1
improves from v17's `0.6904` to `0.8340`. Validation selected activity
probability `0.20`, repeat probability `0.44` and repeat-only direct threshold
`0.45`.

With v18, clean, isolated-stem and mixture-stem TAB F1 changes by `+0.000755`,
`+0.000073` and `+0.000124`; repeated-attack recall changes by `+0.007794`,
`+0.007794` and `+0.010166`. On the locked GuitarSet test, repeated recall
improves by `+0.009011` while TAB F1 changes by only `-0.000010`. On the
approved real guitar-stem test, TAB F1 improves by `+0.006863` and repeated
recall by `+0.009901`.

v18 supersedes v17 as the preferred research activity head but remains
disabled in production. The microscopic player-05 TAB regression still
fails the strict no-regression rule, and absolute real-domain thresholds
remain far below release requirements. No post-lock threshold tuning was
performed. Full evidence is in `candidate-activity-domain-v2.json`.

An evaluation-only cross-source position set now provides a second gate before
any further external test access. It contains 84 validation recordings and
4,780 events from seven groups: 72 AG-PT-set recordings and 12 Guitar-TECHS
recordings, all under CC BY 4.0. The corresponding 84 deterministic
mixture-plus-Demucs stems retain source, mixture, stem and manifest hashes.
The conversion rejects training splits, and both the manifest provenance and
audit mark the records as prohibited for training.

On source audio, v18 changes TAB F1 from `0.353118` to `0.353823` and repeated
recall from `0.333780` to `0.335121`. On the residual stems, however, TAB F1
changes from `0.217172` to `0.216851`, while repeated recall rises only to
`0.150134`. The stem onset F1 is `0.531731`; Guitar-TECHS is the weakest
source, with onset F1 `0.212620` and zero repeated-event recall. This confirms
that missing frontend candidates, rather than only string/fret assignment,
are the dominant residual-domain bottleneck.

Two candidate-recovery routes were evaluated without opening the external
test split. Basic Pitch supplementation at onset/frame thresholds
`0.25`/`0.20` reduced TAB F1 to `0.216238`; allowing repeat-classified
supplemental events through a `0.45` direct threshold raised repeated recall
to `0.156836` but reduced TAB F1 further to `0.210661`. A frozen-backbone
frame-onset calibration mode was then added. Both learning-rate trials failed
to exceed the initialization frame-onset F1 of `0.270484`. Its best
validation-selected fusion configuration improved all three GuitarSet
validation domains, but independent-stem TAB F1 remained `0.216916`
(`-0.000256` versus v12) and repeated recall remained `0.150134`
(`+0.001340`).

One final shared-encoder trial used only residual stems, a `1e-5` learning
rate, and non-overlapping four-second windows. It improved GuitarSet residual
frame-onset F1 from `0.258400` to `0.260291` and passed the clean,
isolated-stem and mixture-stem end-to-end validation checks. The gain did not
transfer across groups: independent-stem TAB F1 was `0.216799`
(`-0.000373` versus v12), and repeated recall again stayed at `0.150134`.
The checkpoint is rejected.

Manual residual-onset annotation is complete and isolated from all locked
evaluation data. The `residual-onset` queue contains eight approved 12-second
Guitar-TECHS P1 excerpts, 96 seconds total, spanning bend, harmonic, palm mute,
pick, pinch harmonic and vibrato recordings. Review used sample-aligned,
RMS-normalized clean audio; training uses only the unaltered residual stems.
All timing, completeness, string/fret and technique checks pass.

Human review reduced 442 recall-oriented prefilled events to 37 verified
events across six groups. The export contains eight training tracks, no test
records and no evaluation data. Eleven of the 37 reference events have no
Basic Pitch candidate at the fixed `0.35`/`0.25` thresholds, while only one of
191 candidate groups is a repeated event. This confirms that the manual set is
useful onset evidence but still sparse for candidate recovery.

`stringtrace-residual-onset-review export` refuses partial queues and writes a
manifest that references only residual-stem `training_audio`, after verifying
all manifest, review-audio, training-audio and approval hashes:

```bash
PYTHONPATH=services/worker .venv/bin/python \
  -m stringtrace_ml.residual_onset_review export \
  --review-directory apps/api/data/training/reviews/residual-onset \
  --source-manifest \
    apps/api/data/training/technique-mixture-demucs-v1/manifest-combined.jsonl \
  --output-manifest \
    apps/api/data/training/residual-onset-manual-v1/manifest.jsonl \
  --output-report \
    apps/api/data/training/reports/residual-onset-manual-v1.json
```

A v22 candidate-activity update initialized from v18 used one epoch, a `1e-5`
learning rate and manual dataset weight `8`. Validation activity F1 changed
from `0.905362` to `0.904977`, and repeat F1 changed from `0.834011` to
`0.827396`, so the trial was rejected before independent evaluation.

A v23 frame-onset update initialized from the v19 frame model trained only the
onset head for three epochs at `3e-5`, with manual window weight `16`.
Validation frame-onset F1 improved from `0.258400` to `0.265712`. With the
fixed fusion decoder, clean, isolated-stem and mixture-stem TAB F1 changed by
`+0.001640`, `+0.001154` and `+0.000891`; repeated recall changed by
`+0.009997`, `+0.010166` and `+0.012199`.

The gain did not pass the independent gate. Source TAB F1 changed by
`+0.000624`, but residual-stem TAB F1 changed by `-0.000198`. Repeated recall
improved by only `+0.001340` in both independent conditions, below the required
`+0.01`. On the manual training excerpts, fusion accepted only one additional
frame event and reduced TAB F1 from v12's `0.116505` to `0.108108`.

v22 and v23 are rejected, their checkpoints and experiment caches are removed,
and their manifests and evaluation reports are retained for reproducibility.
The AG-PT-set test split remains untouched. v12 plus v18 remain research
baselines; v5 remains the stable trained candidate while Basic Pitch remains
the default Worker backend. Another attempt requires broader residual-domain
supervision across performers and recordings, especially missing-candidate and
repeated-attack examples. Full evidence is in
`residual-onset-manual-v1.json` and
`independent-position-validation-v1.json`.

The next annotation batch is available at
`/review?queue=residual-onset-v2`. It contains 24 balanced AG-PT-set excerpts,
six from each of four performers, for 288 seconds of audio. Candidate-aware
selection prioritizes reference events that have no same-pitch Basic Pitch
candidate at the production `0.35`/`0.25`/`70 ms` configuration, followed by
repeated attacks and weak candidates. The selected windows contain 328
reference events, including 285 missing-candidate events and 160 repeated
events. Combined with the completed P1 batch, the manual training effort now
covers five performers.

The v2 review audio uses the same sample-aligned clean/residual pair as v1.
The clean listening copies are normalized to approximately `-20 dBFS`, with a
soft `-1 dBFS` peak limit and up to `60 dB` gain for exceptionally quiet AG-PT
recordings. The residual training copies remain unmodified. To limit reviewer
noise, only supplemental v12 candidates with confidence at least `0.5` are
prefilled, yielding 428 total review events rather than 811 unfiltered events.
The v2 queue does not block unrelated model or product development. It must be
fully approved and exported only before a training run claims to use
`residual-onset-v2`, or before such a run enters an independent promotion gate.
Research on existing licensed datasets, synthetic or weakly supervised data,
training infrastructure, desktop work and community tooling may continue in
parallel.

The production candidate remains trained on the official GuitarSet microphone
archive under CC BY 4.0. Guitar-TECHS and AG-PT-set are imported as licensed
research-only auxiliary data, but their transfer experiments did not improve
the gate candidate. IDMT-SMT-GUITAR is excluded from adaptation training due to
its CC BY-NC-ND 4.0 license. The optional GuitarSet hex-debleeded archive was
not downloaded because local capacity is insufficient for safe download and
extraction. Noise2Fret was not imported because its public repository did not
provide both an explicit license and usable released weights. External paper
metrics are not treated as local benchmark results.
