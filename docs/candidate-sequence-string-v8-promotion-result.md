# Candidate sequence string v8 promotion result

Status: rejected at the first external gate. After the v9 equal-step control
closed the attribution question, the selected v8 checkpoint was evaluated
once on governed GuitarSet. It regressed TAB F1, wrong-string count, and
repeated TAB recall. The fixed order stopped there without reading AG-PT
held-out or Guitar-TECHS P2.

## Frozen inputs

| Artifact | SHA-256 |
|---|---|
| v3 promotion protocol | `e6594381f5bc45a99afa92f29a775f1362e60d6bb4ef5c61c5863f6497f0173d` |
| v8 training result | `bc54b167d6d2eba7b2ab0ebdfeedaeabb8023928fef1a892818179fb41a74202` |
| v9 attribution result | `17612d00d143412de368c35ce3ed4561dc980a3d52a1bcd6fa608799bda4b486` |
| selected v8 checkpoint | `0bf00091daaaf6c8428c982d1bf0e30165460eeae821a96a7552bbad83aad726` |
| research governance report | `4d1f7e5764f2ab5ce6148ab1aca4d2ccdf99a5ba70bb62321ed1b7162e93a040` |
| governed GuitarSet report | `8ed70668e1ce7c29c586ac0904ea1ea49622792e8cd7de40c4699d36d6fcfdc3` |

The report contains all 120 governed GuitarSet validation tracks and records
the selected v8 checkpoint identity. Evaluation used the frozen activity
threshold `0.50`, string margin `0`, full-group assignment, and Basic Pitch
`0.35 / 0.25 / 70 ms` frontend. The checkpoint separately passed the
research-channel source identity, lineage, path, and release-rights checks.

## Governed GuitarSet gate

| Metric | String-v1 baseline | V8 | Change | Requirement | Result |
|---|---:|---:|---:|---|---|
| TAB F1 | `0.664778984` | `0.664674113` | `-0.000104871` | Strict increase | **Fail** |
| Same-pitch wrong strings | `1672` | `1674` | `+2` | Strict decrease | **Fail** |
| Repeated TAB recall | `0.619705185` | `0.619620468` | `-0.000084717` | Non-decrease | **Fail** |
| Onset F1 | `0.809815951` | `0.809815951` | `0` | Unchanged within `1e-12` | Pass |

The unchanged onset result confirms that only the conditional string path
changed. That change nevertheless moved all three string-sensitive metrics in
the wrong direction, so the first external gate failed.

## Decision

The frozen protocol prohibits reading AG-PT held-out or Guitar-TECHS P2 after
a governed GuitarSet failure. Neither downstream report was generated.
Passing all three experiment gates was a prerequisite for a production-gate
assessment, so v8 did not reach the sealed-set, shadow, or canary stages.

The training-only improvement and v9 attribution evidence do not override the
external regression. V8 remains a frozen research checkpoint. No runtime
bundle, default checkpoint, environment configuration, deployed service, tag,
or release changed.
