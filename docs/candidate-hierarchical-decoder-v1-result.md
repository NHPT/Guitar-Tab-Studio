# Candidate hierarchical decoder v1 result

Status: rejected on the source-calibration gates. The frozen
mixture-Demucs gate was not read.

## Experiment

The decoder used candidate position head v4 without retraining. Candidate
activity was derived as the total probability mass of all playable strings.
Candidates were accepted independently by that scalar probability, then the
v4 conditional string distribution assigned strings only among accepted
candidates.

Nine preregistered activity thresholds from `0.10` through `0.90` were
evaluated on the 60-track original-microphone source calibration.

## Result

Threshold `0.50` produced the best TAB F1:

| Gate | Result | Required | Status |
|---|---:|---:|---|
| TAB F1 | `0.601503` | `>= 0.704668` | Fail |
| Same-pitch wrong-string errors | `1,055` | `<= 481` | Fail |
| Onset F1 | `0.790740` | `>= 0.815375` | Fail |
| Repeated TAB recall | `0.593189` | `>= 0.721959` | Fail |

The decoder emitted `9,574` events at the selected threshold. Lower
thresholds increased false activity and wrong-string errors; higher thresholds
reduced recall too sharply. Threshold `0.80` was the first point below the
wrong-string limit, with `416` errors, but TAB F1 had already fallen to
`0.491217` and repeated TAB recall to `0.369027`.

Reports use the
`candidate-hierarchical-decoder-v1-source-activity-*.json` prefix.

## Conclusion

Separating the decoder is necessary but not sufficient. The activity
probability inherited from v2 was trained as part of the old categorical
decision and is not accurate enough to act as an independent candidate-level
activity classifier. The next experiment must train a dedicated activity
residual while freezing v4 conditional string selection.
