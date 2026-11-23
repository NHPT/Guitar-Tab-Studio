# Candidate sequence gradient conflict v1 result

Status: complete. The frozen training-only audit passed its input and
zero-mutation checks. It did not create an optimizer, update a parameter,
write a checkpoint, or read governed GuitarSet, AG-PT held-out, or
Guitar-TECHS P2 result reports.

Machine-readable report:
`reports/candidate-sequence-gradient-conflict-v1.json`

Report SHA-256:
`f7ca64af7037c5e4f319f0a52ce609cb88720fd1fdaadc24e83ba3e53ec30c33`

## Audit integrity

| Check | Result |
|---|---|
| Frozen plan SHA-256 | `3987e2f213745b9c47a4dab7b05d05bccff8e141353812b458c8a5c23dcf909e` |
| GuitarSet units | 48 tracks, 12 per player |
| GuitarSet replication panels | 4 panels, 3 tracks per player per panel |
| P3 units | 12 paired performances, 24 modality tracks |
| Precomputed groups / windows | 10,335 / 607 |
| Sequence parameters | 93,190 in 16 tensors |
| Bootstrap | 10,000 cluster replicates, seed `20261123` |
| Parameter mutation | none |

Feature precomputation used MPS and created no new feature-cache entries.
Every sequence gradient was computed on CPU with both frozen models in
evaluation mode.

## Primary result

No module meets the preregistered `conflict` criterion. Every
GuitarSet-versus-P3 95% interval crosses zero, so all source-direction
comparisons are `inconclusive`.

| Module | GSet vs P3 cosine | 95% CI | Decision | P3/GSet mean-gradient norm |
|---|---:|---:|---|---:|
| Token encoder | `0.830` | `[-0.958, 0.976]` | inconclusive | `114.6x` |
| Sequence encoder | `0.836` | `[-0.875, 0.954]` | inconclusive | `98.4x` |
| Residual hidden | `0.388` | `[-0.579, 0.716]` | inconclusive | `37.5x` |
| Residual output | `0.472` | `[-0.603, 0.741]` | inconclusive | `41.5x` |
| All parameters | `0.539` | `[-0.678, 0.787]` | inconclusive | `59.6x` |

The positive pooled point estimates do not establish alignment because the
unit-level uncertainty is large. Conversely, the intervals do not support a
claim of stable source-wide gradient conflict.

The actionable signal is magnitude, not sign. At string-v1, P3 has a much
larger mean gradient in every module. The imbalance is strongest in the token
and sequence encoders, but remains large in both residual-head layers. This
does not support diagnosing the v3 failure as a head-only or trunk-only
problem.

## Panel stability

GuitarSet panel means change sign against the same pooled P3 mean:

| Module | Panel 1 | Panel 2 | Panel 3 | Panel 4 |
|---|---:|---:|---:|---:|
| Token encoder | `0.850` | `-0.981` | `0.976` | `0.899` |
| Sequence encoder | `0.796` | `-0.966` | `0.934` | `0.595` |
| Residual hidden | `0.237` | `-0.862` | `0.614` | `0.242` |
| Residual output | `0.090` | `-0.561` | `0.578` | `-0.160` |
| All parameters | `0.181` | `-0.807` | `0.645` | `0.158` |

Panel 2 is strongly negative in all modules while Panel 3 is positive in all
modules. The wide bootstrap intervals therefore reflect real GuitarSet
gradient heterogeneity rather than only bootstrap noise. A single pooled
cosine would conceal this instability.

## P3 modality alignment

P3 direct-input and mic+amp means are consistently aligned:

| Module | Direct vs mic cosine | 95% CI | Decision | Mic/direct norm |
|---|---:|---:|---|---:|
| Token encoder | `0.997` | `[0.975, 0.999]` | aligned | `0.987x` |
| Sequence encoder | `0.992` | `[0.971, 0.995]` | aligned | `1.075x` |
| Residual hidden | `0.957` | `[0.913, 0.981]` | aligned | `0.991x` |
| Residual output | `0.972` | `[0.927, 0.991]` | aligned | `0.936x` |
| All parameters | `0.985` | `[0.946, 0.992]` | aligned | `0.978x` |

All-parameter within-pair cosines range from `0.808` to `0.984`. Performance
`11` is the modality outlier: its sequence-encoder cosine is `0.067` and
token-encoder cosine is `0.360`, although its complete-model cosine remains
`0.808`. Keeping direct and mic as an equal paired mean is supported; treating
them as 24 independent performances is not.

## Failure interpretation

The audit does not support PCGrad or another conflict-removal method as the
first response. No module has a confidence interval entirely below zero, and
the panel signs are heterogeneous rather than uniformly antagonistic.

The stronger explanation for v3 is source-scale mismatch:

- the complete P3 mean gradient norm is `59.6x` the GuitarSet mean;
- trunk ratios approach `100x`;
- mean candidate-normalized loss is `0.173` on GuitarSet, `3.017` on P3
  direct, and `3.307` on P3 mic;
- P3 direct and mic reinforce each other instead of cancelling.

These values are measured before optimization. They do not exactly reproduce
AdamW, batch clipping, or the replacement sampler, but they show that a 20%
P3 sampling share was not a conservative replay mixture. Under a simple
source-mean approximation, equalizing the complete-model contribution would
put P3 near `1.6%`, and balancing the two trunk modules would put it near
`0.9-1.0%`.

## Next experiment

The supported next experiment is a single scale-control ablation, not a new
architecture:

1. initialize from the same frozen string-v1 checkpoint;
2. retain the original per-candidate objective and all v3 model settings;
3. reduce expected P3 sampler mass from `20%` to `1%`;
4. keep P3 direct/mic paired for diagnostics and report source-specific
   gradient norms during training;
5. reuse the unchanged training-only preservation gates before reading any
   external result.

This isolates the strongest observed failure mechanism. Freezing only the
trunk or only the head is not yet justified because the scale imbalance spans
all modules. Gradient surgery should be reconsidered only if a later,
step-level audit shows a consistently negative interval after source
magnitudes are controlled.

No v4 training was started.

## Limitations

The audit measures gradients only at the string-v1 initialization. It uses
equal track/performance units rather than reconstructing every stochastic v3
batch. Some GuitarSet tracks are derived recording variants, so the
track-cluster interval may still understate recording-family correlation.
These constraints limit causal claims but do not remove the observed
cross-module magnitude imbalance.
