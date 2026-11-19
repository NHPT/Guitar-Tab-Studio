# Candidate sequence string v1 data audit

Status: complete. Only governed, verified real aligned sources were inspected.
Review queues, test splits and the frozen mixture-Demucs gate were not read.

## Rights and label suitability

The source ledger version `2026-11-06.2` records GuitarSet, Guitar-TECHS and
AG-PT-set as verified `CC-BY-4.0` sources that permit training, evaluation and
model release. All three provide real audio with aligned pitch and string
labels that can be converted to physical frets.

Their sequence content is materially different:

| Source manifest | Tracks | Events | Same-pitch cross-string transitions within 2 s | Tracks containing a transition |
|---|---:|---:|---:|---:|
| GuitarSet original | 360 | 62,476 | 1,032 | 262 |
| GuitarSet governed mixture | 600 | 107,522 | 1,674 | 440 |
| Guitar-TECHS train | 12 | 1,142 | 2 | 2 |
| Guitar-TECHS validation | 12 | 1,090 | 0 | 0 |
| AG-PT-set train | 353 | 16,908 | 0 | 0 |
| AG-PT-set validation | 72 | 3,690 | 0 | 0 |

The transition count orders events of the same physical MIDI pitch within
each track and counts adjacent occurrences on different strings separated by
at most two seconds. It does not infer transitions across tracks.

## Coverage relevant to the current errors

GuitarSet remains the only verified source with enough real phrase-level
cross-string context for an explicit sequence experiment. Its original
manifest contains:

- 4,638 string-6 events;
- 15,355 string-3 events;
- 8,891 events at frets 9-12;
- dominant same-pitch transition pairs `2->3` (204), `3->2` (197),
  `3->4` (111), `1->2` (107) and `5->4` (92).

The governed mixture manifest expands the same aligned performances under
source and derived mixture audio to 8,120 string-6 events, 26,114 string-3
events and 14,882 fret-9-12 events. It does not constitute an independent
performance source.

Guitar-TECHS and AG-PT-set remain useful for single-note position and
cross-domain evaluation, but they cannot honestly be described as new
same-pitch cross-string phrase supervision. Adding them to this experiment
would change timbre and activity coverage without supplying the requested
sequence evidence.

## Decision

Candidate sequence string v1 is a GuitarSet-only architecture baseline. It
uses the real phrase-level alignments already present in the governed
GuitarSet training split and explicitly models them as ordered sequences for
the first time. Guitar-TECHS and AG-PT-set are excluded from training and
model selection.

The missing independent, commercially usable, real multi-phrase TAB source
remains a data blocker. A future data-ingestion experiment must be
preregistered separately after such a source is acquired and its license,
hashes, performer split and string/fret/onset labels are verified.

The machine-readable audit is
`apps/api/data/training/reports/candidate-sequence-string-v1-source-audit.json`.
