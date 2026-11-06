# Privacy and Data Governance

## Scope

This policy covers project media, local corrections, community review clips,
training records, model artifacts and operational logs. It is an engineering
contract for the current implementation; a production operator must publish
jurisdiction-specific terms before accepting public contributions.

## Default Data Flow

- Web uploads are processed only for the requested transcription job.
- Desktop local mode keeps media, projects, revisions and caches in application
  storage. It sends no correction, feature, log or audio telemetry.
- Exporting diagnostics remains local. A future “report issue” flow may upload
  only after the user previews the package and explicitly submits it.
- Future anonymous crash or performance reporting must be separately enabled,
  default off, and must never include audio, features, notation, corrections,
  tokens, cookies, local paths or media URLs.
- Desktop cloud mode uploads media only after the user selects cloud inference.
- Hugging Face login uses a public OAuth client with PKCE. Access and refresh
  tokens are encrypted with the operating system credential service before
  persistence; when secure storage is unavailable they remain memory-only.
- Cloud API roots are stored locally only after an HTTPS `/health` check.
- Editing a project never opts it into community review or model training.
- Community import accepts only review packages with a supported license,
  complete source hashes and explicit training provenance.

## Rights Matrix

| Source | Private processing | Community review | Training | Redistribution |
| --- | --- | --- | --- | --- |
| User-owned recording | Allowed | Separate consent required | Separate consent required | Only under an explicit compatible license |
| CC0 / supported CC-BY dataset | Allowed | Allowed with attribution metadata | Allowed under declared terms | According to the source license |
| Public streaming link | Authorized personal processing only | Not allowed by default | Not allowed by default | Not allowed |
| Authorized encrypted container | Ephemeral local analysis after explicit consent | Not allowed by default | Not allowed by default | Not allowed |
| Unauthorized encrypted container | Rejected | Rejected | Rejected | Rejected |
| Unknown or conflicting license | Private quarantine only | Rejected | Rejected | Rejected |

Access to media is not evidence of redistribution or training rights.

## Source Ledger And Model Gate

Repository-local training uses the tracked
`services/worker/stringtrace_ml/source_ledger.json` as the authoritative
machine-readable policy. A source entry includes a stable ID and version,
license URL, verification state, allowed purposes, rights flags and hash
requirements. Unknown or quarantined sources fail closed for training and
model release.

Production manifests must use relative media paths and stable source IDs. New
checkpoints contain manifest SHA-256, ledger identity and aggregate source
counts, not raw manifests or local paths. Production promotion rejects a
checkpoint when this identity is absent or stale. A model agreement may create
an `auto-candidate`, but cannot create or replace approved human gold data.

## Consent Record

A production contribution request must bind the following immutable values:

- consent text version and acceptance timestamp;
- allowed purposes: review, research training, public release;
- source license and rights basis;
- exact audio/content SHA-256;
- retention deadline and withdrawal route;
- project revision and derived-clip lineage.

Consent is purpose-specific. Cloud inference consent cannot be reused as
training consent.

## Retention And Deletion

| Data | Default retention | Withdrawal or expiry action |
| --- | --- | --- |
| Desktop local media and revisions | Until the user deletes them | Delete through local project/cache controls |
| Incomplete Web uploads and job scratch data | 24 hours | Automatic purge |
| Private completed project media | 30 days unless the operator offers another setting | Delete media and derived stems |
| Community review clips | Until consent expiry or withdrawal | Stop assignment immediately; delete distributable copies within 7 days |
| Frozen release metadata | Indefinite tombstone | Block the release; retain hashes and non-personal audit evidence |
| Training artifacts derived from withdrawn data | Until impact review completes | Block new promotion and delete releasable artifacts where technically possible |
| Security and governance audit events | 365 days minimum | Retain only pseudonymous identifiers and content hashes |

Deletion must cover primary objects, derivatives, caches and queued work.
Immutable audit records retain only the event type, pseudonymous actor,
timestamp and hashes required to prove the action.

## Withdrawal Procedure

1. Authenticate the contributor or governance operator.
2. Mark the submission or source withdrawn before other work.
3. Remove affected open tasks and consensus labels.
4. Block every active dataset release containing the source.
5. Cancel queued or running experiments using an affected release.
6. assess promoted models and record remediation in their model cards.
7. Delete controlled media copies and retain a non-content tombstone.

The current control plane implements steps 2 through 5 and append-only audit
evidence. Production object-store deletion and model-impact review require the
deployment operator.

## Abuse And Quality Controls

- Credentials are stored as hashes and compared in constant time.
- Multiple accounts from one risk cluster count as one consensus vote.
- Per-account submission limits are 60 per minute and 500 per day.
- Twelve consecutive measured responses under 500 ms, or sustained calibration
  accuracy below 50%, causes a 15-minute automatic pause.
- Advanced tasks remain locked until task-specific calibration reaches the
  configured threshold.
- A paused user can still read their contribution history and exercise
  withdrawal rights.

## Dataset Separation

Every community release uses manifest schema v2. Its data card records licenses,
task coverage and unique performer, work, recording, session and lineage counts.
Community releases are training-only. Public validation and sealed evaluation
must use separate manifests, credentials and storage. Group identifiers must
not cross splits; the Python manifest loaders fail closed on leakage.

## Incident Handling

Automatic canary rejection or champion rollback emits a signed HTTPS webhook
when `GTS_DEPLOYMENT_ALERT_URL` and `GTS_DEPLOYMENT_ALERT_SECRET` are configured.
The payload contains model and aggregate deployment identifiers only. Alert
delivery failure never blocks the local rollback.
