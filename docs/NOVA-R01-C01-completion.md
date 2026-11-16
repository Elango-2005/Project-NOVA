# NOVA-R01-C01 Completion Report

## Executive Summary

**C01 is IMPLEMENTED and VERIFIED within the documented local operational scenarios.** F01-F04 corrections passed 167 regression tests and 12 real HTTP/process checks. The previously failing two-runtime revocation scenario is blocked. No unauthorized note/calendar writes, successful approval replays, concurrent duplicate writes, or remaining regression failures were observed in these checks.

Verification used fresh synthetic data, actual local handlers, real Uvicorn processes, SQLite inspection and Markdown contents/hashes. No frameworks, dependencies or external services were added. No commits or pushes were made; NOVA-R02 was not started. This is implementation verification by the implementing assistant, not an independent security review.

Repository: `C:\NOVA`, branch `main`, HEAD `e5785edd5338ea8a5d2b4e2bc2cebed4673777a0`. Existing dirty/untracked work was preserved. C01-specific changes are separated from the earlier R01 work in [C01-source-diff.patch](C01-source-diff.patch).

## F01-F04 Root Causes and Fixes

| Finding | Root cause | Correction | Verification |
|---|---|---|---|
| F01 | Activity labels depended only on tool name. | Descriptions now depend on actual execution status. Successful, denied, failed, blocked and unconfirmed operations use different wording. Unresolved claimed actions are displayed conservatively as Uncertain. | Real denial creates no file; duplicate-title failure preserves the original file and says failed. A crashed write displays completion unconfirmed. |
| F02 | Direct actions returned before completion events. Failure emission was inconsistent. | Shared execution finalization emits tool success/failure events; confirmed direct task completion emits task.completed. Protected checkpoints stop stale continuations. Durable terminal-event reservations prevent normal duplicate/conflicting tool-terminal emission and repeated task-terminal emission. | Success sequence and proposal correlation checked through real SSE. Duplicate-file failure emits tool.failed then task.failed. Replayed approvals do not duplicate terminal events. |
| F03 | Each runtime cached policy; policy checks and claims were separate. | Mutations load the latest policy in a SQLite write transaction. Execution reads authoritative policy and claims authorization in one transaction on the same database. | Two Uvicorn processes sharing isolated storage verify revoke-first and claim-first behavior. Additional tests hold transactions open to prove competing writers wait and respect their ordering. |
| F04 | Early stamp/identity/expiry errors bypassed audit logging. | A companion authorization-attempt trail records rejected attempts with trusted context separate from submitted claims. It also records successful claims atomically. Approval schema errors and unknown-task errors are covered. | Forged tokens, altered arguments, wrong role/task/session/node, expiration, replay and revocation produce rejection records. Audit-trigger failures prevent execution or leave a post-write outcome uncertain. |

Key source references:

- [Activity descriptions](C:/NOVA/services/backend/src/nova/api/assistant_ui.py:141).
- [Shared execution finalization](C:/NOVA/services/backend/src/nova/core/orchestration/engine.py:174), [terminal-event reservations](C:/NOVA/services/backend/src/nova/core/orchestration/engine.py:884), [checkpoint protection](C:/NOVA/services/backend/src/nova/core/orchestration/checkpoint.py:252).
- [Policy mutation transaction](C:/NOVA/services/backend/src/nova/core/permissions/sandbox_and_audit.py:246), [policy snapshot and claim](C:/NOVA/services/backend/src/nova/core/permissions/sandbox_and_audit.py:719).
- [Atomic claim and attempt persistence](C:/NOVA/services/backend/src/nova/core/permissions/authorization.py:145), [central rejection recording](C:/NOVA/services/backend/src/nova/core/permissions/sandbox_and_audit.py:730), [invalid approval payload auditing](C:/NOVA/services/backend/src/nova/api/app.py:432).

## Files Changed

Seven existing application files changed in C01:

| File under `services/backend/src/nova/` | Purpose |
|---|---|
| `core/permissions/authorization.py` | Atomic policy callback during claim; companion attempt table; rejected/claimed attempt persistence; unresolved-action query. |
| `core/permissions/sandbox_and_audit.py` | Authoritative policy reload/mutation, transactional claim checks, centralized rejection capture and audit-before-outcome-finalization ordering. |
| `core/orchestration/checkpoint.py` | Additive terminal-event reservation table and optional protected terminal checkpoint writes. The default low-level checkpoint API still supports explicit recovery used by existing tests. |
| `core/orchestration/engine.py` | Protected checkpoint wrapper, shared tool-terminal handling, direct task completion, terminal-event deduplication, approval rejection attribution and correlated failure events. |
| `api/app.py` | Authorization-attempt query, malformed approval auditing and service wiring. |
| `api/assistant_ui.py` | Truthful activity descriptions, unresolved-claim display and early approval rejection auditing. |
| `main.py` | Pass the existing runtime authorization ledger to the API service. |

New verification files:

- `tests/permissions_security/test_c01_corrections.py`: 16 targeted tests.
- `tests/integration/c01_live_api.py`: real Uvicorn, HTTP/SSE, two-runtime, concurrency, expiry and crash verification with fresh evidence output on each run.

No existing test source was changed by C01. `ports.py`, tool capabilities, note/calendar handlers and model adapters were not changed by C01. No application files were deleted.

New documentation/evidence files include this report, `C01-source-diff.patch`, `C01-changed-files.json`, `C01-migration-evidence.json`, `C01-first-regression.txt`, `C01-regression-164.txt`, three uniquely named `C01-live-*.json` run records, and `docs/nova-c01-final-bp2aigtu/` containing six command logs and `commands.json`. Earlier R01/V01 evidence was preserved. The final evidence bundle also contains the C01 test scripts and source hashes.

## Architecture and ADR Compliance

The existing data/control flow remains:

```text
assistant or task API
  -> graph orchestrator with trusted runtime identity
  -> catalog permission gate
  -> SQLite policy + exact authorization validation + single-use claim
  -> real local handler
  -> audit outcome + authorization finalization
  -> tool terminal event + task checkpoint + task terminal event
  -> truthful API/activity response
```

ADR-001: execution and persistence stay local. ADR-002/009: typed lifecycle events remain on the existing event bus. ADR-003/004: orchestration and permissions remain deterministic; no model decides whether authorization is valid. ADR-005/006: existing native handlers and replaceable ports remain in use. ADR-008: agent scopes are unchanged. ADR-010: persisted policy is authoritative and audit failures cannot silently allow execution.

No new dependency or deployment service is required. Existing public write URLs and request bodies remain unchanged. The additive endpoint `GET /audit/authorization-attempts` exposes the companion attempt trail; `/audit/logs` retains its existing execution-log contract.

## Database and Concurrency Design

The existing policy and authorization tables stay in `nova_audit.db`. A new table, `authorization_attempts`, contains attempt ID, timestamp, outcome, reason code, optional trusted context JSON, and submitted-claims JSON. It does not store approval tokens, tool arguments, note bodies, rationale or arbitrary exception messages. Submitted identifiers are bounded and explicitly labelled as claims; they are not substituted for runtime identity.

The claim transaction is:

```text
BEGIN IMMEDIATE
  verify bound approval
  read current persisted policy on this same connection
  reject revocation or incompatible policy
  validate canonical action/context, expiry and single-use state
  update approved -> claimed
  insert claimed attempt record
COMMIT
invoke handler
```

Revocation/restoration/tightening also begin a SQLite write transaction, read the latest policy, apply the mutation and commit. This prevents both stale-policy claims and lost updates between runtime instances. Instance locks protect local object access; SQLite provides the cross-runtime ordering.

**Ordering rule:** if revocation commits first, the claim is rejected. If the claim commits first, that already-authorized action may finish even if revocation subsequently commits. Revocation does not promise cancellation or rollback of an already-claimed filesystem write. The database transaction does not span handler execution.

A required claim-audit insertion failure rolls back the claim before any handler is invoked. If rejection-audit persistence itself fails, the request remains denied and the storage error surfaces; it is impossible to promise a durable rejection record while its storage is unavailable. If outcome auditing fails after a real write, the authorization remains claimed and the UI reports uncertainty rather than completed success.

`terminal_event_claims` is an additive table in `nova_checkpoints.db`. Unique reservation keys prevent repeated normal terminal publication. Runtime checkpoint writes refuse to replace terminal outcomes, and stale orchestrator continuations stop. Explicit low-level recovery remains available to trusted code; it does not reset consumed authorizations or automatically replay uncertain writes.

**There is no exactly-once event-delivery guarantee across a crash or publication failure.** A crash between reserving an event and publishing it can leave a missing event. Durable checkpoint/authorization state remains the recovery evidence. No durable event-delivery subsystem was introduced.

## Real Execution Evidence

Final live result: **PASS, 12 checks**, parent process exit **0**.

[Final HTTP/process evidence](C01-live-nova-c01-live-5p8_d631.json) contains exact requests/responses, SSE event payloads/correlations, process commands/exits, file hashes/content, authorization snapshots, audit identities and the direct SQLite calendar query.

| Operational scenario | Observed result |
|---|---|
| Denied note | HTTP 200 with action_failed; denied activity text; no file. |
| Approved structured note | Actual Markdown preserves supplied delimiter and synthetic HTML text; note_saved response. |
| Successful lifecycle | permission.requested -> permission.approved -> tool.started -> tool.completed -> task.completed, correlated to the original proposal. |
| Duplicate note title | Real exclusive-create failure; original note preserved; failed activity and tool.failed/task.failed. |
| Six concurrent calendar approvals | Exactly one HTTP 200; other attempts rejected with 409; exactly one calendar row. |
| Runtime B revokes before A executes | A's pending action fails without writing. |
| Runtime A claims before B revokes | Test barrier holds the actual handler after the claim; B commits revocation; releasing the barrier allows the already-claimed action to finish. |
| Restart | Notes, calendar, revocation and replay rejection persist. |
| Controlled crash after actual note write | Process exits 73 before outcome finalization. Restart rejects replay with 409 and shows Uncertain activity. |
| Expired authorization | Real elapsed one-second test TTL; HTTP 409; expired attempt recorded. |
| Malformed approval and unknown task | HTTP 422/409 respectively; sanitized attempt records. Synthetic secret body absent from attempt output. |
| Direct storage inspection | Three expected note files and one calendar event; recorded audit identities match runtime identities. |

Final authorization states were **3 success, 3 denied, 1 error, 1 claimed, 1 expired-but-still-pending**. The claimed record belongs to the crash-after-write scenario; it was not replayed. The expired pending record cannot be consumed despite remaining pending in storage. Claim-first completion after later revocation is permitted by the explicit ordering rule, not an unauthorized write.

Owned Uvicorn processes were terminated after verification; Windows termination exit 1 is expected. The deliberately interrupted process exited 73. All eight child process IDs, commands and exit codes are recorded. No real personal data or external account was used.

## Security Tests

The 16 new targeted tests cover:

- Previously approved stamps rejected after another runtime revokes permission.
- Concurrent policy changes preserving both revocations.
- Both writer orderings with transactions deliberately held open; the competing SQLite writer waits.
- Claim-first completion after revocation using the real note handler.
- Forged token, modified arguments, wrong role/task/session/node and expired authorization, with exactly one companion rejection record per attempt.
- Trusted role distinct from submitted role; no approval token or note content in the attempt trail.
- Real SQLite triggers rejecting claim/rejection audit inserts: no handler write and claim rollback.
- Real outcome-audit insertion failure after a note write: claimed state retained and Uncertain activity shown.
- Truthful denial/failure descriptions and ordered/correlated success/failure events.
- Replayed approvals, repeated resume and stale checkpoint continuation not replacing successful outcomes or duplicating terminal events.
- Unresolved claims surviving restart and remaining non-replayable.

Existing R01 canonical argument binding, narrow tool scope, invalid timestamps, duplicate-note protection, replay, concurrency and sandbox tests also pass.

Instrumentation is confined to tests: transaction barriers, deliberately failing SQLite triggers, a short TTL, and a process exit after calling the actual handler. Operational handlers do not fabricate results. The broader existing suite still includes deterministic model/tool substitutes; its pass count is not evidence of real LLM, speech, vision, browser, mail or MCP-server operation.

## Regression Results and Commands

Final commands ran from `C:\NOVA`; exact argument arrays, stdout/stderr paths and exit codes are in [commands.json](nova-c01-final-bp2aigtu/commands.json).

```powershell
.venv/Scripts/python.exe -B -m pytest -q --tb=short -p no:cacheprovider --basetemp <fresh-temporary-directory>/pytest
.venv/Scripts/python.exe -B tests/integration/c01_live_api.py
git diff --check
```

- Full suite: **167 passed in 11.58 seconds; exit 0**. This includes all 151 prior tests plus 16 C01 tests.
- Final live verification: **12 checks passed; exit 0**.
- Diff whitespace check: **exit 0**.
- Targeted final development run: **33 orchestration/C01 tests passed**.

Earlier development runs exposed checkpoint-recovery and sandbox-exception regressions. They were corrected in application code without changing the existing tests. Intermediate logs are retained rather than represented as final success.

## Existing Data Compatibility

A separate subprocess loaded the saved pre-C01 source and created real synthetic note, policy, authorization, audit and checkpoint data. Current C01 source then opened that same isolated dataset. Every pre-existing row in all six databases compared equal, the note SHA-256 was unchanged, and the two additive tables existed. Both subprocesses exited 0.

[Migration evidence](C01-migration-evidence.json) includes exact commands and before/after rows. No real user database was migrated or overwritten by verification.

## Remaining Risks and Capability Status

| Capability | Classification |
|---|---|
| Outcome-based activity reporting (F01) | VERIFIED in recorded denial, failure, success and uncertainty scenarios. |
| Normal terminal lifecycle ordering/deduplication (F02) | VERIFIED; crash-proof exactly-once delivery is not provided. |
| Authoritative multi-runtime policy/claim ordering (F03) | VERIFIED with real processes and overlapping SQLite transactions. |
| Rejection auditing and required-audit fail-closed behavior (F04) | VERIFIED in HTTP and direct security tests. |
| Existing-data compatibility | VERIFIED on synthetic pre-C01 databases and Markdown. |
| Real LLM reasoning, external services, MCP transport | Not verified by C01; no new capability claim. |

Filesystem writes, indexing and SQLite outcomes are still not one atomic transaction. Uncertain actions require inspection; automatic replay remains forbidden. In-progress claims are conservatively shown as unconfirmed as well. SQLite contention can reject operations rather than execute without authorization. The companion attempt trail preserves submitted identifiers but omits tool content and approval credentials. Authorization history is retained for replay prevention; C01 introduces no cleanup policy. Pydantic 1 was not executed; this run used installed Pydantic 2.14.1. These checks are not an exhaustive proof against every possible crash point or hostile local process.

## Knowledge Transfer

Approval is permission for one exact action, not proof that the action happened. Claiming permission commits the right to attempt that action once. A success outcome requires the real handler and required outcome recording to finish. A claimed record without completion can mean an active operation or an interrupted one; neither justifies automatic replay.

The atomic SQLite transaction determines whether revocation or claim wins. Event notifications describe confirmed transitions, while durable records remain authoritative if notification delivery is interrupted.

## Exactly One Recommended Next Milestone

**Run an independent VERIFY_ONLY re-verification of F01-F04 against this C01 implementation before owner acceptance of R01.** No R02 work was started.
