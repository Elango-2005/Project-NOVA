# NOVA-R01-V03 Final Challenge Verification Report

## Executive verdict: FAIL ? do not close NOVA-R01
The original V02 revoke-before-claim race is fixed in fresh real HTTP execution. However, a new publication-failure challenge produced a false persisted terminal outcome after a successful real note write. A separate component challenge showed that the checkpoint API permits replacing an existing execution owner when the caller supplies the current sequence.

This is a fresh challenge by the same assistant that implemented C02, not an independent reviewer identity. No application or test source was changed or repaired. No packages were installed, no Git cleanup/commit/push occurred, and R02 was not started.

## Scope and environment
Repository C:\NOVA; branch main; HEAD e5785edd5338ea8a5d2b4e2bc2cebed4673777a0. The working tree already contained R01/C01/C02 changes. Exact initial status and diff are retained in git-status.txt and git-diff.txt. environment.json records actual package versions. All live endpoints were localhost Uvicorn processes using new synthetic temporary workspaces and databases. Existing evidence was preserved.

All 110 application/test files covered by the before/after manifest matched. source-integrity.json records zero changed files. Evidence drivers were created only in this new evidence directory; repository application/test source was not edited.

## Source-level findings

### V03-F01: Post-execution publication exception is misclassified as pre-claim rejection ? FAIL
The executor finishes the actual note handler and records authorization status success. The orchestrator then publishes tool.completed. When the injected publisher raises UnapprovedToolExecutionError at that point, the broad catch surrounding approval resolution handles it as a definitive rejection before execution. It persists FAILED and task.failed with execution_started=false.

Fresh HTTP evidence:
- Task: task_assistant_a139c4f13edb4e71a67671ca7de00dd8.
- Proposal: prop_ed115f2d31cb4d5fbc0a89c7c40f223f.
- Approval response: HTTP 409, task_status failed.
- Persisted task: failed, failure_reason Authorization rejected before execution.
- SSE terminal: task.failed, reason authorization_rejected, execution_started false; tool.started was already present.
- Authorization ledger: success.
- Actual Markdown: workspace/notes/v03_synthetic.md contains the requested synthetic note.
- Restarted assistant workspace: Saved a personal note / Success.

The failure is preserved in fresh-challenges.json (post_write_permission_exception) and workspace-observations.json. This fault was deliberately injected into the publisher before appending tool.completed; it did not replace the handler or fabricate its result. It demonstrates incorrect classification for that exception type, not a naturally occurring subscriber outage or an attacker-controlled HTTP exploit. The write was authorized. The defect is the false rejection outcome and disagreement between the task/SSE and actual execution evidence.

### V03-F02: Current sequence bypasses owner identity enforcement ? FAIL at internal API boundary
SqliteCheckpointStore.save_checkpoint checks owner mismatch only when expected_seq is None. A caller with the current sequence can supply another approval_attempt_id and successfully replace a running owner, even with protect_terminal=True.

The component challenge used two actual NovaRuntimeContainer instances with shared fresh SQLite storage. Runtime A persisted owner runtime-A; runtime B passed the current sequence and owner runtime-B. The stored owner became runtime-B. This establishes a missing owner constraint at the internal checkpoint port. No public HTTP route or authorization bypass for stealing ownership was demonstrated. Normal competing HTTP approvals remained single-winner, and completed/failed terminal states resisted replacement.

### Other reviewed boundaries
- Authorization claim retains BEGIN IMMEDIATE across ledger validation, persisted policy checking, status claimed update and claim audit insertion. Revocation uses the same persisted policy database. Tested orderings remained safe.
- The production executor calls on_started after claim returns; the orchestrator's tool.started now comes through that callback. Pre-claim rejection generated zero start events.
- Rejection finalization compares the owned sequence. Stale sequences and terminal winners were preserved in tested paths.
- Task terminal payload and checkpoint persist together; SSE merges event IDs to prevent duplicate inclusion. It remains a finite snapshot, not a long-lived stream.
- The revised mock-model crash test interrupts after a real nonterminal checkpoint with BaseException, rather than rewriting a terminal failure. It retains useful graph-resumption coverage but is not a process-crash test. Separate live process crashes were exercised below.
- A synthetic older checkpoint with approval_attempt_id absent and a valid checksum loaded with the default None, resumed and wrote a real note. This tests that specific upgrade path, not every historical schema.

## Race, crash and failure evidence matrix
| Scenario | Verdict | Actual evidence |
|---|---|---|
| Exact V02 revoke-before-claim ordering | PASS | fresh C02 harness: two Uvicorn processes; 409/FAILED/correlated task.failed; zero tool.started and zero writes |
| Claim commits before later revoke | PASS | fresh C01 live harness: real handler barrier; claimed action completes |
| Competing approvals in two actual processes | PASS | fresh challenge: two servers, responses 200/409, one note, completed state, retry rejected |
| Completed winner | PASS | HTTP completed result survives other-process retry; component checkpoint replacement refused |
| Failed winner | PASS | real denied task survives other-process retry; component checkpoint replacement refused |
| Stale expected sequence | PASS | component write raises Stale checkpoint transition |
| Execution ownership cannot be stolen | FAIL at internal port | current-sequence caller replaced runtime-A with runtime-B |
| Restart terminal SSE recovery | PASS | C02 race restart; failed task publisher fault also recovered one terminal event from SQLite |
| Terminal event publication fails before append | PASS for recovery | injected OSError on task.failed; durable task/event returned after restart |
| Terminal event storage fails | PASS for rollback/error containment | SQLite trigger; HTTP 500, RUNNING remains, no terminal event and no note |
| Post-write publication raises authorization exception | FAIL | real note and ledger success, but persisted false pre-execution rejection |
| Crash after claim, before handler | PASS for non-replay | process exit 82; claimed ledger, RUNNING, no file; recovery and retry do not write |
| Crash after ownership, before claim | PASS for non-replay; visibility limited | process exit 81; pending ledger, RUNNING, no file or pending card; recovery/retry do not write |
| Crash after actual handler write | PASS for non-replay | fresh C01 harness process exit 73; real file survives; no repeat execution |
| Note/calendar success, deny, revoke | PASS in tested paths | C01/C02 live handlers, plus fresh explicit calendar denial with zero SQLite rows |
| Older checkpoint without new owner field | PASS | loaded and completed with real handler |

## HTTP, SSE, filesystem and database observations
The original C02 race now agrees across HTTP 409, persisted FAILED and one correlated task.failed. Its pre-claim barrier is test instrumentation around the real claim. The former V02 tool.started barrier cannot reproduce this pre-claim window because start publication now occurs after claim.

The new publication-failure case disagrees across those same surfaces: HTTP and persisted task say rejection, task.failed says execution never started, while the ledger and Markdown prove success. This is the closing blocker despite passing regressions.

The crash-after-claim workspace explicitly says completion unconfirmed and do not automatically retry. The ownership-only crash has no recent activity entry, while GET task reports RUNNING and hides pending approvals. It is non-replayable, but the UI does not explain that interrupted ownership needs inspection. This is a visibility limitation, not a falsely completed terminal result.

The storage-fault case intentionally cannot persist a terminal event. It returns an error and retains nonterminal state rather than inventing success. Its safety result does not prove availability or automatic reconciliation.

JSON evidence contains sanitized HTTP bodies, full synthetic SSE envelopes, task snapshots, actual Markdown, authorization/attempt rows, process commands, PIDs and exit codes. The C01 harness additionally includes calendar SQLite and detailed tool audit snapshots. Controlled exit codes 81/82/73 denote intentional crashes; other server exit codes of 1 result from Windows terminate() cleanup and are recorded explicitly.

## Commands and regression results
Run from C:\NOVA with existing .venv Python. commands-final.json records command arrays, exit codes and outputs.

| Command | Result |
|---|---|
| .venv/Scripts/python.exe -B -m pytest --collect-only -q | exit 0; 173 collected |
| .venv/Scripts/python.exe -B -m pytest -q --tb=short | exit 0; 173 passed in 17.84 seconds |
| .venv/Scripts/python.exe -B tests/integration/c01_live_api.py | exit 0; 12 checks PASS |
| .venv/Scripts/python.exe -B tests/integration/c02_live_api.py | exit 0; original race PASS |
| .venv/Scripts/python.exe -B docs/V03-20261116-224029/fresh_challenge.py | exit 0; six PASS scenarios and one explicit FAIL retained |
| .venv/Scripts/python.exe -B docs/V03-20261116-224029/component_challenge_v2.py | exit 0; five PASS checks and owner-invariant FAIL retained |
| git diff --check | exit 0 |

An initial component-driver attempt exited 1 because it requested an invalid awaiting_approval-to-completed transition. The original driver remains unchanged; v2 uses the valid intermediate RUNNING transition. This was a verification-driver error, not a product finding. Scripts deliberately preserve explicit FAIL records rather than using process exit 0 as proof of acceptance.

Two supplemental stdin Python executions inspected post-restart workspace responses and ran explicit calendar denial, both exit 0; their HTTP/process/data evidence is in workspace-observations.json and calendar-denial.json.

Mock distinction: the complete suite includes scripted model adapters and in-memory mock executors, including the revised graph crash unit test. The new HTTP drivers use real Uvicorn, actual permission/executor code, real note/calendar handlers and SQLite; only crash/ordering/publication fault points are injected. No real LLM, audio, browser or MCP transport capability is established by this verification.

## Acceptance results
| Criterion | Observed result |
|---|---|
| unauthorized_writes = 0 | PASS: zero observed |
| duplicate_writes = 0 | PASS: zero observed |
| successful_approval_replays = 0 | PASS: zero observed |
| misleading_terminal_outcomes = 0 | FAIL: one demonstrated false rejection after a successful write |
| contradictory_terminal_events = 0 | No completed/failed task-event pair observed; task.failed execution_started=false contradicts the existing tool.started and execution records |
| regression_failures = 0 | PASS: complete suite passed |
| unapproved_source_changes = 0 | PASS: all 110 recorded source/test files unchanged |

## ADR compliance and remaining risks
The tested permission gate remains consistent with ADR-010; no write bypass was demonstrated. The false event/result classification conflicts with truthful observability required by ADR-002/NFR-08. The owner-port weakness limits the concurrency guarantee claimed under ADR-003. ADR-009 streaming remains partial because the SSE endpoint returns a snapshot.

No exhaustive power-loss, disk-corruption or arbitrary plugin-failure proof is claimed. The injected post-write exception establishes a concrete reachable fault branch, but production frequency is unknown. Ownership takeover is an internal-port finding, not proof that an HTTP client can supply arbitrary checkpoint state. Older terminal reservations without payload still cannot reconstruct historical lost events. Interrupted ownership remains indefinitely RUNNING without automatic takeover; this avoids replay but needs clear operational handling.

## Recommendation
Do not close NOVA-R01. Seek owner approval for a narrowly scoped correction that distinguishes claim rejection from post-claim/publication errors and enforces expected owner identity alongside checkpoint sequence. Reverify those boundaries before closure. No repair was attempted in V03.

## Source references and evidence index
- V03-F01 broad approval rejection catch: [core/orchestration/engine.py:430](C:/NOVA/services/backend/src/nova/core/orchestration/engine.py:430).
- V03-F01 publication after executor returns: [core/orchestration/engine.py:218](C:/NOVA/services/backend/src/nova/core/orchestration/engine.py:218).
- Actual handler then successful ledger completion: [core/permissions/sandbox_and_audit.py:1048](C:/NOVA/services/backend/src/nova/core/permissions/sandbox_and_audit.py:1048).
- V03-F02 conditional owner guard: [core/orchestration/checkpoint.py:296](C:/NOVA/services/backend/src/nova/core/orchestration/checkpoint.py:296).
- Atomic authorization claim: [core/permissions/authorization.py:146](C:/NOVA/services/backend/src/nova/core/permissions/authorization.py:146).
- Persisted policy check in claim: [core/permissions/sandbox_and_audit.py:719](C:/NOVA/services/backend/src/nova/core/permissions/sandbox_and_audit.py:719).
- Start callback after claim: [core/permissions/sandbox_and_audit.py:1003](C:/NOVA/services/backend/src/nova/core/permissions/sandbox_and_audit.py:1003).
- SSE deduplication: [api/app.py:328](C:/NOVA/services/backend/src/nova/api/app.py:328).
- Recovery ownership guard: [core/orchestration/engine.py:333](C:/NOVA/services/backend/src/nova/core/orchestration/engine.py:333).
- Uncertain activity rendering: [api/assistant_ui.py:185](C:/NOVA/services/backend/src/nova/api/assistant_ui.py:185).

Fresh harness artifacts: [C01 live evidence](../C01-live-nova-c01-live-kc7bttd8.json), [C02 race evidence](../C02-live-nova-c02-http-c_dg2e_g.json). Detailed fresh findings are in fresh-challenges.json, component-challenges.json, workspace-observations.json and calendar-denial.json beside this report. Exact scenario process launch commands and exit codes are embedded in those records.
