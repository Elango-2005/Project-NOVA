# NOVA-R01-V02 Independent Challenge Verification

## Executive Verdict: FAIL

V02 reproduced one acceptance failure in F02. A real approval race returned HTTP 409 after `tool.started`, but the task remained `awaiting_approval` and the SSE stream ended at `tool.started`; no terminal failure event or terminal task outcome followed. This does not create a write, but it leaves a failed execution attempt in a misleading, nonterminal state.

The tested F01, F03, F04 and data-side-effect checks passed. The complete suite passed **167 tests**, and the C01 live verification script passed **12 checks**. Those results do not override the independently reproduced F02 failure.

This is a fresh challenge pass by the same assistant that implemented C01. I cannot claim independent reviewer identity. No application or test source was modified during V02. Verification used synthetic data in new temporary directories and local loopback endpoints only.

## Git Identity and Environment

- Branch `main`; HEAD `e5785edd5338ea8a5d2b4e2bc2cebed4673777a0`.
- Python 3.10.0; Pydantic 2.14.1; FastAPI 0.146.0; Uvicorn 0.54.0; pytest 8.4.2.
- At V02 start, the working tree already contained the R01/C01 changes plus unrelated pre-existing edits/untracked files. The exact initial `git status`, `git diff`, HEAD and branch are archived.
- SHA-256 snapshots of **108 application/test Python files** were taken before challenge execution. All 108 matched afterward. `git diff --check` returned 0. No commit or push occurred.
- ADR and governance: `NOVA_ADR.md.pdf`, `AGENTS.md`; implementation scope: C01 completion report.

## F01-F04 Evidence Matrix

| Finding | Verdict | Evidence |
|---|---|---|
| F01 truthful success/denial/failure/uncertainty | PASS in tested paths | Fresh chat note success, quick-note denial, duplicate/failed and uncertain cases were exercised. Denied text described denial and no note was written. The C01 HTTP run also exercised these paths. |
| F02 ordered, terminal, correlated outcomes | **FAIL** | The injected real authorization race after `tool.started` returned 409 but left the task `awaiting_approval`; SSE events were `permission.requested`, `permission.approved`, `tool.started`. No `tool.failed` or terminal task event appeared. |
| F03 persisted cross-runtime revocation and claim ordering | PASS in tested paths | Original two-runtime held-stamp exploit rejected with no file; revoke-first HTTP action blocked; claim-first action finished after the later revocation, per the specified ordering. |
| F04 rejection auditing and fail-closed persistence | PASS in tested paths | Forged, altered, identity-mismatched, expired, replayed and revocation-race attempts had redacted attempt records. An injected SQLite claim-audit failure prevented the handler write. |
| Unauthorized note/calendar writes | 0 observed | Before/after filesystem and SQLite checks across these cases. Claim-first completion after a later revocation is allowed by the documented ordering. |
| Successful approval replays | 0 observed | Repeated approvals rejected; note hashes were unchanged. |
| Concurrent duplicate writes | 0 observed | Ten simultaneous approvals yielded one calendar row/one successful request in the real process run. |
| Regression failures | 0 | 167 passed. |
| V02 source changes | 0 | 108 source/test files hash-identical. |

The detailed scenario results and response bodies are in `challenge-results.json` and `component-results.json` in the evidence archive. The F02 reproducer deliberately subscribes to the real `tool.started` event and revokes from a second runtime before the production atomic claim. It does not replace the production tool handler.

## Actual HTTP, Filesystem and SQLite Evidence

V02 launched local Uvicorn child processes with fresh synthetic data. `challenge-results.json` includes commands, child PIDs, task responses, SSE envelopes and process results. In the normal process:

- Chat created a real Markdown note and finished the task.
- Chat scheduled a synthetic calendar event; SQLite contained the expected row.
- Quick-note denial returned a truthful response and created no file.
- Actual note text and SQLite audit/authorization rows were inspected.
- A second Uvicorn runtime reproduced revocation against a held authorization before claim, with no side effect.
- A second ordering allowed a claimed action to finish after revocation.

The prior C01 harness was also rerun against its own new isolated dataset. It returned `{"result":"PASS","checks":12}`. It reported process termination exit 1 for owned servers and exit 73 for the deliberate crash process; those are expected harness outcomes.

Audit-write fault injection used a SQLite trigger in a fresh test database. The authorization claim transaction rolled back before the note handler; the task returned failed and there was no note. The rejection itself could not be persisted because that same database was deliberately rejecting writes. The C01 implementation surfaces the storage error at the direct executor boundary; there is no durable record when audit storage itself refuses writes. This is an unavoidable evidence-storage limitation, not an authorization allow.

## F02 Failure Details

The actual chain is:

```text
permission.requested
permission.approved
HTTP approval resumes task
TOOL_STARTED is published
executor rechecks persisted policy and atomically claims
revocation has committed first, so claim raises UnapprovedToolExecutionError
API returns HTTP 409
checkpoint remains AWAITING_APPROVAL; SSE has no terminal event
```

Source references: approval resume and `TOOL_STARTED` at [engine.py:332](C:/NOVA/services/backend/src/nova/core/orchestration/engine.py:332) and [engine.py:456](C:/NOVA/services/backend/src/nova/core/orchestration/engine.py:456); atomic rejection at [sandbox_and_audit.py:719](C:/NOVA/services/backend/src/nova/core/permissions/sandbox_and_audit.py:719); permission rejection propagation at [engine.py:174](C:/NOVA/services/backend/src/nova/core/orchestration/engine.py:174). The task is not failed because permission exceptions are re-raised to avoid overwriting another runtime's winner. This protects the checkpoint, but it does not finalize this losing task attempt.

For this race the UI can continue showing a pending approval after it is no longer executable. The authorization-attempt audit entry exists and correctly records rejection. The defect is the missing terminal orchestration result/event for that attempt.

## Source Review

I rechecked the assistant chat and quick action call sites, note/calendar handler registration, executor stamp validation, claim path, policy read/write transactions, event reservations, and audit failure handling. Chat and quick writes route through the graph and unified gate; executor validates the bound stamp and atomically checks policy before invoking the handler. No alternate assistant HTTP mutation bypass was found in the inspected routes.

Relevant files and methods:

- `api/assistant_ui.py`: activity text and approval endpoints.
- `core/cognition/conversation_session.py`: chat routing and shared action submit.
- `core/orchestration/engine.py`: approval resume, `tool.started`, executor invocation and terminal events.
- `core/permissions/sandbox_and_audit.py`: cross-runtime policy snapshot and claim gate.
- `core/permissions/authorization.py`: SQLite transaction, claim and authorization-attempt audit.
- `core/orchestration/checkpoint.py`: terminal event reservations and protected terminal checkpoints.
- `core/tools/personal_assistant_tools.py`: actual local note/calendar handlers.

## Commands and Regression Results

Commands, arguments, exit codes and stdout/stderr paths are in `commands.json`:

```powershell
.venv/Scripts/python.exe -B -m pytest --collect-only -q -p no:cacheprovider
.venv/Scripts/python.exe -B -m pytest -q --tb=short -p no:cacheprovider --basetemp <fresh-v02-temp>/pytest
.venv/Scripts/python.exe -B tests/integration/c01_live_api.py
git diff --check
```

Results: **167 collected and passed in 9.52 seconds; exit 0**. C01 harness: 12 checks, exit 0. `git diff --check`: exit 0. Targeted live challenge results are captured with per-scenario PASS/FAIL; the challenge harness exit status is not used to hide the F02 FAIL.

The suite includes deterministic mock-based model/tool and adapter tests. They are useful for software regression but do not verify model reasoning, external mail/calendar accounts, real MCP servers, browser/desktop automation or speech/vision capabilities. V02's operational note/calendar checks used real local handlers, processes, files and SQLite.

## Remaining Limitations

- F02 fails for a revoke-between-start-and-claim race. No source repair was attempted.
- Event publication is not exactly-once across crashes; C01 documents this. The reproduced failure was an ordinary runtime policy race, not a crash gap.
- If the audit database is unavailable, the rejection cannot be durably recorded; tested behavior fails closed before handler execution.
- The same assistant performed implementation and verification, so independent reviewer independence remains unestablished.
- No source or test fixes, dependency changes, commits, pushes, or R02 work occurred.

## Recommendation

**Do not close NOVA-R01 yet.** Address the F02 race so a claim rejection after `tool.started` produces a truthful terminal task result and correlated failure event without overwriting an already-completed outcome; then repeat V02 independently.
