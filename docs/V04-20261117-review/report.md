# NOVA-R01-V04 — Final Acceptance Verification

## Executive verdict: PASS

The specified R01 acceptance criteria passed against the current working tree in fresh local execution. The original V03-F01 and V03-F02 challenge conditions no longer produced their reported failures. Recommend closing **R01 for this verified source snapshot**. This verdict does not verify unrelated NOVA capabilities or authorize R02.

This review inspected current source and executed new challenges before running the full regression and C01/C02/C03 harnesses. Prior reports were treated as descriptions to challenge, not proof. No application/test source was modified, dependencies installed, or commits/pushes made. Evidence drivers were added only in this new directory.

## Git identity and test environment

- Repository: `C:\NOVA`; branch: `main`.
- HEAD before and after: `d94a2ff2bc836ef23deab158178576a191b89392`.
- The working tree was already dirty: ten tracked source/test files modified, plus untracked C03 tests and evidence. This is acceptance of that working tree, **not HEAD alone**. Exact starting state: [status-before.txt](status-before.txt), [diff-before.txt](diff-before.txt), [before-hashes.json](before-hashes.json).
- Existing venv: Python 3.10.0; pytest 8.4.2; Pydantic 2.14.1; FastAPI 0.146.0; Uvicorn 0.54.0; SQLite 3.35.5. Windows platform build reported `10.0.26300`. See [environment.json](environment.json).
- Real HTTP used loopback Uvicorn child processes and fresh synthetic temporary workspaces. No external model/service was invoked by the acceptance harnesses.
- Governing documents: AGENTS.md, NOVA_ADR.md.pdf, V03 failure report, C03 completion report. Existing system PDF support extracted the ADR locally; no package installation was needed.
- [integrity.json](integrity.json): 121 files before/after; zero changed or added files across services, tests, apps, packages and scripts, excluding bytecode caches. The tracked Git diff remained byte-for-byte identical. New artifacts are evidence only; pytest may update its cache.

## V03-F01 reproduction evidence

[challenge.py](challenge.py) ran the original fault condition against actual note and calendar handlers: publish `tool.completed` raises `UnapprovedToolExecutionError` after the handler returns. It uses the inspected C03 process/HTTP utilities, with separate V04 assertions. Production handlers and outcomes were not replaced.

Both cases returned HTTP 200, persisted COMPLETED, and authorization ledger `success`. The note case produced exactly one Markdown file; the calendar case produced exactly one calendar SQLite row. Neither recorded a fabricated rejection or task.failed event. The original server was stopped before starting its replacement; the replacement returned the **same task.completed event**, including event identity. See [independent.json](independent.json).

Fresh C03 harness execution independently exercised the same two conditions again. The full suite also exercised three exception types at tool.started, tool.completed and task.completed for both handlers. Source inspection confirms publication exceptions are isolated by `_publish_notification`, while execution progress and the bound ledger outcome govern rejection classification.

Result: **PASS — the historical false denial was not reproduced on current source.** No historical checkout was substituted for current source.

## V03-F02 reproduction evidence

The V04 component challenge created an owned RUNNING checkpoint through the current explicit acquisition API. This setup differs from the old driver because implicit acquisition is now rejected; copying the old driver verbatim would fail before reaching its takeover attempt.

For both SQLite and in-memory stores:

| Challenge | Fresh result |
|---|---|
| Replace A with B using current sequence | CheckpointConflict; rejected |
| Replace A with B using current sequence and correct expected owner A | CheckpointConflict; rejected, including protect_terminal=False |
| Legitimate same-owner update | Accepted |
| Stale sequence update | Rejected |
| Release while a proposal remains pending | Rejected |
| Implicit owner release | Rejected |
| Explicit release after clearing pending proposal | Accepted |
| Overwrite completed state with RUNNING | Rejected |

Evidence: owner-sqlite and owner-memory records in [independent.json](independent.json). Existing C03 regression tests additionally cover failed terminal overwrite and stale-state rebasing. The real C03 separate-process probe rejected takeover during a committed claim; the original action subsequently completed exactly once. Concurrent HTTP approvals produced one 200 and one 409, one note, and a preserved completed winner.

Result: **PASS at the checkpoint port and exercised HTTP concurrency boundaries.** Ownership is an internal concurrency identity, not authentication against arbitrary code with database access.

## Full R01 security acceptance matrix

| Required invariant/scenario | Result and fresh evidence |
|---|---|
| Unauthorized writes = 0 | PASS; denied, revoked, malformed, mismatched and forged attempts produced no writes |
| Duplicate writes = 0 | PASS; C01 concurrent calendar requests and C03 two-process note requests each produced exactly one write |
| Approval replay successes = 0 | PASS; retries after completion, restart and interrupted claims rejected |
| False execution outcomes = 0 | PASS; successful publication-fault writes remain completed; uncertain post-write persistence remains nonterminal |
| Unauthorized checkpoint takeovers = 0 | PASS; V04 SQLite/memory challenges and separate-process C03 probe |
| Contradictory terminal states = 0 | PASS; no contradictory completed/failed terminal pair observed; restart preserves terminal winner |
| Revocation commits before claim | PASS; C02 HTTP 409, FAILED/task.failed, no tool.started, zero files |
| Claim commits before revocation | PASS; C01/C03 barriers permit the already-claimed action to finish once |
| Forged/mismatched approval binding | PASS; real HTTP task/node/proposal mismatches and extra arguments rejected; real executor tests reject token, role, task, session, node, expiry and argument mutations |
| Sanitized rejection evidence | PASS; four HTTP rejections audited; synthetic argument secret absent from audit; component sentinel checks passed |
| Audit-storage failure fails closed | PASS; real SQLite trigger rolls back claim; ledger remains approved; no handler write; unauthorized request with broken audit returns 500 and no write |
| Crash after claim does not replay | PASS; real process exit 82, RUNNING/claimed, zero files after restart/recovery/retry |
| Crash after write does not replay | PASS; exits 73/83, one real note survives; no second write |
| Post-write persistence reports uncertainty | PASS; audit/ledger faults return HTTP 503, recovery_required, RUNNING/claimed, one file and no fabricated terminal event |
| Regression failures = 0 | PASS; 203/203 existing pytest tests |
| Unapproved source changes = 0 | PASS; 121 unchanged hashes, identical tracked diff, unchanged HEAD |

## Real HTTP/SSE/filesystem/SQLite results

Fresh artifacts:

- [C01 — 12 passing checks](../C01-live-nova-c01-live-f_27ni0r.json).
- [C02 — revocation-before-claim race](../C02-live-nova-c02-http-v_vnb54c.json).
- [C03 — 13 passing scenarios](../C03-live-1794929432341773600/fresh-challenges.json).
- [V04 HTTP security and audit fault](http-security-v2.json).
- [Compact C03 surface comparison](live-summary.json).

Records include HTTP bodies, task snapshots, actual Markdown, calendar rows, authorization status, SSE envelopes, process commands/PIDs and exit codes. C03 also exercised ownership-only crash, lost claim return, failed terminal publication and failed terminal-event storage. Unconfirmed cases remained RUNNING without invented terminal events; retries did not execute them.

The supplemental audit fault occurs **before** execution and is not post-write uncertainty: HTTP 200 contains `action_failed`, task and durable SSE say FAILED, ledger remains approved after claim rollback, and no file exists. HTTP status alone is not an execution-success claim. The initial V04 driver incorrectly expected 503/RUNNING here and exited 1. Inspection and restart confirmed the actual consistent failure; a separately retained v2 driver asserts the appropriate invariant and passed. No application/test expectation was changed. [Original observation](http-security.json), [restart inspection](http-security-followup.json), [v2 driver](http_security_v2.py).

## Test commands and exit codes

Run from C:\NOVA. `PY` below means `.venv/Scripts/python.exe -B`.

| Command | Exit/result |
|---|---|
| `PY docs/V04-20261117-review/challenge.py` | 0; note, calendar, SQLite and memory challenges PASS |
| `PY -m pytest -q --tb=short` | 0; 203 passed in 23.97 seconds |
| `PY tests/integration/c01_live_api.py` | 0; 12 checks PASS |
| `PY tests/integration/c02_live_api.py` | 0; race PASS |
| `PY tests/integration/c03_live_api.py` | 0; 13 scenarios PASS |
| `PY docs/V04-20261117-review/http_security.py` | 1; incorrect driver assumption described above; evidence retained |
| `PY docs/V04-20261117-review/http_security_v2.py` | 0; HTTP mismatch/audit/rollback checks PASS |
| Supplemental restart inspection via stdin Python | 0; FAILED, approved ledger, zero files; forged request 500/no write |
| `git diff --check` (twice) | 0; no whitespace errors |

[commands.json](commands.json) retains the ordered principal commands, exit codes and durations. Per-command .txt files retain their output. Ancillary inspection errors: initial PDF extraction wrote adr.txt successfully but console printing exited 1 due to Windows encoding; subsequent UTF-8 reads succeeded. An rg command used a Windows-incompatible wildcard path; the file was subsequently read directly. Neither prevented a material check.

The complete pytest suite mixes deterministic tests, mocked models/executors and real component interactions. Passing mock tests is not operational evidence. Real operational evidence here comes from actual native handlers, SQLite, filesystem and loopback processes; injected hooks control fault timing only. Intentional crash codes 73/81/82/83 differ from normal Windows terminate cleanup code 1. Harnesses stopped their owned servers.

## Remaining risks and scope limitations

- A filesystem write and SQLite outcome are not one transaction. Interrupted or uncertain actions require inspection; automatic replay is deliberately blocked. Owned actions can remain RUNNING indefinitely, and ownership-only crashes have limited UI explanation.
- SSE is a finite event snapshot. Terminal task events survive restart; tool notifications are not guaranteed durable or exactly-once. This does not establish complete ADR-009 streaming compliance.
- Pre-claim audit failure exposes the injected SQLite error text in the friendly failure reply; this review used synthetic text. Broken rejection auditing cannot persist its own rejection record and returns an error instead of executing.
- Tests cover controlled process crashes and SQLite faults, not power loss, disk corruption, every interleaving, or adversarial arbitrary in-process/database access.
- Native note/calendar security paths are verified. Live LLM reasoning, speech, vision, browser/desktop control and operational MCP transport are not verified by this acceptance report.
- Tests ran against the recorded dirty source snapshot. Subsequent code changes require revalidation; a clean commit containing only HEAD is not the tested artifact.

## Recommendation

Close **NOVA-R01** against the recorded working-tree hashes, retaining the limitations above. V03-F01 and V03-F02 are **VERIFIED** for their specified fault conditions. No repair was performed and R02 was not started.
