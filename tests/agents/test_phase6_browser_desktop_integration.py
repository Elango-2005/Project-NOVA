"""NOVA Task 6.2: Phase 6 End-to-End Browser + Desktop + Orchestrator Integration Suite.

Real-Life Analogy:
This test file is the **Final Exam for Phase 6 (Layer 5 — Specialized Browser & Desktop Agents)**.
It connects `PlaywrightBrowserAutomationService` (Phase 13) and `CrossPlatformDesktopAutomationService`
(Phase 14) directly into our **Office Manager (`NovaGraphOrchestrator`)**, **Tool Catalog (`ToolRegistry`)**,
**Permission Gate (`CatalogBackedPermissionGate`)**, **Sandbox (`SandboxedMcpToolExecutor`)**, and
**Black-Box Flight Recorder (`SqliteToolAuditLog`)**:

1. `test_end_to_end_browser_dom_and_desktop_accessibility_workflow_in_orchestrator`:
   - **Step 1 (`Browser` Agent)**: Navigates to a vendor portal in an isolated profile, extracts the
     invoice status from the DOM as `<untrusted_data source_type="web_page">`, and clicks the download
     button (`playwright.click`).
   - **Step 2 (`Desktop` Agent)**: Invokes the local accounting app's `"Import Invoice"` button via the
     OS Accessibility API (`accessibility.invoke_element` on `Windows UI Automation`).
   - Both steps pass through `CatalogBackedPermissionGate` and `SandboxedMcpToolExecutor` and are
     recorded in `SqliteToolAuditLog`.

2. `test_orchestrator_halts_cleanly_when_browser_encounters_captcha_challenge`:
   - Proves Phase 13's non-bypass CAPTCHA rule inside `NovaGraphOrchestrator`:
   - When a webpage presents a CAPTCHA challenge (`captcha_detected=True`), the `Browser` agent emits
     the `ClarificationProposal` (`kind="ask_user"`) produced by `inspect_page_or_escalate_captcha()`,
     and the orchestrator halts the task immediately without executing any click or form-submit tool!
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from nova.core.agents.base import BaseSpecializedAgent
from nova.core.agents.browser_and_desktop import (
    CrossPlatformDesktopAutomationService,
    DesktopElementTarget,
    DesktopPlatformKind,
    PlaywrightBrowserAutomationService,
    WebPageSnapshot,
)
from nova.core.domain.primitives import AgentRole, TaskStatus
from nova.core.model_serving.interface import DeterministicMockModelAdapter
from nova.core.orchestration.checkpoint import SqliteCheckpointStore
from nova.core.orchestration.engine import NovaGraphOrchestrator
from nova.core.permissions.sandbox_and_audit import (
    CatalogBackedPermissionGate,
    FilesystemSandbox,
    SandboxedMcpToolExecutor,
    SqliteToolAuditLog,
    UserPermissionPolicyStore,
)
from nova.core.platform.event_bus import InMemoryEventBus
from nova.core.tools.registry import build_default_tool_registry


class TestPhase6BrowserAndDesktopOrchestratorIntegration:
    """End-to-end integration tests uniting Browser/Desktop services with NovaGraphOrchestrator."""

    def test_end_to_end_browser_dom_and_desktop_accessibility_workflow_in_orchestrator(
        self,
    ) -> None:
        """Full workflow: Browser Agent clicks download on web portal -> Desktop Agent imports via Windows UIA."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            workspace = tmp_path / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)

            bus = InMemoryEventBus()
            registry = build_default_tool_registry()
            policy_store = UserPermissionPolicyStore(registry=registry)
            audit_log = SqliteToolAuditLog(tmp_path / "p6_audit.db", retention_days=30)

            # Auto-approve reversible local writes (`playwright.click` and `accessibility.invoke_element`)
            gate = CatalogBackedPermissionGate(
                registry=registry,
                policy_store=policy_store,
                audit_log=audit_log,
                auto_approve_reversible_local_writes=True,
                default_task_id="task_p6_flow",
                default_session_id="sess_p6_01",
            )

            executor = SandboxedMcpToolExecutor(
                registry=registry,
                filesystem_sandbox=FilesystemSandbox(allowed_roots=(workspace,)),
                audit_log=audit_log,
                event_bus=bus,
                tool_handlers={
                    "playwright.click": lambda args: {"clicked": True},
                    "accessibility.invoke_element": lambda args: {"invoked": True},
                },
                default_task_id="task_p6_flow",
                default_session_id="sess_p6_01",
            )

            browser_service = PlaywrightBrowserAutomationService()
            desktop_service = CrossPlatformDesktopAutomationService(
                platform_backend=DesktopPlatformKind.WINDOWS_UIA,
                event_bus=bus,
            )

            portal_snapshot = WebPageSnapshot(
                url="https://billing.acme.example.com/invoices/2026-10",
                title="Acme Billing Portal",
                dom_text="Invoice #INV-2026-10 ($1,250.00) is ready for download.",
                has_semantic_dom=True,
                captcha_detected=False,
            )
            web_untrusted_item = browser_service.inspect_page_or_escalate_captcha(
                task_id="task_p6_flow",
                session_id="sess_p6_01",
                node_id="step_browser_download",
                snapshot=portal_snapshot,
            )
            assert not hasattr(web_untrusted_item, "question")

            click_proposal, used_vision = browser_service.build_click_proposal_with_vision_fallback(
                node_id="step_browser_download",
                snapshot=portal_snapshot,
                target_selector="#download-inv-2026-10",
                target_visual_description="Download Invoice #INV-2026-10",
            )
            assert used_vision is False

            desktop_plan = desktop_service.build_desktop_action_proposal(
                node_id="step_desktop_import",
                target=DesktopElementTarget(
                    window_title="LedgerPro Accounting — Invoices",
                    accessibility_element_id="UIA_Btn_ImportInvoice",
                    visual_description="Import Invoice Button",
                ),
            )
            assert desktop_plan.coordinate_fallback_engaged is False

            planner_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "plan",
                        "agent_role": "planner",
                        "node_id": "node_planner",
                        "thought": "Step 1: Browser downloads invoice. Step 2: Desktop imports invoice.",
                        "goal_summary": "Download invoice from web portal and import into LedgerPro.",
                        "steps": [
                            {
                                "step_id": "step_browser_download",
                                "target_agent": "browser",
                                "objective": "Click #download-inv-2026-10 on Acme Billing Portal.",
                                "depends_on": [],
                                "estimated_sensitivity": "reversible_write",
                            },
                            {
                                "step_id": "step_desktop_import",
                                "target_agent": "desktop",
                                "objective": "Invoke UIA_Btn_ImportInvoice in LedgerPro Accounting.",
                                "depends_on": ["step_browser_download"],
                                "estimated_sensitivity": "reversible_write",
                            },
                        ],
                    }
                ]
            )

            browser_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "tool_call",
                        "agent_role": "browser",
                        "node_id": "step_browser_download",
                        "thought": click_proposal.thought,
                        "tool_name": click_proposal.tool_name,
                        "arguments": dict(click_proposal.arguments),
                        "declared_sensitivity": click_proposal.declared_sensitivity.value,
                        "rationale": click_proposal.rationale,
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "browser",
                        "node_id": "step_browser_download",
                        "thought": "Clicked download button on Acme Billing Portal.",
                        "content": "Downloaded #INV-2026-10 ($1,250.00) from Acme Billing Portal.",
                        "cited_source_ids": ["web_step_browser_download"],
                        "confidence": 0.99,
                    },
                ]
            )

            desktop_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "tool_call",
                        "agent_role": "desktop",
                        "node_id": "step_desktop_import",
                        "thought": desktop_plan.proposal.thought,
                        "tool_name": desktop_plan.proposal.tool_name,
                        "arguments": dict(desktop_plan.proposal.arguments),
                        "declared_sensitivity": desktop_plan.proposal.declared_sensitivity.value,
                        "rationale": desktop_plan.proposal.rationale,
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "desktop",
                        "node_id": "step_desktop_import",
                        "thought": "Invoked UIA_Btn_ImportInvoice in LedgerPro Accounting.",
                        "content": "Imported #INV-2026-10 into LedgerPro Accounting via Windows UIA.",
                        "cited_source_ids": ["web_step_browser_download"],
                        "confidence": 1.0,
                    },
                ]
            )

            orchestrator = NovaGraphOrchestrator(
                agents={
                    AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model, bus),
                    AgentRole.BROWSER: BaseSpecializedAgent(AgentRole.BROWSER, browser_model, bus),
                    AgentRole.DESKTOP: BaseSpecializedAgent(AgentRole.DESKTOP, desktop_model, bus),
                },
                checkpoint_store=SqliteCheckpointStore(tmp_path / "p6_checkpoints.db"),
                permission_gate=gate,
                tool_executor=executor,
                event_bus=bus,
            )

            final_state = orchestrator.start_task(
                task_id="task_p6_flow",
                session_id="sess_p6_01",
                user_goal="Download #INV-2026-10 from Acme Portal and import into LedgerPro.",
                untrusted_data=(web_untrusted_item,),  # type: ignore[arg-type]
            )

            assert final_state.status == TaskStatus.COMPLETED
            assert final_state.completed_node_ids == {
                "step_browser_download",
                "step_desktop_import",
            }

            logs = audit_log.query_logs(task_id="task_p6_flow")
            assert len(logs) == 2
            assert [entry.tool_name for entry in logs] == [
                "playwright.click",
                "accessibility.invoke_element",
            ]

    def test_orchestrator_halts_cleanly_when_browser_encounters_captcha_challenge(
        self,
    ) -> None:
        """Phase 13 (p. 45): CAPTCHA page causes Browser Agent to emit ClarificationProposal and halt without tool calls."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            workspace = tmp_path / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)

            bus = InMemoryEventBus()
            registry = build_default_tool_registry()
            policy_store = UserPermissionPolicyStore(registry=registry)
            audit_log = SqliteToolAuditLog(tmp_path / "p6_captcha_audit.db", retention_days=30)
            gate = CatalogBackedPermissionGate(
                registry=registry,
                policy_store=policy_store,
                audit_log=audit_log,
            )
            executor = SandboxedMcpToolExecutor(
                registry=registry,
                filesystem_sandbox=FilesystemSandbox(allowed_roots=(workspace,)),
                audit_log=audit_log,
                event_bus=bus,
            )

            browser_service = PlaywrightBrowserAutomationService()
            captcha_snapshot = WebPageSnapshot(
                url="https://billing.acme.example.com/login",
                title="Acme Login",
                dom_text="Verify you are human",
                captcha_detected=True,
                captcha_provider="reCAPTCHA v2",
            )
            clarification = browser_service.inspect_page_or_escalate_captcha(
                task_id="task_p6_captcha",
                session_id="sess_p6_captcha",
                node_id="step_browser",
                snapshot=captcha_snapshot,
            )

            planner_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "plan",
                        "agent_role": "planner",
                        "node_id": "node_planner",
                        "thought": "Navigate to Acme Login.",
                        "goal_summary": "Log into Acme Portal.",
                        "steps": [
                            {
                                "step_id": "step_browser",
                                "target_agent": "browser",
                                "objective": "Open Acme Login page.",
                                "depends_on": [],
                                "estimated_sensitivity": "read_only",
                            }
                        ],
                    }
                ]
            )

            # Browser Agent emits the exact ClarificationProposal generated by `inspect_page_or_escalate_captcha`
            browser_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "ask_user",
                        "agent_role": "browser",
                        "node_id": "step_browser",
                        "thought": clarification.thought,  # type: ignore[union-attr]
                        "question": clarification.question,  # type: ignore[union-attr]
                        "missing_fields": list(clarification.missing_fields),  # type: ignore[union-attr]
                        "options": list(clarification.options),  # type: ignore[union-attr]
                    }
                ]
            )

            orchestrator = NovaGraphOrchestrator(
                agents={
                    AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model, bus),
                    AgentRole.BROWSER: BaseSpecializedAgent(AgentRole.BROWSER, browser_model, bus),
                },
                checkpoint_store=SqliteCheckpointStore(tmp_path / "captcha_checkpoints.db"),
                permission_gate=gate,
                tool_executor=executor,
                event_bus=bus,
            )

            halted_state = orchestrator.start_task(
                task_id="task_p6_captcha",
                session_id="sess_p6_captcha",
                user_goal="Log into Acme Portal.",
            )

            assert halted_state.status == TaskStatus.FAILED
            assert "reCAPTCHA v2" in (halted_state.failure_reason or "")
            assert len(executor.executed_proposals) == 0