"""NOVA Layer 6 State-Graph Orchestrator Engine (ADR-003 / ADR-004 / ADR-006 / ADR-010 / NFR-09).

Real-Life Analogy:
`NovaGraphOrchestrator` is the **Office Manager** of NOVA.
Before this file, we had employees (`BaseSpecializedAgent`), a Project Clipboard (`state.py`),
an Auto-Save system (`checkpoint.py`), a Quality Inspector (`reflection.py`), a Security Guard
(`PermissionGatePort`), and a Tool Room Operator (`ToolExecutorPort`), but you had to call each
one manually in code.

Now, `NovaGraphOrchestrator` runs the entire office automatically:
1. `start_task()`: Asks the `Planner` employee to create a step-by-step flowchart, clips it to the
   Project Clipboard, saves to SQLite, and starts running the steps.
2. `run_until_blocked_or_complete()`:
   - Picks the next ready step from the flowchart.
   - Hands the step to the assigned employee (`Research`, `Comms`, `Coding`, etc.).
   - If the employee proposes a risky tool (like `gmail.send_email`), the Manager first asks the
     **Quality Inspector (Reflection)** to double-check the arguments. If the Inspector spots a
     mistake (like the wrong email address), the Manager sends it back to the employee to fix
     (up to `max_retries`) BEFORE bothering you!
   - Next, the Manager asks the **Security Guard (`PermissionGatePort`)**:
     * If it's safe (`read_only`), the Manager hands the approval stamp to the **Tool Room Operator**,
       writes the tool's output onto the employee's ReAct notepad, saves a checkpoint, and continues.
     * If it needs your permission (`requires_interactive_prompt=True`), the Manager pauses the task
       in `AWAITING_APPROVAL`, saves a checkpoint to SQLite, and waits for your click.
3. `submit_user_approval()`: When you click **Approve** or **Deny** (even after a PC restart!),
   the Manager loads the checkpoint from SQLite, runs the approved tool (or aborts if denied),
   and finishes the rest of the flowchart.
"""

from __future__ import annotations

import logging
from nova.core.orchestration.ports import AuthorizationAlreadyClaimedError

import uuid

from typing import Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from nova.core.agents.base import BaseSpecializedAgent
from nova.core.cognition.planning import compile_plan_proposal
from nova.core.cognition.reflection import evaluate_reflection_trigger
from nova.core.config.settings import NovaSettings, load_settings
from nova.core.domain.agent_spec import get_agent_spec
from nova.core.domain.context import (
    AgentContext,
    TrustedInstruction,
    UntrustedDataItem,
)
from nova.core.domain.events import EventType, NovaEvent
from nova.core.domain.plan import PlanNode, PlanNodeStatus, ExecutionPlanGraph
from nova.core.domain.primitives import AgentRole, TaskStatus, SensitivityTier
from nova.core.domain.proposals import (
    ClarificationProposal,
    CritiqueProposal,
    FinalAnswerProposal,
    PlanProposal,
    ToolCallProposal,
)
from nova.core.orchestration.checkpoint import CheckpointStorePort, CheckpointEnvelope
from nova.core.orchestration.ports import (
    PermissionGatePort,
    PermissionCheckResult,
    UnapprovedToolExecutionError,
    ToolActionContext,
    copy_model,
    ToolExecutionResult,
    ToolExecutorPort,
)
from nova.core.orchestration.state import (
    NodeExecutionRecord,
    TaskExecutionState,
)
from nova.core.platform.event_bus import EventPublisherPort


class OrchestrationExecutionError(RuntimeError):
    """Raised when the orchestrator encounters an unrecoverable protocol or lookup error."""


@runtime_checkable
class OrchestratorEnginePort(Protocol):
    """Abstract interface for NOVA's Layer 6 Graph Orchestrator (ADR-006)."""

    def start_task(
        self,
        task_id: str,
        session_id: str,
        user_goal: str,
        untrusted_data: Sequence[UntrustedDataItem] = (),
        require_citations: bool = False,
    ) -> TaskExecutionState:
        """Create a task, generate its execution plan, and run until completion or approval pause."""

    def run_until_blocked_or_complete(
        self,
        task_id: str,
        untrusted_data: Sequence[UntrustedDataItem] = (),
        require_citations: bool = False,
    ) -> TaskExecutionState:
        """Resume a RUNNING task from its latest checkpoint and advance all ready nodes."""

    def submit_user_approval(
        self,
        task_id: str,
        node_id: str,
        proposal_id: str,
        approved: bool,
        reason: str = "User interactive decision",
        untrusted_data: Sequence[UntrustedDataItem] = (),
        require_citations: bool = False,
    ) -> TaskExecutionState:
        """Apply the user's Approve/Deny decision to a paused task and continue execution."""


class NovaGraphOrchestrator:
    """Deterministic state-graph orchestrator implementing ADR-003, ADR-004, ADR-006, and ADR-010."""

    def __init__(
        self,
        agents: Mapping[AgentRole, BaseSpecializedAgent],
        checkpoint_store: CheckpointStorePort,
        permission_gate: PermissionGatePort,
        tool_executor: ToolExecutorPort,
        event_bus: EventPublisherPort,
        critic_agent: Optional[BaseSpecializedAgent] = None,
        settings: Optional[NovaSettings] = None,
    ) -> None:
        if AgentRole.PLANNER not in agents:
            raise ValueError(
                "NovaGraphOrchestrator requires a registered AgentRole.PLANNER agent."
            )
        self._agents: Dict[AgentRole, BaseSpecializedAgent] = dict(agents)
        self._critic_agent = critic_agent
        self._checkpoint_store = checkpoint_store
        self._permission_gate = permission_gate
        self._tool_executor = tool_executor
        self._event_bus = event_bus
        self._settings = settings if settings is not None else load_settings(environ={})

    def _save_checkpoint(self, state: TaskExecutionState, node_id: Optional[str] = None,
                         *, expected_seq: Optional[int] = None) -> CheckpointEnvelope:
        event = None
        if state.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            rec = state.node_records.get(node_id)
            event = NovaEvent(event_type=EventType.TASK_COMPLETED if state.status == TaskStatus.COMPLETED else EventType.TASK_FAILED,
                task_id=state.task_id, session_id=state.session_id, node_id=node_id,
                agent_role=rec.agent_role if rec else None,
                correlation_id=(rec.pending_tool_proposal.proposal_id if rec and rec.pending_tool_proposal else
                    self._terminal_correlation(state.task_id, node_id)),
                payload=({"completed_nodes": sorted(state.completed_node_ids)} if state.status == TaskStatus.COMPLETED
                         else {"reason": state.failure_reason}))
        try:
            saved = self._checkpoint_store.save_checkpoint(state, node_id=node_id, protect_terminal=True,
                                                           expected_seq=expected_seq, terminal_event=event)
        except ValueError as exc:
            if expected_seq is not None:
                raise
            raise OrchestrationExecutionError("Task has a terminal outcome or another execution owner; stale execution stopped.") from exc
        if saved.state != state:
            raise OrchestrationExecutionError("Task already has a terminal outcome; stale execution stopped.")
        if event and any(e.event_id == event.event_id for e in self._checkpoint_store.terminal_events(state.task_id)):
            try:
                self._event_bus.publish(event)
            except Exception:
                logging.getLogger(__name__).exception("Terminal event persisted; live publication failed")
        return saved

    def _terminal_correlation(self, task_id: str, node_id: Optional[str]) -> Optional[str]:
        latest = self._checkpoint_store.load_latest(task_id)
        rec = latest.state.node_records.get(node_id) if latest else None
        return rec.pending_tool_proposal.proposal_id if rec and rec.pending_tool_proposal else None

    @staticmethod
    def _action_context(state: TaskExecutionState, node_id: str) -> ToolActionContext:
        # Identity comes from the runtime graph, never from proposal fields or HTTP arguments.
        rec = state.node_records[node_id]
        return ToolActionContext(task_id=state.task_id, session_id=state.session_id,
                                 node_id=node_id, agent_role=rec.agent_role)

    def start_assistant_action(self, *, tool_name: str, arguments: dict,
                               session_id: str = "sess_personal_chat") -> TaskExecutionState:
        roles = {"notes.create_note": AgentRole.CODING, "calendar.create_event": AgentRole.COMMS}
        if tool_name not in roles:
            raise ValueError("Unsupported assistant action.")
        role = roles[tool_name]
        task_id = "task_assistant_" + uuid.uuid4().hex
        node_id = "assistant_action"
        proposal = ToolCallProposal(agent_role=role, node_id=node_id,
            thought="Explicit assistant command; no model inference performed.",
            tool_name=tool_name, arguments=arguments,
            declared_sensitivity=SensitivityTier.REVERSIBLE_WRITE,
            rationale="Execute the exact note/calendar action requested by the user.")
        node = PlanNode(node_id=node_id, target_agent=role, objective=tool_name)
        state = TaskExecutionState.create_initial(task_id=task_id, session_id=session_id,
                                                  user_goal=tool_name)
        state = state.attach_plan_graph(ExecutionPlanGraph(task_id=task_id,
            goal_summary=tool_name, nodes=(node,)))
        rec = copy_model(state.node_records[node_id], **{
            "direct_action": True, "pending_tool_proposal": proposal})
        state = state.update_node_record(rec)
        self._save_checkpoint(state, node_id=node_id)
        return self.run_until_blocked_or_complete(task_id)

    def _execute_authorized(self, state: TaskExecutionState, proposal: ToolCallProposal,
                            stamp: PermissionCheckResult) -> ToolExecutionResult:
        context = self._action_context(state, proposal.node_id)
        try:
            result = self._tool_executor.execute_approved_tool(proposal, stamp, context=context,
                on_started=lambda: self._emit(state, EventType.TOOL_STARTED,
                    {"tool_name": proposal.tool_name}, proposal.node_id, context.agent_role, proposal.proposal_id))
        except UnapprovedToolExecutionError:
            # Rejected attempts are not new task outcomes and cannot overwrite the winner.
            raise
        except Exception:
            self._emit(state, EventType.TOOL_FAILED, {"tool_name": proposal.tool_name},
                       proposal.node_id, context.agent_role, proposal.proposal_id)
            raise
        self._emit(state, EventType.TOOL_COMPLETED if result.status == "success" else EventType.TOOL_FAILED,
                   {"tool_name": proposal.tool_name, "status": result.status},
                   proposal.node_id, context.agent_role, proposal.proposal_id)
        return result

    def _complete_direct_action(self, state: TaskExecutionState, rec: NodeExecutionRecord,
                                proposal: ToolCallProposal, result: ToolExecutionResult) -> TaskExecutionState:
        if result.status != "success":
            node = next(n for n in state.plan_graph.nodes if n.node_id == rec.node_id)
            return self._fail_node(state, node, rec, result.sanitized_output)
        final = FinalAnswerProposal(agent_role=rec.agent_role, node_id=rec.node_id,
            thought="Result returned by the actual local handler.", content=result.sanitized_output)
        done = copy_model(rec, **{"status": PlanNodeStatus.COMPLETED,
            "pending_tool_proposal": None, "final_answer": final})
        state = state.update_node_record(done)
        state = self._save_checkpoint(state, node_id=rec.node_id).state
        if state.status == TaskStatus.COMPLETED:
            self._emit(state, EventType.TASK_COMPLETED, {"completed_nodes": sorted(state.completed_node_ids)},
                       rec.node_id, rec.agent_role, proposal.proposal_id)
        return state

    def start_task(
        self,
        task_id: str,
        session_id: str,
        user_goal: str,
        untrusted_data: Sequence[UntrustedDataItem] = (),
        require_citations: bool = False,
    ) -> TaskExecutionState:
        """Initialize a new task, run the PlannerAgent, compile the plan graph, and execute."""
        # 1. Create initial QUEUED clipboard state and save Checkpoint #0
        state = TaskExecutionState.create_initial(
            task_id=task_id,
            session_id=session_id,
            user_goal=user_goal,
            max_graph_steps=self._settings.max_graph_steps,
        )
        self._save_checkpoint(state, node_id=None)
        self._emit(
            state=state,
            event_type=EventType.TASK_STARTED,
            payload={"user_goal": user_goal},
        )

        # 2. Transition to PLANNING and invoke PlannerAgent
        state = state.transition_task_status(TaskStatus.PLANNING)
        self._save_checkpoint(state, node_id="node_planner")
        self._emit(
            state=state,
            event_type=EventType.TASK_PLANNING,
            node_id="node_planner",
            agent_role=AgentRole.PLANNER,
            payload={"user_goal": user_goal},
        )

        planner_agent = self._agents[AgentRole.PLANNER]
        planner_spec = get_agent_spec(AgentRole.PLANNER)
        planner_ctx = AgentContext(
            task_id=task_id,
            session_id=session_id,
            node_id="node_planner",
            instruction=TrustedInstruction(
                agent_role=AgentRole.PLANNER,
                system_mandate=planner_spec.mandate,
                task_objective=user_goal,
                allowed_tools=planner_spec.allowed_tools,
            ),
            untrusted_data=tuple(untrusted_data),
        )

        plan_proposal = planner_agent.propose_next_action(planner_ctx)
        if not isinstance(plan_proposal, PlanProposal):
            state = state.transition_task_status(
                TaskStatus.FAILED,
                failure_reason=f"PlannerAgent must return a PlanProposal, got {plan_proposal.kind!r}.",
            )
            self._save_checkpoint(state, node_id="node_planner")
            self._emit(
                state=state,
                event_type=EventType.TASK_FAILED,
                node_id="node_planner",
                agent_role=AgentRole.PLANNER,
                payload={"reason": state.failure_reason},
            )
            return state

        plan_graph = compile_plan_proposal(
            task_id=task_id,
            proposal=plan_proposal,
            max_graph_steps=self._settings.max_graph_steps,
            default_max_react_iterations=self._settings.max_react_iterations_per_node,
            default_max_retries=self._settings.max_retries_per_node,
        )
        state = state.attach_plan_graph(plan_graph)
        self._save_checkpoint(state, node_id="node_planner")

        # 3. Advance the execution graph until completion or human approval pause
        return self.run_until_blocked_or_complete(
            task_id=task_id,
            untrusted_data=untrusted_data,
            require_citations=require_citations,
        )

    def run_until_blocked_or_complete(
        self,
        task_id: str,
        untrusted_data: Sequence[UntrustedDataItem] = (),
        require_citations: bool = False,
    ) -> TaskExecutionState:
        """Load the latest checkpoint for `task_id` and execute ready nodes until paused or finished."""
        envelope = self._checkpoint_store.load_latest(task_id)
        if envelope is None:
            raise OrchestrationExecutionError(f"No checkpoint found for task_id {task_id!r}.")

        state = envelope.state
        if state.approval_attempt_id is not None:
            return state
        if state.status != TaskStatus.RUNNING or state.plan_graph is None:
            return state

        while state.status == TaskStatus.RUNNING and state.plan_graph is not None:
            ready_nodes = state.plan_graph.ready_nodes(
                completed_node_ids=state.completed_node_ids,
                running_node_ids=set(),
                max_concurrency=self._settings.max_concurrent_agents,
            )
            if not ready_nodes:
                break

            for node in ready_nodes:
                state = self._execute_single_node(
                    state=state,
                    node=node,
                    untrusted_data=tuple(untrusted_data),
                    require_citations=require_citations,
                )
                if state.status != TaskStatus.RUNNING:
                    break

        if state.status == TaskStatus.COMPLETED:
            self._emit(
                state=state,
                event_type=EventType.TASK_COMPLETED,
                payload={"completed_nodes": sorted(state.completed_node_ids)},
            )

        return state

    def submit_user_approval(self, task_id: str, node_id: str, proposal_id: str,
                             approved: bool, reason: str = "User interactive decision",
                             untrusted_data: Sequence[UntrustedDataItem] = (),
                             require_citations: bool = False) -> TaskExecutionState:
        try:
            return self._submit_user_approval(task_id, node_id, proposal_id, approved, reason,
                                             untrusted_data, require_citations)
        except (PermissionError, ValueError, OrchestrationExecutionError) as exc:
            recorder = getattr(self._permission_gate, "record_rejected_attempt", None)
            if recorder:
                context = None
                envelope = self._checkpoint_store.load_latest(task_id)
                if envelope and node_id in envelope.state.node_records:
                    context = self._action_context(envelope.state, node_id)
                recorder(None, context, exc, submitted={"task_id": task_id,
                         "node_id": node_id, "proposal_id": proposal_id})
            raise

    def _submit_user_approval(
        self,
        task_id: str,
        node_id: str,
        proposal_id: str,
        approved: bool,
        reason: str = "User interactive decision",
        untrusted_data: Sequence[UntrustedDataItem] = (),
        require_citations: bool = False,
    ) -> TaskExecutionState:
        """Apply a human Approve/Deny click to a paused task and resume execution."""
        envelope = self._checkpoint_store.load_latest(task_id)
        if envelope is None:
            raise OrchestrationExecutionError(f"No checkpoint found for task_id {task_id!r}.")

        state = envelope.state
        if state.status != TaskStatus.AWAITING_APPROVAL:
            raise OrchestrationExecutionError(
                f"Task {task_id!r} is in status {state.status.value!r}, not 'awaiting_approval'."
            )

        rec = state.node_records.get(node_id)
        if rec is None or rec.pending_tool_proposal is None:
            raise OrchestrationExecutionError(
                f"Node {node_id!r} in task {task_id!r} has no pending_tool_proposal."
            )

        pending_proposal = rec.pending_tool_proposal
        if pending_proposal.proposal_id != proposal_id:
            raise OrchestrationExecutionError(
                f"Pending proposal_id {pending_proposal.proposal_id!r} does not match "
                f"submitted proposal_id {proposal_id!r}."
            )

        owned = copy_model(state.transition_task_status(TaskStatus.RUNNING),
                           approval_attempt_id=uuid.uuid4().hex)
        try:
            self._save_checkpoint(owned, node_id=node_id, expected_seq=envelope.checkpoint_seq)
        except ValueError:
            raise OrchestrationExecutionError("Approval is already being resolved; reload task status.")
        try:
            return self._resolve_owned_approval(owned, node_id, proposal_id, approved, reason,
                                                untrusted_data, require_citations)
        except AuthorizationAlreadyClaimedError:
            # Legacy or externally claimed action: outcome may be uncertain. Never replay.
            raise
        except UnapprovedToolExecutionError as exc:
            failed_rec = copy_model(rec, status=PlanNodeStatus.FAILED, pending_tool_proposal=None,
                                    last_error="Authorization rejected before execution.")
            failed = owned.update_node_record(failed_rec, new_task_status=TaskStatus.FAILED,
                                              failure_reason="Authorization rejected before execution.")
            event = NovaEvent(event_type=EventType.TASK_FAILED, task_id=failed.task_id,
                session_id=failed.session_id, node_id=node_id, agent_role=rec.agent_role,
                correlation_id=proposal_id, payload={"reason": "authorization_rejected", "execution_started": False})
            try:
                saved = self._checkpoint_store.save_checkpoint(failed, node_id=node_id, protect_terminal=True,
                    expected_seq=owned.checkpoint_seq, terminal_event=event)
            except ValueError:
                saved = self._checkpoint_store.load_latest(owned.task_id)
            if saved.state == failed:
                try:
                    self._event_bus.publish(event)
                except Exception:
                    logging.getLogger(__name__).exception("Rejection event persisted; live publication failed")
            exc.task_status = saved.state.status.value
            exc.task_id = saved.state.task_id
            raise

    def _resolve_owned_approval(self, state: TaskExecutionState, node_id: str, proposal_id: str,
                                approved: bool, reason: str, untrusted_data: Sequence[UntrustedDataItem],
                                require_citations: bool) -> TaskExecutionState:
        task_id = state.task_id
        rec = state.node_records[node_id]
        pending_proposal = rec.pending_tool_proposal
        # Record the human decision in the Permission Gate
        self._permission_gate.record_user_decision(
            proposal_id=proposal_id,
            approved=approved,
            reason=reason,
            proposal=pending_proposal, context=self._action_context(state, node_id),
        )

        if not approved:
            self._permission_gate.evaluate_tool_proposal(pending_proposal, context=self._action_context(state, node_id))
            self._emit(
                state=state,
                event_type=EventType.PERMISSION_DENIED,
                node_id=node_id,
                agent_role=rec.agent_role,
                correlation_id=proposal_id,
                payload={"tool_name": pending_proposal.tool_name, "reason": reason},
            )
            failed_rec = NodeExecutionRecord(
                node_id=rec.node_id,
                agent_role=rec.agent_role,
                status=PlanNodeStatus.FAILED,
                attempt_count=rec.attempt_count,
                react_state=rec.react_state,
                pending_tool_proposal=None,
                last_error=f"Permission denied by user: {reason}",
            )
            state = state.update_node_record(
                failed_rec,
                new_task_status=TaskStatus.FAILED,
                failure_reason=f"User denied tool {pending_proposal.tool_name!r}: {reason}",
            )
            self._save_checkpoint(state, node_id=node_id)
            self._emit(
                state=state,
                event_type=EventType.TASK_FAILED,
                node_id=node_id,
                agent_role=rec.agent_role,
                payload={"reason": state.failure_reason},
                correlation_id=proposal_id,
            )
            return state

        # Approved! Re-evaluate through the Permission Gate to obtain the official USER_APPROVED stamp
        stamp = self._permission_gate.evaluate_tool_proposal(pending_proposal, context=self._action_context(state, node_id))
        if not stamp.is_approved:
            node = next(n for n in state.plan_graph.nodes if n.node_id == node_id)
            return self._fail_node(state, node, rec, stamp.reason)
        self._emit(
            state=state,
            event_type=EventType.PERMISSION_APPROVED,
            node_id=node_id,
            agent_role=rec.agent_role,
            correlation_id=proposal_id,
            payload={
                "tool_name": pending_proposal.tool_name,
                "decision": stamp.decision.value if stamp.decision else "user_approved",
                "reason": reason,
            },
        )

        # Resume task to RUNNING and execute the approved tool
        try:
            tool_result = self._execute_authorized(state, pending_proposal, stamp)
        except UnapprovedToolExecutionError:
            raise
        except Exception as exc:
            node = next(n for n in state.plan_graph.nodes if n.node_id == node_id)
            return self._fail_node(state, node, rec, str(exc))
        if rec.direct_action:
            return self._complete_direct_action(state, rec, pending_proposal, tool_result)
        if tool_result.status != "success":
            node = next(n for n in state.plan_graph.nodes if n.node_id == node_id)
            return self._fail_node(state, node, rec, tool_result.sanitized_output)

        updated_react = rec.react_state.record_observation(
            proposal=pending_proposal,
            observation_text=tool_result.sanitized_output,
            status="success" if tool_result.status == "success" else "error",
        )
        resumed_rec = NodeExecutionRecord(
            node_id=rec.node_id,
            agent_role=rec.agent_role,
            status=PlanNodeStatus.READY,
            attempt_count=rec.attempt_count,
            react_state=updated_react,
            pending_tool_proposal=None,
            last_error=None,
        )
        expected_seq = state.checkpoint_seq
        state = copy_model(state.update_node_record(resumed_rec, new_task_status=TaskStatus.RUNNING), approval_attempt_id=None)
        self._save_checkpoint(state, node_id=node_id, expected_seq=expected_seq)

        # Continue running the node (and any downstream nodes)
        return self.run_until_blocked_or_complete(
            task_id=task_id,
            untrusted_data=untrusted_data,
            require_citations=require_citations,
        )

    def _execute_single_node(
        self,
        state: TaskExecutionState,
        node: PlanNode,
        untrusted_data: Tuple[UntrustedDataItem, ...],
        require_citations: bool,
    ) -> TaskExecutionState:
        """Run bounded ReAct & Reflection turns inside `node` until it completes, pauses, or fails."""
        agent = self._agents.get(node.target_agent)
        if agent is None:
            raise OrchestrationExecutionError(
                f"No specialized agent registered for role {node.target_agent.value!r}."
            )
        agent_spec = get_agent_spec(node.target_agent)

        while state.status == TaskStatus.RUNNING:
            rec = state.node_records[node.node_id]

            # Assemble working memory from upstream dependencies + any revision instructions
            wm_parts = []
            upstream_summary = state.build_upstream_outputs_summary(node.node_id)
            if upstream_summary:
                wm_parts.append(upstream_summary)
            if rec.last_error:
                wm_parts.append(f"[Inspector Revision Feedback]:\n{rec.last_error}")
            working_memory = "\n\n".join(wm_parts) if wm_parts else None

            ctx = AgentContext(
                task_id=state.task_id,
                session_id=state.session_id,
                node_id=node.node_id,
                instruction=TrustedInstruction(
                    agent_role=node.target_agent,
                    system_mandate=agent_spec.mandate,
                    task_objective=node.objective,
                    allowed_tools=agent_spec.allowed_tools,
                ),
                untrusted_data=untrusted_data,
                working_memory_summary=working_memory,
            )

            try:
                proposal = (rec.pending_tool_proposal if rec.pending_tool_proposal is not None
                            else agent.propose_next_action(ctx, react_state=rec.react_state))
            except Exception as exc:
                failed_rec = NodeExecutionRecord(
                    node_id=node.node_id,
                    agent_role=node.target_agent,
                    status=PlanNodeStatus.FAILED,
                    attempt_count=rec.attempt_count + 1,
                    react_state=rec.react_state,
                    last_error=str(exc),
                )
                state = state.update_node_record(
                    failed_rec,
                    new_task_status=TaskStatus.FAILED,
                    failure_reason=str(exc),
                )
                self._save_checkpoint(state, node_id=node.node_id)
                self._emit(
                    state=state,
                    event_type=EventType.TASK_FAILED,
                    node_id=node.node_id,
                    agent_role=node.target_agent,
                    payload={"reason": str(exc)},
                )
                return state

            # --- Case 1: Agent proposed a FinalAnswerProposal ---
            if isinstance(proposal, FinalAnswerProposal):
                reflection = evaluate_reflection_trigger(
                    proposal=proposal,
                    node=node,
                    require_citations=require_citations,
                )
                if reflection.should_reflect and self._critic_agent is not None and not rec.direct_action:
                    critique = self._run_reflection_check(
                        state=state,
                        node=node,
                        candidate_proposal=proposal,
                        verification_questions=reflection.verification_questions,
                    )
                    if critique.verdict == "revise":
                        state = self._apply_revision_or_fail(
                            state=state,
                            node=node,
                            rec=rec,
                            revision_instructions=critique.revision_instructions
                            or "Revise answer per reflection critique.",
                        )
                        if state.status != TaskStatus.RUNNING:
                            return state
                        continue
                    if critique.verdict == "abort":
                        return self._fail_node(
                            state=state,
                            node=node,
                            rec=rec,
                            reason="Reflection critic aborted final answer.",
                        )

                done_rec = NodeExecutionRecord(
                    node_id=node.node_id,
                    agent_role=node.target_agent,
                    status=PlanNodeStatus.COMPLETED,
                    attempt_count=rec.attempt_count + 1,
                    react_state=rec.react_state,
                    pending_tool_proposal=None,
                    final_answer=proposal,
                    last_error=None,
                )
                state = state.update_node_record(done_rec)
                self._save_checkpoint(state, node_id=node.node_id)
                return state

            # --- Case 2: Agent proposed a ToolCallProposal ---
            if isinstance(proposal, ToolCallProposal):
                reflection = evaluate_reflection_trigger(
                    proposal=proposal,
                    node=node,
                    require_citations=require_citations,
                )
                if reflection.should_reflect and self._critic_agent is not None and not rec.direct_action:
                    self._emit(
                        state=state,
                        event_type=EventType.AGENT_REFLECTION_TRIGGERED,
                        node_id=node.node_id,
                        agent_role=node.target_agent,
                        correlation_id=proposal.proposal_id,
                        payload={
                            "reason": reflection.reason.value,
                            "verification_questions": list(reflection.verification_questions),
                        },
                    )
                    critique = self._run_reflection_check(
                        state=state,
                        node=node,
                        candidate_proposal=proposal,
                        verification_questions=reflection.verification_questions,
                    )
                    if critique.verdict == "revise":
                        state = self._apply_revision_or_fail(
                            state=state,
                            node=node,
                            rec=rec,
                            revision_instructions=critique.revision_instructions
                            or "Revise tool arguments per reflection critique.",
                        )
                        if state.status != TaskStatus.RUNNING:
                            return state
                        continue
                    if critique.verdict == "abort":
                        return self._fail_node(
                            state=state,
                            node=node,
                            rec=rec,
                            reason="Reflection critic aborted unsafe tool call.",
                        )

                # Evaluate through the Permission Gate (ADR-010)
                try:
                    stamp = self._permission_gate.evaluate_tool_proposal(proposal,
                        context=self._action_context(state, node.node_id))
                except UnapprovedToolExecutionError:
                    raise
                except Exception as exc:
                    return self._fail_node(state, node, rec, str(exc))

                if stamp.requires_interactive_prompt:
                    paused_rec = NodeExecutionRecord(
                        node_id=node.node_id,
                        agent_role=node.target_agent,
                        status=PlanNodeStatus.AWAITING_APPROVAL,
                        attempt_count=rec.attempt_count + 1,
                        react_state=rec.react_state,
                        pending_tool_proposal=proposal,
                        direct_action=rec.direct_action,
                        approval_expires_at=stamp.expires_at,
                        last_error=None,
                    )
                    state = state.update_node_record(
                        paused_rec,
                        new_task_status=TaskStatus.AWAITING_APPROVAL,
                    )
                    self._save_checkpoint(state, node_id=node.node_id)
                    self._emit(
                        state=state,
                        event_type=EventType.PERMISSION_REQUESTED,
                        node_id=node.node_id,
                        agent_role=node.target_agent,
                        correlation_id=proposal.proposal_id,
                        payload={
                            "proposal_id": proposal.proposal_id,
                            "tool_name": proposal.tool_name,
                            "arguments": dict(proposal.arguments),
                            "effective_sensitivity": stamp.effective_sensitivity.value,
                            "rationale": proposal.rationale,
                        },
                    )
                    return state

                if stamp.is_denied:
                    self._emit(
                        state=state,
                        event_type=EventType.PERMISSION_DENIED,
                        node_id=node.node_id,
                        agent_role=node.target_agent,
                        correlation_id=proposal.proposal_id,
                        payload={
                            "tool_name": proposal.tool_name,
                            "reason": stamp.reason,
                        },
                    )
                    return self._fail_node(
                        state=state,
                        node=node,
                        rec=rec,
                        reason=f"Permission denied for tool {proposal.tool_name!r}: {stamp.reason}",
                    )

                # Tool is approved (e.g., AUTO_APPROVED read_only tool) -> execute immediately
                self._emit(
                    state=state,
                    event_type=EventType.PERMISSION_APPROVED,
                    node_id=node.node_id,
                    agent_role=node.target_agent,
                    correlation_id=proposal.proposal_id,
                    payload={
                        "tool_name": proposal.tool_name,
                        "decision": stamp.decision.value if stamp.decision else "auto_approved",
                    },
                )
                # Persist the exact proposal before claiming it. Recovery must not ask
                # the model for a new proposal ID and repeat an uncertain side effect.
                rec = copy_model(rec, **{"pending_tool_proposal": proposal})
                state = state.update_node_record(rec)
                self._save_checkpoint(state, node_id=node.node_id)
                try:
                    tool_result = self._execute_authorized(state, proposal, stamp)
                except UnapprovedToolExecutionError:
                    raise
                except Exception as exc:
                    return self._fail_node(state, node, rec, str(exc))
                if rec.direct_action:
                    return self._complete_direct_action(state, rec, proposal, tool_result)
                if tool_result.status != "success":
                    return self._fail_node(state, node, rec, tool_result.sanitized_output)

                updated_react = rec.react_state.record_observation(
                    proposal=proposal,
                    observation_text=tool_result.sanitized_output,
                    status="success" if tool_result.status == "success" else "error",
                )
                running_rec = NodeExecutionRecord(
                    node_id=node.node_id,
                    agent_role=node.target_agent,
                    status=PlanNodeStatus.RUNNING,
                    attempt_count=rec.attempt_count,
                    react_state=updated_react,
                    pending_tool_proposal=None,
                    last_error=None,
                )
                state = state.update_node_record(running_rec)
                self._save_checkpoint(state, node_id=node.node_id)
                continue

            # --- Case 3: ClarificationProposal or unexpected proposal kind ---
            if isinstance(proposal, ClarificationProposal):
                return self._fail_node(
                    state=state,
                    node=node,
                    rec=rec,
                    reason=f"Clarification required from user: {proposal.question}",
                )

            return self._fail_node(
                state=state,
                node=node,
                rec=rec,
                reason=f"Unsupported proposal kind {proposal.kind!r} inside node execution.",
            )

        return state

    def _run_reflection_check(
        self,
        state: TaskExecutionState,
        node: PlanNode,
        candidate_proposal: ToolCallProposal | FinalAnswerProposal,
        verification_questions: Tuple[str, ...],
    ) -> CritiqueProposal:
        """Invoke the critic agent to evaluate a high-stakes proposal against verification questions."""
        assert self._critic_agent is not None
        critic_spec = get_agent_spec(self._critic_agent.role)
        questions_str = "\n".join(f"- {q}" for q in verification_questions)
        critic_ctx = AgentContext(
            task_id=state.task_id,
            session_id=state.session_id,
            node_id=f"{node.node_id}_reflect",
            instruction=TrustedInstruction(
                agent_role=self._critic_agent.role,
                system_mandate=critic_spec.mandate,
                task_objective=(
                    f"Evaluate candidate proposal {candidate_proposal.proposal_id} for node "
                    f"'{node.node_id}' ({node.objective}). Verification questions:\n{questions_str}"
                ),
                allowed_tools=critic_spec.allowed_tools,
            ),
            working_memory_summary=str(candidate_proposal),
        )
        critique = self._critic_agent.propose_next_action(critic_ctx)
        if not isinstance(critique, CritiqueProposal):
            raise OrchestrationExecutionError(
                f"Critic agent must return CritiqueProposal, got {critique.kind!r}."
            )
        return critique

    def _apply_revision_or_fail(
        self,
        state: TaskExecutionState,
        node: PlanNode,
        rec: NodeExecutionRecord,
        revision_instructions: str,
    ) -> TaskExecutionState:
        """Record revision feedback and retry if within `node.max_retries`, else fail closed."""
        next_attempt = rec.attempt_count + 1
        if next_attempt > node.max_retries:
            return self._fail_node(
                state=state,
                node=node,
                rec=rec,
                reason=(
                    f"Node {node.node_id!r} exceeded max_retries={node.max_retries} "
                    f"after reflection revision: {revision_instructions}"
                ),
            )

        revised_rec = NodeExecutionRecord(
            node_id=node.node_id,
            agent_role=node.target_agent,
            status=PlanNodeStatus.RUNNING,
            attempt_count=next_attempt,
            react_state=rec.react_state,
            pending_tool_proposal=None,
            last_error=revision_instructions,
        )
        state = state.update_node_record(revised_rec)
        self._save_checkpoint(state, node_id=node.node_id)
        return state

    def _fail_node(
        self,
        state: TaskExecutionState,
        node: PlanNode,
        rec: NodeExecutionRecord,
        reason: str,
    ) -> TaskExecutionState:
        failed_rec = NodeExecutionRecord(
            node_id=node.node_id,
            agent_role=node.target_agent,
            status=PlanNodeStatus.FAILED,
            attempt_count=rec.attempt_count + 1,
            react_state=rec.react_state,
            pending_tool_proposal=None,
            last_error=reason,
        )
        state = state.update_node_record(
            failed_rec,
            new_task_status=TaskStatus.FAILED,
            failure_reason=reason,
        )
        state = self._save_checkpoint(state, node_id=node.node_id).state
        if state.status != TaskStatus.FAILED:
            return state
        self._emit(
            state=state,
            event_type=EventType.TASK_FAILED,
            node_id=node.node_id,
            agent_role=node.target_agent,
            payload={"reason": reason},
            correlation_id=rec.pending_tool_proposal.proposal_id if rec.pending_tool_proposal else None,
        )
        return state

    def _emit(
        self,
        state: TaskExecutionState,
        event_type: EventType,
        payload: dict[str, object],
        node_id: Optional[str] = None,
        agent_role: Optional[AgentRole] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        terminal = (EventType.TOOL_COMPLETED, EventType.TOOL_FAILED,
                    EventType.TASK_COMPLETED, EventType.TASK_FAILED)
        if event_type in terminal:
            # Completed and failed share a reservation, preventing conflicting outcomes.
            key = ("tool:" + str(correlation_id)) if event_type in terminal[:2] else event_type.value
            if event_type in terminal[2:]:
                latest = self._checkpoint_store.load_latest(state.task_id)
                expected = TaskStatus.COMPLETED if event_type == EventType.TASK_COMPLETED else TaskStatus.FAILED
                if latest is None or latest.state.status != expected:
                    return
            if not self._checkpoint_store.claim_terminal_event(state.task_id, key):
                return
        self._event_bus.publish(
            NovaEvent(
                event_type=event_type,
                task_id=state.task_id,
                session_id=state.session_id,
                node_id=node_id,
                agent_role=agent_role,
                correlation_id=correlation_id,
                payload=payload,
            )
        )
