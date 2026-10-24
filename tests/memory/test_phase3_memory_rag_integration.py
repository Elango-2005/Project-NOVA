"""NOVA Task 3.3: Phase 3 End-to-End Memory + Hybrid RAG + Orchestrator Integration Suite.

Real-Life Analogy:
This test file is the **Final Exam for Phase 3**.
In Task 3.1 we built the **Sticky-Note Pad & 3-Drawer Filing Cabinet (`memory.py`)**, and in
Task 3.2 we built the **Office Librarian (`retrieval.py`)**.
Now, in Task 3.3, we plug BOTH of them directly into our **Office Manager (`NovaGraphOrchestrator`)**:

1. `test_orchestrator_runs_rag_retrieve_and_uses_semantic_memory_for_email_workflow`:
   - Pre-loads Drawer 1 (`SEMANTIC` memory) with Sam's email (`sam@accounting.example.com`).
   - Pre-loads the Office Librarian (`HybridRetrievalEngine`) with 3 local invoices (`sept`, `oct`, `nov`).
   - Registers `"rag.retrieve"` on the Tool Room Operator (`InMemoryMockToolExecutor`) so when the
     `Research` agent proposes `rag.retrieve(query="#INV-2026-10")`, it runs a real Hybrid BM25+Vector+Rerank
     search and returns the exact chunk ID (`doc_inv_oct#chunk_0`).
   - Runs the full workflow through `NovaGraphOrchestrator` (Plan -> Research via RAG -> Pause at
     Permission Gate -> User Approves -> Email Sent -> Promote Working Note & Log Episodic Diary).

2. `test_reflection_gate_forces_research_agent_to_cite_rag_chunk_id`:
   - Proves Phase 9's citation-grounding rule (`require_citations=True`):
   - On Attempt #1, the `Research` agent forgets to include `cited_source_ids` (`[]`).
   - The Quality Inspector (`evaluate_reflection_trigger`) catches `MISSING_REQUIRED_CITATIONS`,
     and the Critic forces the `Research` agent to revise its answer and cite `"doc_inv_oct#chunk_0"`
     on Attempt #2!
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.core.agents.base import BaseSpecializedAgent
from nova.core.cognition.memory import (
    MemoryManager,
    SqliteLongTermMemoryStore,
    WorkingMemoryStore,
)
from nova.core.cognition.retrieval import (
    HybridRetrievalEngine,
    SqliteHybridChunkIndex,
)
from nova.core.domain.events import EventType
from nova.core.domain.primitives import (
    AgentRole,
    DataClassification,
    MemoryPartition,
    TaskStatus,
)
from nova.core.model_serving.interface import DeterministicMockModelAdapter
from nova.core.orchestration.checkpoint import SqliteCheckpointStore
from nova.core.orchestration.engine import NovaGraphOrchestrator
from nova.core.orchestration.ports import (
    DefaultPolicyPermissionGate,
    InMemoryMockToolExecutor,
)
from nova.core.platform.event_bus import InMemoryEventBus


class TestPhase3MemoryAndRagIntegration(unittest.TestCase):
    """End-to-end integration tests uniting MemoryManager, HybridRetrievalEngine, and NovaGraphOrchestrator."""

    def test_orchestrator_runs_rag_retrieve_and_uses_semantic_memory_for_email_workflow(self) -> None:
        """Full Phase 3 workflow: Semantic Memory lookup + Hybrid RAG tool call + Approval + Episodic logging."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bus = InMemoryEventBus()

            # 1. Set up the Two-Tier Memory Manager (Sticky-Note Pad + 3-Drawer SQLite Cabinet)
            memory_manager = MemoryManager(
                working_store=WorkingMemoryStore(),
                long_term_store=SqliteLongTermMemoryStore(tmp_path / "nova_memory.db"),
                event_bus=bus,
            )

            # Pre-populate Drawer 1 (SEMANTIC) with Sam's contact info from a past session
            memory_manager.write_memory(
                agent_role=AgentRole.RESEARCH,
                partition=MemoryPartition.SEMANTIC,
                memory_key="contact.sam.email",
                content="Sam in accounting uses email address sam@accounting.example.com.",
                tags=("sam", "accounting", "email"),
                importance=0.95,
                task_id="task_setup",
                session_id="sess_setup",
            )

            # 2. Set up the Office Librarian (HybridRetrievalEngine) and index 3 local invoices
            rag_index = SqliteHybridChunkIndex(tmp_path / "nova_rag.db")
            rag_engine = HybridRetrievalEngine(index=rag_index)

            rag_engine.ingest_document(
                document_id="doc_inv_sept",
                source_uri="file:///invoices/sept_2026.pdf",
                content="Vendor: Beta Cloud. Invoice #INV-2026-09. Total Due: $410.00.",
                data_classification=DataClassification.SENSITIVE_CATEGORY,
            )
            rag_engine.ingest_document(
                document_id="doc_inv_oct",
                source_uri="file:///invoices/oct_2026.pdf",
                content="Vendor: Acme Corp. Invoice #INV-2026-10. Total Due: $1,250.00.",
                data_classification=DataClassification.SENSITIVE_CATEGORY,
            )
            rag_engine.ingest_document(
                document_id="doc_inv_nov",
                source_uri="file:///invoices/nov_2026.pdf",
                content="Vendor: Delta Supplies. Invoice #INV-2026-11. Total Due: $620.00.",
                data_classification=DataClassification.SENSITIVE_CATEGORY,
            )

            # 3. Connect `rag.retrieve` and `gmail.send_email` to the Tool Room Operator!
            sent_emails: list[dict[str, object]] = []
            executor = InMemoryMockToolExecutor(
                tool_handlers={
                    "rag.retrieve": lambda args: rag_engine.retrieve(
                        query=str(args.get("query", "")),
                        top_k=int(args.get("top_k", 2)),
                    ).format_observation_summary(),
                    "gmail.send_email": lambda args: (
                        sent_emails.append(dict(args)) or f"Sent email to {args['to']}"
                    ),
                }
            )

            # 4. Planner looks up SEMANTIC memory before planning so it knows Sam's email address
            semantic_hits = memory_manager.query_memory(
                agent_role=AgentRole.PLANNER,
                partition=MemoryPartition.SEMANTIC,
                query="Sam accounting email",
                task_id="task_p3_01",
                session_id="sess_p3_01",
            )
            self.assertEqual(len(semantic_hits), 1)
            sam_fact = semantic_hits[0].record.content
            self.assertIn("sam@accounting.example.com", sam_fact)

            planner_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "plan",
                        "agent_role": "planner",
                        "node_id": "node_planner",
                        "thought": f"Using memory ({sam_fact}), retrieve #INV-2026-10 via RAG and email Sam.",
                        "goal_summary": "Retrieve Invoice #INV-2026-10 via Hybrid RAG and email Sam.",
                        "steps": [
                            {
                                "step_id": "step_rag_invoice",
                                "target_agent": "research",
                                "objective": "Search local RAG index for invoice #INV-2026-10 and cite chunk ID.",
                                "depends_on": [],
                                "estimated_sensitivity": "read_only",
                            },
                            {
                                "step_id": "step_email_sam",
                                "target_agent": "comms",
                                "objective": "Email the grounded invoice summary to sam@accounting.example.com.",
                                "depends_on": ["step_rag_invoice"],
                                "estimated_sensitivity": "irreversible_write",
                            },
                        ],
                    }
                ]
            )

            research_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "tool_call",
                        "agent_role": "research",
                        "node_id": "step_rag_invoice",
                        "thought": "Query the local Hybrid RAG index for #INV-2026-10.",
                        "tool_name": "rag.retrieve",
                        "arguments": {"query": "Invoice #INV-2026-10 total", "top_k": 1},
                        "declared_sensitivity": "read_only",
                        "rationale": "Retrieve exact chunk for invoice #INV-2026-10",
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "research",
                        "node_id": "step_rag_invoice",
                        "thought": "Retrieved chunk doc_inv_oct#chunk_0 showing Acme Corp Total Due $1,250.00.",
                        "content": "Vendor: Acme Corp | Invoice: #INV-2026-10 | Total Due: $1,250.00",
                        "cited_source_ids": ["doc_inv_oct#chunk_0"],
                        "confidence": 0.99,
                    },
                ]
            )

            comms_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "tool_call",
                        "agent_role": "comms",
                        "node_id": "step_email_sam",
                        "thought": "Send grounded summary to sam@accounting.example.com.",
                        "tool_name": "gmail.send_email",
                        "arguments": {
                            "to": "sam@accounting.example.com",
                            "subject": "Invoice #INV-2026-10 Summary",
                            "body": "Vendor: Acme Corp | Invoice: #INV-2026-10 | Total Due: $1,250.00",
                        },
                        "declared_sensitivity": "irreversible_write",
                        "rationale": "Email October invoice summary to Sam",
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "comms",
                        "node_id": "step_email_sam",
                        "thought": "Email sent after user approval.",
                        "content": "Sent #INV-2026-10 summary to sam@accounting.example.com.",
                        "cited_source_ids": ["doc_inv_oct#chunk_0"],
                        "confidence": 1.0,
                    },
                ]
            )

            orchestrator = NovaGraphOrchestrator(
                agents={
                    AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model, bus),
                    AgentRole.RESEARCH: BaseSpecializedAgent(AgentRole.RESEARCH, research_model, bus),
                    AgentRole.COMMS: BaseSpecializedAgent(AgentRole.COMMS, comms_model, bus),
                },
                checkpoint_store=SqliteCheckpointStore(tmp_path / "nova_checkpoints.db"),
                permission_gate=DefaultPolicyPermissionGate(),
                tool_executor=executor,
                event_bus=bus,
            )

            # Phase A: Start task -> Research runs rag.retrieve -> Comms pauses at Permission Gate
            paused_state = orchestrator.start_task(
                task_id="task_p3_01",
                session_id="sess_p3_01",
                user_goal="Find invoice #INV-2026-10 and email the total to Sam in accounting.",
                require_citations=True,
            )
            self.assertEqual(paused_state.status, TaskStatus.AWAITING_APPROVAL)

            # Verify Research node cited the exact RAG chunk ID (`doc_inv_oct#chunk_0`)!
            res_answer = paused_state.node_records["step_rag_invoice"].final_answer
            self.assertIsNotNone(res_answer)
            assert res_answer is not None
            self.assertEqual(res_answer.cited_source_ids, ("doc_inv_oct#chunk_0",))

            # Research saves a working note and promotes it to SEMANTIC memory
            working_note = memory_manager.write_memory(
                agent_role=AgentRole.RESEARCH,
                partition=MemoryPartition.WORKING,
                memory_key="invoice.2026_10.summary",
                content=res_answer.content,
                tags=("invoice", "acme", "2026-10"),
                importance=0.9,
                task_id="task_p3_01",
                session_id="sess_p3_01",
            )

            # Phase B: User approves the email -> Task completes!
            pending_email = paused_state.node_records["step_email_sam"].pending_tool_proposal
            assert pending_email is not None
            final_state = orchestrator.submit_user_approval(
                task_id="task_p3_01",
                node_id="step_email_sam",
                proposal_id=pending_email.proposal_id,
                approved=True,
                reason="Verified citation doc_inv_oct#chunk_0 and recipient Sam",
                require_citations=True,
            )
            self.assertEqual(final_state.status, TaskStatus.COMPLETED)
            self.assertEqual(len(sent_emails), 1)
            self.assertEqual(sent_emails[0]["to"], "sam@accounting.example.com")

            # Promote working note to SEMANTIC memory and record EPISODIC diary entry
            promoted = memory_manager.promote_working_to_long_term(
                session_id="sess_p3_01",
                task_id="task_p3_01",
                working_memory_id=working_note.memory_id,
                target_partition=MemoryPartition.SEMANTIC,
                promoting_agent=AgentRole.RESEARCH,
            )
            self.assertEqual(promoted.partition, MemoryPartition.SEMANTIC)

            episodic_entry = memory_manager.write_memory(
                agent_role=AgentRole.COMMS,
                partition=MemoryPartition.EPISODIC,
                content="Completed task_p3_01: Emailed #INV-2026-10 ($1,250.00) to sam@accounting.example.com.",
                tags=("completed_task", "invoice", "sam"),
                importance=0.85,
                task_id="task_p3_01",
                session_id="sess_p3_01",
            )
            self.assertEqual(episodic_entry.partition, MemoryPartition.EPISODIC)

    def test_reflection_gate_forces_research_agent_to_cite_rag_chunk_id(self) -> None:
        """Phase 9: When require_citations=True, an uncited FinalAnswerProposal is rejected by Reflection until cited."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bus = InMemoryEventBus()
            rag_engine = HybridRetrievalEngine(index=SqliteHybridChunkIndex(tmp_path / "rag_cite.db"))
            rag_engine.ingest_document(
                document_id="doc_oct_inv",
                source_uri="file:///invoices/oct_2026.pdf",
                content="Vendor: Acme Corp | Invoice #INV-2026-10 | Total Due: $1,250.00",
            )

            executor = InMemoryMockToolExecutor(
                tool_handlers={
                    "rag.retrieve": lambda args: rag_engine.retrieve(
                        str(args.get("query", ""))
                    ).format_observation_summary()
                }
            )

            planner_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "plan",
                        "agent_role": "planner",
                        "node_id": "node_planner",
                        "thought": "Single research step with mandatory citation.",
                        "goal_summary": "Find invoice #INV-2026-10 total with citation.",
                        "steps": [
                            {
                                "step_id": "step_research",
                                "target_agent": "research",
                                "objective": "Retrieve #INV-2026-10 and cite exact chunk ID.",
                                "depends_on": [],
                                "estimated_sensitivity": "read_only",
                            }
                        ],
                    }
                ]
            )

            # Research Agent:
            # Turn 1: Calls rag.retrieve
            # Turn 2 (Attempt 1): Returns FinalAnswerProposal WITHOUT cited_source_ids ([]) -> triggers Reflection!
            # Turn 3 (Attempt 2): Fixes FinalAnswerProposal to include cited_source_ids=["doc_oct_inv#chunk_0"]!
            research_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "tool_call",
                        "agent_role": "research",
                        "node_id": "step_research",
                        "thought": "Search RAG for #INV-2026-10.",
                        "tool_name": "rag.retrieve",
                        "arguments": {"query": "#INV-2026-10"},
                        "declared_sensitivity": "read_only",
                        "rationale": "Search local RAG index",
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "research",
                        "node_id": "step_research",
                        "thought": "Draft answer without citation.",
                        "content": "Total Due is $1,250.00.",
                        "cited_source_ids": [],  # Missing citation!
                        "confidence": 0.95,
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "research",
                        "node_id": "step_research",
                        "thought": "Added required citation ID doc_oct_inv#chunk_0 per critic feedback.",
                        "content": "Total Due is $1,250.00 (Source: doc_oct_inv#chunk_0).",
                        "cited_source_ids": ["doc_oct_inv#chunk_0"],
                        "confidence": 0.99,
                    },
                ]
            )

            critic_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "critique",
                        "agent_role": "planner",
                        "node_id": "step_research_reflect",
                        "thought": "Final answer is missing cited_source_ids for the retrieved RAG chunk.",
                        "target_proposal_id": "prop_uncited",
                        "verdict": "revise",
                        "issues_found": ["Missing cited_source_ids"],
                        "verification_questions_checked": ["Which specific source chunk IDs ground this answer?"],
                        "revision_instructions": "Include 'doc_oct_inv#chunk_0' in cited_source_ids.",
                    }
                ]
            )

            orchestrator = NovaGraphOrchestrator(
                agents={
                    AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model, bus),
                    AgentRole.RESEARCH: BaseSpecializedAgent(AgentRole.RESEARCH, research_model, bus),
                },
                checkpoint_store=SqliteCheckpointStore(tmp_path / "cite_checkpoints.db"),
                permission_gate=DefaultPolicyPermissionGate(),
                tool_executor=executor,
                event_bus=bus,
                critic_agent=BaseSpecializedAgent(AgentRole.PLANNER, critic_model, bus),
            )

            final_state = orchestrator.start_task(
                task_id="task_cite_enforced",
                session_id="sess_cite",
                user_goal="Find #INV-2026-10 total and cite the source chunk.",
                require_citations=True,
            )

            self.assertEqual(final_state.status, TaskStatus.COMPLETED)
            final_ans = final_state.node_records["step_research"].final_answer
            self.assertIsNotNone(final_ans)
            assert final_ans is not None
            self.assertEqual(final_ans.cited_source_ids, ("doc_oct_inv#chunk_0",))
            # Attempt 1 was revised by critic, Attempt 2 succeeded (attempt_count == 2)
            self.assertEqual(final_state.node_records["step_research"].attempt_count, 2)


if __name__ == "__main__":
    unittest.main()