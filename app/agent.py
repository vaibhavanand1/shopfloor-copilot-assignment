"""Agentic Reasoning Layer for ShopFloor Copilot.

Implements an explicit ReAct-style loop (Reason -> Act -> Observe) where each tool invocation
is tracked with timing, structured rationale, and shop-floor auditability.

Tools:
1. cv_ocr_inspect: Runs classical OpenCV defect classification and OCR text extraction (invoked when image is present).
2. check_work_order_status: Queries mock factory ERP/MES for part traveler metadata.
3. retrieve_sop_knowledge: Vector store lookup using Bandit-selected RAG top-K.
4. propose_escalation_or_scrap: Packages governance-compliant human checkpoint actions.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from app.audit_logger import AuditLogger
from app.rag_engine import OllamaClient, SOPVectorStore
from app.rl_bandit import ContextualBandit
from app.schemas import InspectResponse, PipelineConfig, ProposedAction, TraceStep
from app.vision_ocr import VisionOCREngine

# Mock ERP Database for Work Orders
MOCK_ERP_REGISTRY: Dict[str, Dict[str, Any]] = {
    "WO-1001": {
        "work_order_id": "WO-1001",
        "part_number": "PN-6061-BILLET-STD",
        "part_name": "Standard Machined Mounting Plate",
        "material": "6061-T6 Aluminum",
        "machine": "CNC-Mill-01",
        "batch_size": 500,
        "units_produced": 120,
        "batch_status": "NORMAL_PRODUCTION",
        "defect_history": "0 defects reported in current shift",
        "priority": "STANDARD",
    },
    "WO-1002": {
        "work_order_id": "WO-1002",
        "part_number": "PN-6061-FLANGE-B",
        "part_name": "Precision Hydraulic Flange",
        "material": "6061-T6 Aluminum",
        "machine": "CNC-Mill-04",
        "batch_size": 250,
        "units_produced": 45,
        "batch_status": "DEBURRING_PENDING",
        "defect_history": "Minor tool mark observed on previous batch",
        "priority": "HIGH",
    },
    "WO-1003": {
        "work_order_id": "WO-1003",
        "part_number": "PN-7075-SPINDLE-ARM",
        "part_name": "Critical Aerospace Bracket",
        "material": "7075-T651 Aerospace Alloy",
        "machine": "CNC-Mill-02",
        "batch_size": 80,
        "units_produced": 12,
        "batch_status": "HOLD_PENDING_QA",
        "defect_history": "Thermal drift alarm logged 30 min ago",
        "priority": "CRITICAL_CLASS_A",
    },
    "WO-1004": {
        "work_order_id": "WO-1004",
        "part_number": "PN-6061-BEVEL-BASE",
        "part_name": "Precision Bevel Base",
        "material": "6061-T6 Aluminum",
        "machine": "CNC-Mill-05",
        "batch_size": 300,
        "units_produced": 94,
        "batch_status": "DIMENSIONAL_AUDIT",
        "defect_history": "Tool wear offset flagged on Z-axis",
        "priority": "HIGH",
    },
}


class ShopFloorAgent:
    def __init__(
        self,
        vision_engine: Optional[VisionOCREngine] = None,
        vector_store: Optional[SOPVectorStore] = None,
        ollama_client: Optional[OllamaClient] = None,
        bandit: Optional[ContextualBandit] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.vision_engine = vision_engine or VisionOCREngine()
        self.vector_store = vector_store or SOPVectorStore()
        self.ollama_client = ollama_client or OllamaClient()
        self.bandit = bandit or ContextualBandit()
        self.audit_logger = audit_logger or AuditLogger()

    def check_work_order_status(self, work_order_id: str) -> Dict[str, Any]:
        """Mock tool: Queries ERP traveler database for the given Work Order ID."""
        clean_id = work_order_id.strip().upper()
        if clean_id in MOCK_ERP_REGISTRY:
            return MOCK_ERP_REGISTRY[clean_id]
        return {
            "work_order_id": clean_id,
            "part_number": "PN-CUSTOM-FAB",
            "part_name": "General CNC Precision Component",
            "material": "6061-T6 Aluminum",
            "machine": "CNC-Mill-General",
            "batch_size": 100,
            "units_produced": 1,
            "batch_status": "ACTIVE_RUN",
            "defect_history": "No prior records for this lot",
            "priority": "STANDARD",
        }

    def run_inspection(
        self,
        query: str,
        image_input: Optional[Union[str, bytes]] = None,
        work_order_id: Optional[str] = None,
    ) -> InspectResponse:
        """Executes the full ReAct agent inspection cycle and returns structured response."""
        pipeline_start = time.time()
        transaction_id = str(uuid.uuid4())
        trace: List[TraceStep] = []
        step_num = 1

        # State initialization
        defect_category = "no_image_supplied"
        extracted_ocr_text = None
        ocr_engine_name = "none"
        cv_metrics: Dict[str, Any] = {}

        # ---------------------------------------------------------
        # TOOL 1: CV / OCR Inspection (Conditional on image presence)
        # ---------------------------------------------------------
        if image_input is not None:
            t0 = time.time()
            thought = (
                "Operator uploaded a part image. Executing classical OpenCV defect classification "
                "(contour solidity + Canny edge analysis) and extracting stamped Work Order ID via OCR."
            )
            cv_res = self.vision_engine.inspect(image_input)
            t1 = time.time()

            defect_category = cv_res["defect_category"]
            extracted_ocr_text = cv_res.get("extracted_ocr_text")
            ocr_engine_name = cv_res.get("ocr_engine", "pytesseract")
            cv_metrics = cv_res.get("metrics", {})

            trace.append(TraceStep(
                step_number=step_num,
                thought=thought,
                action_tool="cv_ocr_inspect",
                tool_input={"image_provided": True, "ocr_engine": ocr_engine_name},
                observation={
                    "defect_category": defect_category,
                    "confidence": cv_res["confidence"],
                    "extracted_ocr_text": extracted_ocr_text,
                    "metrics": cv_metrics,
                },
                latency_ms=round((t1 - t0) * 1000, 2),
            ))
            step_num += 1

        # Determine effective Work Order ID (Request input takes precedence, fallback to OCR)
        active_wo_id = work_order_id or extracted_ocr_text

        # ---------------------------------------------------------
        # TOOL 2: ERP Work Order Status Lookup
        # ---------------------------------------------------------
        wo_status = None
        if active_wo_id:
            t0 = time.time()
            thought = (
                f"Work Order identifier '{active_wo_id}' identified from "
                f"{'explicit request input' if work_order_id else 'OCR stamped text'}. "
                "Querying factory ERP/MES traveler for active batch specs and machine telemetry."
            )
            wo_status = self.check_work_order_status(active_wo_id)
            t1 = time.time()

            trace.append(TraceStep(
                step_number=step_num,
                thought=thought,
                action_tool="check_work_order_status",
                tool_input={"work_order_id": active_wo_id},
                observation=wo_status,
                latency_ms=round((t1 - t0) * 1000, 2),
            ))
            step_num += 1

        # ---------------------------------------------------------
        # RL CONTEXTUAL BANDIT: Select RAG Top-K Action for this State
        # ---------------------------------------------------------
        chosen_top_k = self.bandit.select_action(state=defect_category, explore=True)

        # ---------------------------------------------------------
        # TOOL 3: Vector Store Knowledge Retrieval (RAG)
        # ---------------------------------------------------------
        t0 = time.time()
        thought = (
            f"Contextual Bandit selected action top_k={chosen_top_k} for state '{defect_category}'. "
            f"Querying TF-IDF SOP vector store for tolerance, rework, and escalation rules."
        )
        retrieved_chunks = self.vector_store.retrieve(
            query=query,
            defect_category=defect_category,
            top_k=chosen_top_k,
        )
        t1 = time.time()

        trace.append(TraceStep(
            step_number=step_num,
            thought=thought,
            action_tool="retrieve_sop_knowledge",
            tool_input={
                "query": query,
                "defect_category": defect_category,
                "top_k": chosen_top_k,
            },
            observation={
                "retrieved_chunk_count": len(retrieved_chunks),
                "chunks": [
                    {"doc": c["doc_name"], "title": c["title"], "score": c["score"]}
                    for c in retrieved_chunks
                ],
            },
            latency_ms=round((t1 - t0) * 1000, 2),
        ))
        step_num += 1

        # ---------------------------------------------------------
        # TOOL 4: Human Checkpoint / Proposed Action Gate
        # ---------------------------------------------------------
        proposed_action_obj: Optional[ProposedAction] = None
        lower_query = query.lower()
        needs_action = (
            defect_category in ["crack", "dimensional_deviation"]
            or "scrap" in lower_query
            or "escalat" in lower_query
            or "quarantine" in lower_query
        )

        if needs_action:
            t0 = time.time()
            thought = (
                f"Defect category '{defect_category}' or query intent indicates critical scrap or escalation. "
                "Factory governance SOP-OPS-04 mandates human sign-off before ERP write. Creating proposed action."
            )

            if defect_category == "crack" or "crack" in lower_query:
                action_type = "SCRAP_PART"
                reason = "Cracks violate 0.0mm zero-tolerance limit (SOP-ENG-01). Proposing terminal scrap and quarantine."
                approver = "Quality Lead Marcus Vance (ext. 4402)"
            elif defect_category == "dimensional_deviation":
                action_type = "ESCALATE_TO_SUPERVISOR"
                reason = "Dimensional variance > 0.2mm exceeds nominal CAD tolerance. Requires spindle check & scrap review."
                approver = "Maintenance Lead Priya Nair (ext. 4190) & Quality Lead Marcus Vance (ext. 4402)"
            elif defect_category == "surface_scratch":
                action_type = "REWORK_STATION_B4"
                reason = "Superficial scratch (< 0.5mm) salvageable via deburring at Station B-4 (SOP-ENG-02)."
                approver = "Shift Operations Supervisor Elena Rostova (ext. 4105)"
            else:
                action_type = "ESCALATE_TO_SUPERVISOR"
                reason = "Escalation requested by operator. Pending supervisor review."
                approver = "Shift Operations Supervisor Elena Rostova (ext. 4105)"

            proposed_action_obj = ProposedAction(
                status="PROPOSED_PENDING_HUMAN_APPROVAL",
                action_type=action_type,
                requires_human_approval=True,
                reason=reason,
                designated_approver=approver,
            )
            t1 = time.time()

            trace.append(TraceStep(
                step_number=step_num,
                thought=thought,
                action_tool="propose_escalation_or_scrap",
                tool_input={"defect_category": defect_category, "action_type": action_type},
                observation=proposed_action_obj.model_dump(),
                latency_ms=round((t1 - t0) * 1000, 2),
            ))
            step_num += 1

        # ---------------------------------------------------------
        # FINAL SYNTHESIS: Ollama Grounded Generation + Latency Ceiling
        # ---------------------------------------------------------
        t0 = time.time()
        final_answer, fallback_triggered, model_used = self.ollama_client.generate_grounded_answer(
            query=query,
            defect_category=defect_category,
            work_order_status=wo_status,
            retrieved_chunks=retrieved_chunks,
            proposed_action=proposed_action_obj.model_dump() if proposed_action_obj else None,
        )
        t1 = time.time()

        synthesis_thought = (
            f"Synthesizing final shop-floor guidance via {model_used}. "
            f"{'LLM latency ceiling exceeded; triggered deterministic SOP fallback.' if fallback_triggered else 'Grounded generation completed successfully.'}"
        )
        trace.append(TraceStep(
            step_number=step_num,
            thought=synthesis_thought,
            action_tool="synthesize_final_guidance",
            tool_input={"llm_model": model_used, "top_k": chosen_top_k},
            observation={
                "answer_length_chars": len(final_answer),
                "fallback_triggered": fallback_triggered,
                "synthesis_latency_ms": round((t1 - t0) * 1000, 2),
            },
            latency_ms=round((t1 - t0) * 1000, 2),
        ))

        # Overall execution latency
        total_latency_seconds = round(time.time() - pipeline_start, 4)

        pipeline_config = PipelineConfig(
            ocr_engine=ocr_engine_name,
            llm_model=model_used,
            rag_top_k=chosen_top_k,
            fallback_triggered=fallback_triggered,
        )

        # ---------------------------------------------------------
        # AUDIT LOGGING (Append-only immutable record)
        # ---------------------------------------------------------
        self.audit_logger.log_inspection(
            transaction_id=transaction_id,
            inputs={
                "query": query,
                "image_provided": image_input is not None,
                "work_order_id": work_order_id,
            },
            defect_category=defect_category,
            extracted_ocr_text=extracted_ocr_text,
            work_order_status=wo_status,
            pipeline_config=pipeline_config.model_dump(),
            decision_trace=[t.model_dump() for t in trace],
            proposed_action=proposed_action_obj.model_dump() if proposed_action_obj else None,
            final_answer=final_answer,
            execution_latency_seconds=total_latency_seconds,
        )

        return InspectResponse(
            transaction_id=transaction_id,
            final_answer=final_answer,
            defect_category=defect_category,
            extracted_ocr_text=extracted_ocr_text,
            work_order_status=wo_status,
            pipeline_config=pipeline_config,
            decision_trace=trace,
            proposed_action=proposed_action_obj,
            execution_latency_seconds=total_latency_seconds,
        )
