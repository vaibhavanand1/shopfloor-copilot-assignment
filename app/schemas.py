"""Pydantic schemas for the ShopFloor Copilot API, agent decision traces, and audit logging."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    """Represents a single step in the ReAct agent's decision loop."""
    step_number: int
    thought: str
    action_tool: str
    tool_input: Dict[str, Any]
    observation: Any
    latency_ms: float


class PipelineConfig(BaseModel):
    """Configuration used for the inspection pipeline run."""
    ocr_engine: str
    llm_model: str
    rag_top_k: int
    fallback_triggered: bool = False


class ProposedAction(BaseModel):
    """Structured human-checkpoint action object."""
    status: str = "PROPOSED_PENDING_HUMAN_APPROVAL"
    action_type: str  # SCRAP_PART | ESCALATE_TO_SUPERVISOR | REWORK_STATION_B4 | PASS_INSPECTION
    requires_human_approval: bool = True
    reason: str
    designated_approver: Optional[str] = None


class InspectJSONRequest(BaseModel):
    """JSON payload for /inspect endpoint."""
    query: str
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    work_order_id: Optional[str] = None


class InspectResponse(BaseModel):
    """Response returned by the /inspect endpoint."""
    transaction_id: str
    final_answer: str
    defect_category: str  # no_defect, surface_scratch, crack, dimensional_deviation, no_image_supplied
    extracted_ocr_text: Optional[str] = None
    work_order_status: Optional[Dict[str, Any]] = None
    pipeline_config: PipelineConfig
    decision_trace: List[TraceStep]
    proposed_action: Optional[ProposedAction] = None
    execution_latency_seconds: float


class FeedbackRequest(BaseModel):
    """Payload for submitting operator feedback."""
    transaction_id: str
    feedback_score: int = Field(..., ge=0, le=1, description="Binary score: 1 for helpful, 0 for unhelpful")


class FeedbackResponse(BaseModel):
    """Response returned by the /feedback endpoint."""
    status: str = "success"
    transaction_id: str
    state: str
    action: int
    feedback_score: int
    computed_reward: float
    updated_q_values: Dict[str, float]
    action_counts: Dict[str, int]


class TransactionRecord(BaseModel):
    """Structure of an audit log entry in data/audit_log.jsonl."""
    timestamp: str
    transaction_id: str
    inputs: Dict[str, Any]
    defect_category: str
    extracted_ocr_text: Optional[str] = None
    work_order_status: Optional[Dict[str, Any]] = None
    pipeline_config: PipelineConfig
    decision_trace: List[TraceStep]
    proposed_action: Optional[ProposedAction] = None
    final_answer: str
    execution_latency_seconds: float
    feedback: Optional[Dict[str, Any]] = None
