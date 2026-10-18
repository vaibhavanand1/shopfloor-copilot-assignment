"""FastAPI Service for ShopFloor Copilot.

Endpoints:
- POST /inspect: Multi-modal part inspection accepting multipart/form-data or application/json.
- POST /feedback: Online RL reinforcement loop updating Contextual Bandit Q-table.
- GET /transactions/{id}: Replays audit log decision trace, inputs, config, and human checkpoint.
- GET /bandit/state: Displays active bandit Q-table and action counts across all states.
- GET /health: Health check and service telemetry.
"""

from contextlib import asynccontextmanager
import json
import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import ShopFloorAgent
from app.audit_logger import AuditLogger
from app.rag_engine import OllamaClient, SOPVectorStore
from app.rl_bandit import ContextualBandit
from app.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    InspectJSONRequest,
    InspectResponse,
    TransactionRecord,
)
from app.vision_ocr import VisionOCREngine

# Global application components
vision_engine = VisionOCREngine()
vector_store = SOPVectorStore()
ollama_client = OllamaClient()
bandit = ContextualBandit()
audit_logger = AuditLogger()
agent = ShopFloorAgent(
    vision_engine=vision_engine,
    vector_store=vector_store,
    ollama_client=ollama_client,
    bandit=bandit,
    audit_logger=audit_logger,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure SOP index is loaded and Ollama model detected
    vector_store._load_and_index()
    yield


app = FastAPI(
    title="ShopFloor Copilot",
    description="AI-Assisted Visual Inspection & SOP Agent for Precision Machining and Injection Molding",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check returning local LLM model and bandit state summary."""
    return {
        "status": "healthy",
        "service": "ShopFloor Copilot",
        "llm_model": ollama_client.get_preferred_model(),
        "tesseract_available": vision_engine.tesseract_available,
        "indexed_sop_chunks": len(vector_store.chunks),
    }


@app.get("/bandit/state")
def get_bandit_state():
    """Returns active Contextual Bandit Q-table and action counts."""
    return {
        "q_table": bandit.q_table,
        "counts": bandit.counts,
        "actions": bandit.actions,
        "epsilon": bandit.epsilon,
    }


@app.post("/bandit/reset")
def reset_bandit_state():
    """Resets the Contextual Bandit Q-table to 0.0 for simulation or test runs."""
    bandit.reset()
    return {"status": "reset", "q_table": bandit.q_table}


@app.post("/inspect", response_model=InspectResponse)
async def inspect_part(
    request: Request,
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    work_order_id: Optional[str] = Form(None),
):
    """Multi-modal inspection endpoint supporting both multipart/form-data and application/json."""
    content_type = request.headers.get("content-type", "")

    effective_query: Optional[str] = None
    image_payload: Optional[Any] = None
    effective_wo_id: Optional[str] = None

    if "application/json" in content_type:
        try:
            body_bytes = await request.body()
            body_json = json.loads(body_bytes.decode("utf-8"))
            parsed_req = InspectJSONRequest(**body_json)
            effective_query = parsed_req.query
            effective_wo_id = parsed_req.work_order_id
            if parsed_req.image_base64:
                image_payload = parsed_req.image_base64
            elif parsed_req.image_path:
                image_payload = parsed_req.image_path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON request body: {str(e)}")
    else:
        # Multipart / Form-Data
        effective_query = query
        effective_wo_id = work_order_id
        if image is not None:
            image_payload = await image.read()

    if not effective_query:
        raise HTTPException(
            status_code=422,
            detail="Field 'query' is required (either in multipart form or JSON body).",
        )

    try:
        response = agent.run_inspection(
            query=effective_query,
            image_input=image_payload,
            work_order_id=effective_wo_id,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inspection pipeline failure: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest):
    """Submits line worker binary feedback (1=helpful, 0=unhelpful) and updates Contextual Bandit."""
    transaction = audit_logger.get_transaction(payload.transaction_id)
    if not transaction:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction with ID '{payload.transaction_id}' not found in audit log.",
        )

    defect_cat = transaction.get("defect_category", "no_image_supplied")
    pipeline_cfg = transaction.get("pipeline_config", {})
    chosen_top_k = pipeline_cfg.get("rag_top_k", 3)
    latency_sec = float(transaction.get("execution_latency_seconds", 1.0))

    # Reward formula: (feedback_score * 10.0) - latency_in_seconds
    computed_reward = round((float(payload.feedback_score) * 10.0) - latency_sec, 4)

    # Live update of Epsilon-Greedy Contextual Bandit
    updated_q, updated_counts = bandit.update(
        state=defect_cat,
        action=chosen_top_k,
        reward=computed_reward,
    )

    # Log feedback event to audit log
    audit_logger.log_feedback(
        transaction_id=payload.transaction_id,
        feedback_score=payload.feedback_score,
        computed_reward=computed_reward,
    )

    return FeedbackResponse(
        status="success",
        transaction_id=payload.transaction_id,
        state=defect_cat,
        action=chosen_top_k,
        feedback_score=payload.feedback_score,
        computed_reward=computed_reward,
        updated_q_values=updated_q,
        action_counts=updated_counts,
    )


@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str):
    """Replays the complete past decision trace, configuration, and audit record."""
    transaction = audit_logger.get_transaction(transaction_id)
    if not transaction:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction with ID '{transaction_id}' not found.",
        )
    return transaction
