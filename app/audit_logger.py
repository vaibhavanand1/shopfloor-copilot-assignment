"""Append-only JSONL audit logger for ShopFloor Copilot transactions and human feedback."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

AUDIT_LOG_FILE = Path(__file__).resolve().parent.parent / "data" / "audit_log.jsonl"


class AuditLogger:
    def __init__(self, log_file: Path = AUDIT_LOG_FILE):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self.log_file.touch()

    def log_inspection(
        self,
        transaction_id: str,
        inputs: Dict[str, Any],
        defect_category: str,
        extracted_ocr_text: Optional[str],
        work_order_status: Optional[Dict[str, Any]],
        pipeline_config: Dict[str, Any],
        decision_trace: List[Dict[str, Any]],
        proposed_action: Optional[Dict[str, Any]],
        final_answer: str,
        execution_latency_seconds: float,
    ) -> Dict[str, Any]:
        """Appends an immutable audit log record for an /inspect transaction."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transaction_id": transaction_id,
            "inputs": inputs,
            "defect_category": defect_category,
            "extracted_ocr_text": extracted_ocr_text,
            "work_order_status": work_order_status,
            "pipeline_config": pipeline_config,
            "decision_trace": decision_trace,
            "proposed_action": proposed_action,
            "final_answer": final_answer,
            "execution_latency_seconds": round(execution_latency_seconds, 4),
            "feedback": None,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def log_feedback(
        self,
        transaction_id: str,
        feedback_score: int,
        computed_reward: float,
    ) -> Optional[Dict[str, Any]]:
        """Appends a feedback event line and updates the in-memory/latest state of the transaction."""
        feedback_event = {
            "event_type": "FEEDBACK_SUBMITTED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transaction_id": transaction_id,
            "feedback_score": feedback_score,
            "computed_reward": round(computed_reward, 4),
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_event) + "\n")

        return feedback_event

    def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a past transaction by its ID, merging any feedback submitted."""
        if not self.log_file.exists():
            return None

        found_record = None
        feedback_info = None

        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("transaction_id") == transaction_id:
                        if entry.get("event_type") == "FEEDBACK_SUBMITTED":
                            feedback_info = entry
                        else:
                            found_record = entry
                except Exception:
                    continue

        if found_record:
            if feedback_info:
                found_record["feedback"] = {
                    "score": feedback_info.get("feedback_score"),
                    "reward": feedback_info.get("computed_reward"),
                    "feedback_timestamp": feedback_info.get("timestamp"),
                }
            return found_record

        return None
