"""Retrieval-Augmented Generation (RAG) engine for ShopFloor Copilot.

Includes:
- In-memory vector store using TF-IDF n-grams + cosine similarity over SOP documents in data/sops/.
- Local Ollama client with auto-detection of models and a 3.5s latency ceiling fallback.
- Deterministic fallback synthesizer for guaranteed sub-millisecond local responses when LLM is unavailable or times out.
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SOPS_DIR = Path(__file__).resolve().parent.parent / "data" / "sops"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_LATENCY_CEILING_SECONDS = float(os.getenv("LLM_LATENCY_CEILING_SECONDS", "3.5"))


class SOPVectorStore:
    def __init__(self, sops_dir: Path = SOPS_DIR):
        self.sops_dir = sops_dir
        self.chunks: List[Dict[str, str]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors = None
        self._load_and_index()

    def _load_and_index(self):
        """Loads and indexes SOP markdown chunks."""
        self.chunks.clear()
        if not self.sops_dir.exists():
            return

        for filepath in sorted(self.sops_dir.glob("*.md")):
            text = filepath.read_text(encoding="utf-8")
            # Split sections by '## '
            raw_sections = re.split(r"\n(?=##\s+)", text)
            for section in raw_sections:
                section_str = section.strip()
                if not section_str or section_str.startswith("# SOP-"):
                    continue
                lines = section_str.split("\n", 1)
                title = lines[0].replace("##", "").strip()
                body = lines[1].strip() if len(lines) > 1 else ""
                combined = f"{title}. {body}"
                self.chunks.append({
                    "doc_name": filepath.name,
                    "title": title,
                    "content": combined,
                    "raw_body": body,
                })

        if self.chunks:
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            corpus = [c["content"] for c in self.chunks]
            self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, defect_category: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top_k relevant SOP chunks using cosine similarity."""
        if not self.chunks or self.vectorizer is None or self.doc_vectors is None:
            return []

        search_text = query
        if defect_category and defect_category != "no_image_supplied":
            clean_cat = defect_category.replace("_", " ")
            search_text = f"{search_text} {clean_cat}"

        query_vec = self.vectorizer.transform([search_text])
        sims = cosine_similarity(query_vec, self.doc_vectors)[0]

        top_k = min(top_k, len(self.chunks))
        ranked_indices = sims.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(round(sims[idx], 4))
            results.append(chunk)

        return results


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, latency_ceiling: float = LLM_LATENCY_CEILING_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.latency_ceiling = latency_ceiling
        self._cached_model = None

    def get_preferred_model(self) -> str:
        """Auto-detects available local Ollama models, preferring fast instruction models."""
        if self._cached_model:
            return self._cached_model

        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    models = [m.get("name", "") for m in res.json().get("models", [])]
                    for preferred in ["llama3.2:latest", "llama3.2", "llama3.1:latest", "llama3.1", "qwen2.5:latest", "mistral:latest"]:
                        if preferred in models:
                            self._cached_model = preferred
                            return preferred
                    if models:
                        self._cached_model = models[0]
                        return models[0]
        except Exception:
            pass

        self._cached_model = "llama3.2:latest"
        return self._cached_model

    def generate_grounded_answer(
        self,
        query: str,
        defect_category: str,
        work_order_status: Optional[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
        proposed_action: Optional[Dict[str, Any]],
    ) -> Tuple[str, bool, str]:
        """Generates grounded final answer via Ollama.

        Returns:
            Tuple of (final_answer, fallback_triggered, model_name)
        """
        model_name = self.get_preferred_model()
        context_blocks = "\n\n".join([f"[{c['doc_name']} - {c['title']}]: {c['content']}" for c in retrieved_chunks])

        prompt = f"""You are ShopFloor Copilot, an AI assistant for precision parts manufacturing (CNC machining & injection molding).
Synthesize a concise, clear, shop-floor actionable response for the line worker based STRICTLY on the retrieved SOP documentation and inspection evidence below.

INSPECTION FINDINGS:
- Defect Category: {defect_category}
- Work Order Status: {work_order_status.get('status') if work_order_status else 'N/A'} (Part: {work_order_status.get('part_number', 'N/A') if work_order_status else 'N/A'}, Machine: {work_order_status.get('machine', 'N/A') if work_order_status else 'N/A'})
- Proposed Action: {proposed_action.get('action_type') if proposed_action else 'None'} ({proposed_action.get('reason') if proposed_action else 'None'})

RELEVANT FACTORY SOPS:
{context_blocks}

OPERATOR QUESTION:
"{query}"

INSTRUCTIONS:
1. Directly state whether the component is acceptable, requires rework, or is proposed for scrap according to SOP tolerances.
2. If rework is permitted, specify the exact destination station and procedure.
3. If scrap or escalation is recommended, state that factory policy SOP-OPS-04 requires human supervisor sign-off before ERP write, and name the designated escalation contact and phone extension.
4. Keep the answer professional, crisp, and under 4 paragraphs.
"""

        start_time = time.time()
        try:
            with httpx.Client(timeout=self.latency_ceiling) as client:
                res = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9,
                            "num_predict": 250,
                        },
                    },
                )
                elapsed = time.time() - start_time
                if res.status_code == 200 and elapsed <= self.latency_ceiling:
                    ans = res.json().get("response", "").strip()
                    if ans:
                        return ans, False, model_name
        except Exception:
            pass  # Fall through to deterministic fallback

        # Fallback triggered due to latency ceiling or Ollama disconnect
        fallback_ans = self.generate_deterministic_fallback(
            query=query,
            defect_category=defect_category,
            work_order_status=work_order_status,
            retrieved_chunks=retrieved_chunks,
            proposed_action=proposed_action,
        )
        return fallback_ans, True, f"{model_name}-fallback"

    def generate_deterministic_fallback(
        self,
        query: str,
        defect_category: str,
        work_order_status: Optional[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
        proposed_action: Optional[Dict[str, Any]],
    ) -> str:
        """Deterministic, grounded synthesis generated directly from retrieved SOP chunks."""
        lines = []
        clean_cat = defect_category.replace("_", " ").title()
        lines.append(f"### Visual Inspection & SOP Assessment: {clean_cat}")

        # Part & Work Order details
        if work_order_status:
            wo_id = work_order_status.get("work_order_id", "Unknown")
            pn = work_order_status.get("part_number", "N/A")
            mach = work_order_status.get("machine", "N/A")
            lines.append(f"**Work Order:** {wo_id} | **Part Number:** {pn} | **Assigned Unit:** {mach}")

        # Core SOP Findings
        if defect_category == "no_defect":
            lines.append("- **Condition:** No surface anomalies or dimensional boundary deviations detected. Part meets nominal drawing CAD envelope.")
            lines.append("- **Disposition:** Acceptable for production advancement. Proceed to the next designated machining operation.")
        elif defect_category == "surface_scratch":
            lines.append("- **Tolerance Rule:** Per SOP-ENG-01, surface scratches under 0.5 mm depth on non-mating surfaces are salvageable.")
            lines.append("- **Disposition / Routing:** Route to **Station B-4 (Secondary Deburring & Polishing)** with a yellow traveler tag for Scotch-Brite buffing.")
        elif defect_category == "crack":
            lines.append("- **Tolerance Rule:** Per SOP-ENG-01 Section 3, structural cracks and fissures have an absolute **0.0 mm tolerance threshold**.")
            lines.append("- **Disposition:** Part cannot be salvaged. Immediate physical quarantine in the locked Red Quarantine Bin is required.")
        elif defect_category == "dimensional_deviation":
            lines.append("- **Tolerance Rule:** Per SOP-ENG-01 Section 4, dimensional variance exceeding 0.2 mm over nominal CAD violates geometric limits.")
            lines.append("- **Disposition:** Halt milling cycle on the active spindle and verify tool wear offsets.")

        # Escalation & Governance Checkpoint
        if proposed_action:
            lines.append(f"\n> **Governance Guardrail:** {proposed_action.get('reason')}")
            lines.append(f"> **Action Status:** `{proposed_action.get('status')}` (Requires sign-off by {proposed_action.get('designated_approver', 'Shift Supervisor')}).")
            lines.append("> Automated ERP scrap mutations are prohibited under SOP-OPS-04.")

        # Grounded SOP References
        if retrieved_chunks:
            lines.append("\n**Grounded SOP References:**")
            for chunk in retrieved_chunks[:3]:
                lines.append(f"- *{chunk['doc_name']} ({chunk['title']})*: {chunk['content'][:140]}...")

        return "\n".join(lines)
