# ShopFloor Copilot — Master Technical Code Walkthrough & Interview Guide
**Role:** Forward-Deployed Engineer (FDE) — Manufacturing Vertical  
**Target:** Senior Engineering Technical Interview Deep-Dive  
**Author:** AI Technical Mentor  

---

## Architecture at a Glance

```mermaid
flowchart TD
    Client([Operator / Test Harness]) -->|POST /inspect (Image, Query, WO-ID)| API[app/main.py: inspect_part]
    API --> Agent[app/agent.py: ShopFloorAgent.run_inspection]
    
    subgraph Agent Loop [ReAct Explicit Decision Trace]
        Agent -->|1. If Image Provided| Vision[app/vision_ocr.py: VisionOCREngine.inspect]
        Vision -->|Defect Category, OCR WO-ID| Agent
        
        Agent -->|2. If WO-ID Found| ERP[app/agent.py: check_work_order_status]
        ERP -->|Batch Specs & Spindle Status| Agent
        
        Agent -->|3. Get Policy Action| Bandit[app/rl_bandit.py: ContextualBandit.select_action]
        Bandit -->|top_k in 1, 3, 5| Agent
        
        Agent -->|4. Retrieve SOPs| VectorStore[app/rag_engine.py: SOPVectorStore.retrieve]
        VectorStore -->|Ranked Chunks| Agent
        
        Agent -->|5. If Scrap or Escalation| Gate[app/agent.py: propose_escalation_or_scrap]
        Gate -->|Status: PROPOSED_PENDING_HUMAN_APPROVAL| Agent
        
        Agent -->|6. Grounded Answer| LLM[app/rag_engine.py: OllamaClient.generate_grounded_answer]
    end
    
    Agent --> Audit[app/audit_logger.py: AuditLogger.log_inspection]
    Audit --> DiskLog[(data/audit_log.jsonl)]
    Agent --> API
    API --> Client
    
    Client -->|POST /feedback (tx_id, score)| FBAPI[app/main.py: submit_feedback]
    FBAPI --> AuditGet[app/audit_logger.py: get_transaction]
    FBAPI --> BanditUp[app/rl_bandit.py: ContextualBandit.update]
    BanditUp --> DiskBandit[(data/bandit_state.json)]
```

---

# MODULE 1: Vision & OCR (`app/vision_ocr.py` & `scripts/generate_mock_data.py`)

### 1.1 Synthetic Image Generation (`scripts/generate_mock_data.py`)
To test visual defect classification without proprietary customer images, `scripts/generate_mock_data.py` synthesizes 4 realistic CNC billet images ($500 \times 500$ RGB) with stamped Work Order text:
- `part_clean.png` (lines 73–80): Nominal machined rectangle, uniform metallic gray `(198, 204, 212)`, 4 mounting holes, stamped `WO-1001`.
- `part_scratch.png` (lines 82–95): Nominal outer billet with a thin, linear 2-pixel scratch across `(140, 275)` to `(340, 235)`, stamped `WO-1002`.
- `part_crack.png` (lines 97–121): Nominal outer billet with a thick (4px) jagged branching polyline fissure with multiple branching vertices, stamped `WO-1003`.
- `part_deviation.png` (lines 123–130): Outer contour itself is clipped diagonally from `(290, 80)` to `(420, 210)`, dropping area by ~8% and simulating a sheared flange, stamped `WO-1004`.

### 1.2 Image Ingestion & Normalization (`app/vision_ocr.py`: lines 44–80)
```python
def load_image(self, image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
```
- **Inputs:** Supports raw `bytes` (from FastAPI `UploadFile`), base64 encoded strings, file path strings, or direct NumPy arrays.
- **Bytes decoding (lines 49–54):** Converts bytes to `np.frombuffer(image_input, np.uint8)` and decodes via `cv2.imdecode(nparr, cv2.IMREAD_COLOR)`. This avoids saving temporary files to disk, eliminating I/O bottlenecks.
- **Base64 decoding (lines 56–61, 70–78):** Handles data URI prefixes (`data:image/png;base64,...`) and decodes using Python's native `base64.b64decode`.

### 1.3 Mathematical Defect Classification (`app/vision_ocr.py`: lines 82–159)
The function `classify_defect(bgr_img)` operates in two sequential stages:

#### Stage 1: Dimensional Envelope Analysis (Lines 91–118)
```python
gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
main_contour = max(contours, key=cv2.contourArea)
contour_area = cv2.contourArea(main_contour)
bx, by, bw, bh = cv2.boundingRect(main_contour)
bbox_area = float(bw * bh) if (bw * bh) > 0 else 1.0
solidity = contour_area / bbox_area
aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0
```
- **Heuristic:** **Solidity** is defined as $\frac{\text{Contour Area}}{\text{Bounding Box Area}}$.
  - A nominal square machined billet has $\text{Solidity} \approx 0.98 - 0.995$.
  - When a corner is sheared or chipped off (as in `part_deviation.png`), $\text{Solidity}$ drops to $0.922$.
- **Decision Boundary (Line 111):**
  If `solidity < 0.96` or `aspect_ratio < 0.85` or `aspect_ratio > 1.15`, the algorithm immediately flags `dimensional_deviation` with confidence `0.95`.

#### Stage 2: Central Machining ROI Surface Inspection (Lines 120–158)
If outer dimensions are nominal, the algorithm inspects the interior face.
```python
# Crop Central ROI avoiding corner mounting bores and stamping pad (Lines 123-128)
roi_y1 = int(by + bh * 0.28)
roi_y2 = int(by + bh * 0.82)
roi_x1 = int(bx + bw * 0.15)
roi_x2 = int(bx + bw * 0.85)
roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]

# Edge Detection (Lines 133-139)
blurred_roi = cv2.GaussianBlur(roi, (3, 3), 0)
edges = cv2.Canny(blurred_roi, 50, 150)
edge_pixels = cv2.countNonZero(edges)
```
- **Mathematical Heuristics:**
  - `edge_pixels < 50`: Machined surface is clean $\rightarrow$ `no_defect` (Confidence 0.98).
  - `50 <= edge_pixels < 450`: High-aspect-ratio linear defect $\rightarrow$ `surface_scratch` (Confidence 0.92).
  - `edge_pixels >= 450`: High edge density with branching contours $\rightarrow$ `crack` (Confidence 0.96).

### 1.4 Work Order OCR Extraction & Fallbacks (Lines 160–230)
```python
def extract_work_order(self, bgr_img: np.ndarray, engine: Optional[str] = None) -> Tuple[Optional[str], str]:
```
1. **Targeted ROI Cropping (lines 170–173):** Extracts the upper stamping pad ($y \in [18\%, 35\%]$, $x \in [28\%, 72\%]$). Cropping reduces character confusion and cuts OCR runtime by over $70\%$.
2. **Pytesseract Execution (lines 178–196):**
   - Uses Page Segmentation Mode 6 (`--psm 6`: assume a single uniform block of text).
   - Applies regex `re.search(r"WO-\d+", text)`.
   - If ROI crop misses, runs full image scan with `--psm 11` (sparse text).
3. **Graceful Fallback:** If `tesseract` binary is missing or errors out, catches exceptions and cascades to `EasyOCR` (`_get_easyocr_reader()`) or contour character parsing, ensuring the service never crashes on a production line.

---

# MODULE 2: Vector Store & Local LLM RAG (`app/rag_engine.py`)

### 2.1 Chunking & In-Memory Vector Store (`SOPVectorStore`: lines 23–88)
- **Corpus (data/sops/):** 4 documents (`01_tolerance_specs.md`, `02_rework_vs_scrap_rules.md`, `03_escalation_contacts.md`, `04_erp_traveler_procedures.md`).
- **Chunking (lines 37–55):** Splits on markdown section boundaries (`re.split(r"\n(?=##\s+)", text)`), producing 15 distinct semantic units.
- **Indexing (lines 56–64):**
  ```python
  self.vectorizer = TfidfVectorizer(
      stop_words="english",
      ngram_range=(1, 2),
      sublinear_tf=True,
  )
  corpus = [c["content"] for c in self.chunks]
  self.doc_vectors = self.vectorizer.fit_transform(corpus)
  ```
- **Retrieval Math (lines 65–87):**
  1. Computes Query Vector: `query_vec = self.vectorizer.transform([search_text])`.
  2. Cosine Similarity: `sims = cosine_similarity(query_vec, self.doc_vectors)[0]`.
  3. Ranking: `ranked_indices = sims.argsort()[::-1][:top_k]`.
  4. Returns top-$k$ chunks with individual relevance scores. Runs in $<2\text{ms}$ entirely in RAM without external network calls.

### 2.2 Ollama Local Client & Latency Ceiling (`OllamaClient`: lines 90–237)
- **Auto-Detection (lines 96–118):** Queries `GET /api/tags` on `http://localhost:11434`, prioritizing fast local instruction models: `llama3.2:latest`, `llama3.1:latest`, `qwen2.5:latest`, `mistral:latest`.
- **Latency Ceiling Fallback (lines 160–185):**
  ```python
  LLM_LATENCY_CEILING_SECONDS = 3.5
  with httpx.Client(timeout=self.latency_ceiling) as client:
      res = client.post(f"{self.base_url}/api/generate", json={...})
  ```
  If Ollama takes longer than $3.5\text{s}$ or fails, the call times out and triggers `generate_deterministic_fallback` (lines 187–237):
  - Formats grounded disposition rules, exact tolerance thresholds, and contact numbers directly from the retrieved SOP chunks.
  - Sets `fallback_triggered = True` in `pipeline_config`.

---

# MODULE 3: Contextual Bandit RL Loop (`app/rl_bandit.py`)

### 3.1 State, Action, and State Representation (Lines 28–40)
- **State Space (5 states):** `no_defect`, `surface_scratch`, `crack`, `dimensional_deviation`, `no_image_supplied`.
- **Action Space (3 actions):** `top_k \in [1, 3, 5]`.
- **Persisted State (`data/bandit_state.json`):**
  ```json
  {
    "q_table": {
      "no_defect": {"1": 8.98, "3": 7.46, "5": 7.39},
      "crack": {"1": 0.0, "3": 6.57, "5": 0.0}
    },
    "counts": {
      "no_defect": {"1": 7, "3": 2, "5": 1},
      "crack": {"1": 0, "3": 7, "5": 0}
    }
  }
  ```

### 3.2 Action Selection: Epsilon-Greedy Policy (Lines 99–114)
```python
def select_action(self, state: str, explore: bool = True) -> int:
    if explore and random.random() < self.epsilon:
        return random.choice(self.actions) # Exploration (20%)
    
    action_q = self.q_table[state]
    max_q = max(action_q.values())
    best_actions = [int(a_str) for a_str, q in action_q.items() if q == max_q]
    return random.choice(best_actions)     # Exploitation (80%)
```

### 3.3 Incremental Sample-Average Update Rule (Lines 115–138)
```python
def update(self, state: str, action: int, reward: float):
    action_key = str(action)
    current_n = self.counts[state][action_key] + 1
    self.counts[state][action_key] = current_n

    old_q = self.q_table[state][action_key]
    new_q = old_q + (1.0 / current_n) * (reward - old_q)
    self.q_table[state][action_key] = round(new_q, 4)
    self.save()
```
- **Mathematical Form:** $Q_{k+1} = Q_k + \frac{1}{N} (R - Q_k)$.
- **Convergence Dynamics:**
  - `no_defect`: `top_k=1` satisfies tolerance rule in 0.9s, giving `Reward = (1 * 10) - 0.9 = 9.1`. `top_k=5` takes 2.6s (`Reward = 7.4`). The bandit shifts to `top_k=1`.
  - `crack`: `top_k=1` only retrieves tolerance limits and misses escalation contact (Marcus Vance ext. 4402), earning `feedback_score = 0` (`Reward = -1.5`). `top_k=3` retrieves both chunks, earning `feedback_score = 1` (`Reward = 6.6`). The bandit shifts to `top_k=3`.

---

# MODULE 4: The ReAct Agent & Guardrails (`app/agent.py`)

### 4.1 Step Construction & Structured Decision Trace (Lines 109–351)
Every tool invocation appends an auditable step to `decision_trace`:
```python
trace.append(TraceStep(
    step_number=step_num,
    thought="Operator uploaded part image. Executing classical OpenCV defect classification...",
    action_tool="cv_ocr_inspect",
    tool_input={"image_provided": True, "ocr_engine": ocr_engine_name},
    observation={...},
    latency_ms=round((t1 - t0) * 1000, 2)
))
```

### 4.2 Tool Sequencing Pipeline
1. **Tool 1 (`cv_ocr_inspect` - lines 130–157):** Executes only when image is present.
2. **Tool 2 (`check_work_order_status` - lines 163–187):** Queries mock ERP traveler database. Triggered if `work_order_id` is supplied in the request OR extracted via OCR.
3. **Contextual Bandit Policy Call (lines 191–193):** Picks $top\_k$ based on detected `defect_category`.
4. **Tool 3 (`retrieve_sop_knowledge` - lines 197–234):** Queries TF-IDF vector store with selected $top\_k$.
5. **Tool 4 (`propose_escalation_or_scrap` - lines 238–290):** Factory Governance Checkpoint. If defect is `crack` or `dimensional_deviation`, creates:
   ```json
   {
     "status": "PROPOSED_PENDING_HUMAN_APPROVAL",
     "action_type": "SCRAP_PART",
     "requires_human_approval": true,
     "designated_approver": "Quality Lead Marcus Vance (ext. 4402)"
   }
   ```
6. **Final Synthesis Tool (lines 294–323):** Invokes Ollama or deterministic fallback and records execution metrics.

---

# MODULE 5: FastAPI Endpoints, Schemas & Audit Logging (`app/main.py`, `app/schemas.py`, `app/audit_logger.py`)

### 5.1 Multi-Modal `POST /inspect` (`app/main.py`: lines 100–149)
- Handles both `multipart/form-data` and `application/json` dynamically by inspecting `request.headers.get("content-type")`.
- Returns `InspectResponse` containing `transaction_id`, `defect_category`, `extracted_ocr_text`, `pipeline_config`, `decision_trace`, and `proposed_action`.

### 5.2 Reinforcement Feedback Loop `POST /feedback` (`app/main.py`: lines 151–189)
1. Looks up `transaction_id` in `data/audit_log.jsonl`.
2. Extracts state (`defect_category`), action (`rag_top_k`), and `execution_latency_seconds`.
3. Computes `reward = (feedback_score * 10.0) - latency`.
4. Invokes `bandit.update(state, action, reward)`.
5. Logs feedback event to audit log and returns updated Q-values.

### 5.3 Audit Logger (`app/audit_logger.py`)
- **Append-only Guarantee (lines 53–55):** Uses `"a"` mode on `data/audit_log.jsonl` with atomic line writes.
- **Transaction Replay (`GET /transactions/{id}`):** Parses historical records and merges subsequent feedback events to replay past decision states for regulatory QA audits.

---

# MODULE 6: Simulation Script & FDE Tradeoffs (`scripts/simulate_rl_shift.py` & `DECISIONS.md`)

### 6.1 Simulation Engine (`scripts/simulate_rl_shift.py`)
1. Runs functional health, JSON, multipart, feedback, and audit replay tests.
2. Resets bandit state to $0.0$.
3. Simulates 25 interleaved worker requests:
   - 10 clean part queries $\rightarrow top\_k=1$ minimizes latency and wins.
   - 8 surface scratch queries $\rightarrow top\_k=1$ answers Station B-4 routing and wins.
   - 7 crack queries $\rightarrow top\_k=1$ fails to retrieve escalation contacts (score=0 penalty); $top\_k=3$ captures both documents and converges as optimal.

### 6.2 Key FDE Interview Tradeoff Talking Points (`DECISIONS.md`)
1. **Vector RAG vs. 20-Page Long-Context Prompt:** In real plants, SOPs expand continuously with ECNs and shift logs. RAG allows sub-millisecond incremental indexing. If frozen at 20 pages, a KV-cached long-context prompt would eliminate chunking boundary issues.
2. **ReAct Agent vs. Hardcoded Pipeline:** Agents allow dynamic sequencing (OCR $\rightarrow$ ERP lookup only if ID found). The terminal scrap gate must **never** be agentic due to financial and safety risk of unverified scrap charges.
3. **Classical CV vs. Deep Learning:** Classical CV runs in $<15\text{ms}$ on CPU, requires no GPU/cloud APIs, and gives mathematically explainable metrics (solidity $<0.96$). Fine-tuning is justified only for complex non-rigid injection parts or subtle metallurgical inclusions.
4. **Scale from 1 to 10 Plants:** Model tiering (caching + compact SLMs for 70% routine queries, routing ambiguous cases to larger models) saves more cost than fine-tuning or quantization.
5. **Shop-Floor Risk:** Automation bias ("ghost scrapping"). Mitigated via **hardware-interlocked RFID/PLC badge gates** on quarantine bins.
