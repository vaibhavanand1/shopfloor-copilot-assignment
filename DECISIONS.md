# Engineering Decision Rationale
**Project:** ShopFloor Copilot — AI-Assisted Visual Inspection & SOP Agent  
**Domain:** Manufacturing (CNC Machining & Plastics Injection)

---

### System Architecture & Operational Trade-offs

#### 1. Why Vector RAG instead of feeding all SOPs into the LLM context?
When running in a real factory, SOPs are not static—teams update tooling notes, add change orders (ECNs), and tweak tolerance rules pretty frequently.
- **Why RAG works better here:** With RAG, when a new SOP document is added or edited, we just parse and index that new text file into our vector store in a few milliseconds. The LLM only receives the 1 to 3 paragraphs it actually needs to answer the question.
- **The problem with full context on edge hardware:** If we dumped 50+ pages of factory manuals into the prompt every time, a local edge model (like Llama 3.2 on an on-prem mini PC) would get very slow—taking 5 to 10 seconds just to start outputting tokens. Also, local models easily get confused or miss tiny numbers (like a 0.2mm tolerance limit) when buried inside a huge wall of text.
- **When would I change my mind?** If the plant had a small, fixed set of 15–20 pages that never changed, I'd probably skip the vector database entirely. We could just cache the prompt tokens once in memory and let the LLM see the whole manual. That would save us from having to chunk documents or worry about vector search missing related sentences across section boundaries.

#### 2. Why use an Agent Loop instead of a simple hardcoded script? And what should NEVER be agentic?
- **Why an agent makes sense here:** A fixed `if/else` script breaks down quickly because every inspection is different. Sometimes a worker only asks a text question ("what's the scrap code for Station B?"). Sometimes they upload a photo with no text. Sometimes the photo has a stamped Work Order ID, which means we should look up the traveler in the ERP first before checking the SOPs. An agent loop lets the system decide what tool to run based on what was provided (image -> OpenCV, Work Order ID -> ERP lookup, then RAG -> LLM).
- **The one part that must NEVER be agentic:** Actually marking a part as scrapped or changing batch status in the ERP. Scrapping precision aerospace or automotive parts costs real money and can halt an assembly line. An LLM should never have write access to trigger scrap transactions on its own. That's why our tool `propose_escalation_or_scrap` only creates a `PROPOSED_PENDING_HUMAN_APPROVAL` action. A real human supervisor has to review it and sign off before anything changes in the database.

#### 3. Why classical OpenCV instead of a deep learning model (like YOLO)?
- **Fast and runs on cheap hardware:** In a precision machining plant, parts sit under standard inspection lighting. Simple OpenCV operations (grayscale, Gaussian blur, Canny edge detection, and contour solidity) run in under 15ms directly on a basic CPU. We don't need an expensive GPU or have to manage gigabytes of model weights on every station.
- **Explainable math:** Quality engineers like numbers they can understand. If a corner is chipped off, the contour solidity drops below `0.96`. If there are cracks, the edge pixel count jumps above `450`. It's 100% deterministic and easy to debug.
- **When would I use a trained deep learning model?** If we were inspecting organic or curved injection-molded parts, parts under varying factory lighting, or subtle material defects like internal porosity or surface discoloration where basic edge detection fails. In those cases, fine-tuning something like YOLOv8-seg would definitely be worth the extra latency and compute.

#### 4. If scaling from 1 plant to 10 plants, what cost-saving lever would you use first?
I would use **model tiering (routing basic queries to smaller models or cached templates)** before doing anything complex like fine-tuning or quantization across 10 plants.
- Most questions asked by line workers are simple and repetitive—like "who is the shift lead?", "what's the tolerance for this part?", or "where does scrap go?".
- We can handle those straightforward queries using deterministic lookup templates or an ultra-lightweight 1B/3B model running locally with almost zero latency and cost.
- We only send the ambiguous, complicated cases (like multi-step crack investigations) to a larger model.
- This cuts server load and power costs right away without having to spend weeks collecting training data and fine-tuning models for each different facility.

#### 5. What is a specific shop-floor governance risk, and how would you fix it?
The biggest risk is **automation bias (or "ghost scrapping")**:
- An operator might blindly trust the AI's answer. If the screen says "PROPOSE SCRAP", they might just toss a $500 part into the physical scrap bin without getting the supervisor's physical review, leading to inventory discrepancies in the ERP. Or worse, if they take a blurry photo where a crack doesn't show up, they might pass a defective part into assembly.
- **How to fix this in production:**
  1. **Physical interlock on scrap bins:** Connect an electronic lock (via PLC/relay) to the red quarantine bin. The bin only opens when a supervisor scans their physical RFID badge to confirm the Copilot's proposed scrap action.
  2. **Fixed camera jig:** Instead of workers taking handheld photos with their phones from weird angles, mount a fixed camera with consistent lighting over the inspection table so image quality is always standard.

#### 6. Handling Concurrency and Race Conditions (Redis & PostgreSQL)
In this pilot prototype, we write directly to a local append-only file (`audit_log.jsonl`) and save bandit weights to `bandit_state.json`. In a production plant with multiple assembly stations running simultaneously, multiple workers could inspect parts or submit feedback at the exact same millisecond. That leads to concurrency issues and race conditions. Here is how we handle this for production:

1. **Race Conditions in Audit Logging (`audit_log.jsonl` -> PostgreSQL):**
   - *The problem:* Concurrent writes to a flat file can result in corrupted lines, file locks, or lost records. Also, reading through the whole file to find a `transaction_id` during feedback (`GET /transactions/{id}`) slows down as the file grows.
   - *Production fix:* Replace the JSONL file with **PostgreSQL**.
     - Each inspection inserts an immutable row with a unique `transaction_id` UUID primary key.
     - When an operator submits feedback (`POST /feedback`), we run a database transaction with row-level locking (`SELECT ... FOR UPDATE`) to update the feedback score and reward. This ensures that two simultaneous feedback submissions for the same transaction don't clash or overwrite each other.

2. **Race Conditions in RL Bandit Updates (`bandit_state.json` -> Redis):**
   - *The problem:* If two workers submit feedback at the same time for the same defect state (e.g., both submit feedback for `crack`), both requests would read the current Q-values and counts from disk, update them in Python memory, and write them back. One worker's update would silently overwrite the other (classic lost-update problem).
   - *Production fix:* Use **Redis** for bandit state management.
     - **Option A (Distributed Lock):** Acquire a short-lived Redis lock (`SET lock:bandit:state <token> NX EX 2`) before reading, calculating, and saving the updated Q-value, then release the lock.
     - **Option B (Atomic Redis Counters / Message Queue):** Keep running totals (`sum_rewards` and `visit_count`) inside a Redis Hash using atomic commands (`HINCRBYFLOAT` / `HINCRBY`). Or push feedback events to a Redis queue/stream (or Celery task) so a single worker process updates the Q-table sequentially without any lock contention.

3. **Race Conditions in Human Approval Gate (PostgreSQL):**
   - *The problem:* If two shift supervisors look at the same flagged work order and both click "Approve Scrap" at the same time, we could trigger duplicate scrap entries in the ERP.
   - *Production fix:* In PostgreSQL, use an atomic conditional update:
     ```sql
     UPDATE proposed_actions 
     SET status = 'APPROVED', approved_by = :supervisor_id, approved_at = NOW() 
     WHERE transaction_id = :tx_id AND status = 'PROPOSED_PENDING_HUMAN_APPROVAL';
     ```
     Only one query will update 1 row; the second query will return 0 rows affected, preventing duplicate ERP calls cleanly.

---

### Contextual Bandit Action-Space Rationale

#### Why optimize `rag_top_k` instead of switching the OCR engine?
When deciding what parameter the reinforcement learning bandit should tune, we looked at `rag_top_k` (how many document chunks to retrieve: 1, 3, or 5) versus switching between OCR engines (like Tesseract vs EasyOCR).
- **OCR is an on/off tool, not a dynamic trade-off:** In an edge factory setting, Tesseract is already installed locally, runs in ~15ms, and either reads the stamped "WO-100X" label or it doesn't. Switching OCR engines dynamically doesn't give a smooth trade-off curve—it just adds extra libraries.
- **`rag_top_k` is a real, measurable trade-off:**
  - Retrieving fewer chunks (`top_k=1`) means smaller prompts, faster LLM generation, and lower latency. For simple parts with no defects, 1 chunk has all the tolerance specs needed, so operators get their answer fast and leave positive feedback.
  - But for severe defects like `crack`, 1 chunk only gives the tolerance rule ("cracks have 0.0mm tolerance") and leaves out the supervisor contact details from the escalation SOP. The operator finds that unhelpful and gives a score of 0. Retrieving 3 chunks (`top_k=3`) includes both documents, giving a complete answer that earns a thumbs-up.
- **The result:** The bandit learns this balance automatically: it picks `top_k=1` for clean parts to save time, and `top_k=3` for cracks where complete information is worth the extra half-second.
