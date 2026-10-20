# Forward-Deployed Engineer (FDE) Decision Rationale
**Project:** ShopFloor Copilot — AI-Assisted Visual Inspection & SOP Agent  
**Domain:** Manufacturing (CNC Machining & Plastics Injection)

---

### Architectural & Operational Decision Rationale

#### 1. Why use Vector RAG instead of passing the whole SOP manual into the LLM context? What would change your answer if the plant's SOP set were fixed at 20 pages instead of continuously growing?
I chose RAG because factory SOPs are not static—they keep getting updated with new engineering revisions. Pasting 50+ pages into the prompt every time on local edge hardware would drastically increase latency and memory usage, and local models easily miss tiny numbers like a 0.2mm tolerance inside a huge wall of text. With RAG, we quickly update the vector store and fetch only the 1 to 3 relevant paragraphs.

I would choose long-context injection if we had a small, fixed set of 15 to 20 pages that never changed—then we could cache the static prompt tokens once in memory and consider the whole manual on every query.

#### 2. Why is an agent loop justified over a fixed pipeline for selecting tools? And what is the one part of this system you would NEVER make agentic, and why?
A simple if/else script breaks down quickly because every inspection request is different—sometimes a worker sends only text, sometimes an image without a Work Order ID where we have to run OCR first before calling the ERP. An agent loop simplifies this by dynamically running only the required tools.

However, directly updating high-risk defect status in the ERP should never be agentic, because any false positive impacts cost and halts production. That is why our Human Gate only creates a pending action for a supervisor to approve.

#### 3. Why did you choose classical Computer Vision (OpenCV) instead of a pretrained model for defect classification? What would push you to fine-tune a custom model instead?
First, OpenCV operations (contour solidity and Canny edge detection) are fast and run in under 15 milliseconds on a basic CPU, whereas pretrained models are not tailored to company-specific parts and require managing training data and GPU servers at each station. Also, OpenCV and OCR results are 100% deterministic and easy to debug.

I would switch to fine-tuning a deep learning model like YOLO to identify complex issues like curved parts, internal porosity, or surface discoloration where basic edge detection fails.

#### 4. If this pilot needed to scale from 1 plant to 10, which cost-optimization technique (caching, model tiering, batching, fine-tuning, or quantization) would you apply first, and why that one before the others?
To scale to ten plants, I would apply Model Tiering first—routing basic queries to a small 1B/3B model or template, and complex queries to a larger model. On the shop floor, most parts are good and most queries are repetitive. Routing only borderline scratch or crack cases to the large model cuts compute load immediately without needing fine-tuning for each plant.

#### 5. What is one shop-floor governance risk (specific to manufacturing, not a generic AI risk) that this design does not yet fully solve, and how would you fix it in a real deployment?
A major shop-floor risk is Automation Bias (or 'ghost scrapping')—blind trust in the AI's answer, which could lead to a worker throwing a part into the physical scrap bin without supervisor review, or taking a blurry handheld photo that misses a crack.

To close this in production, we would install fixed camera and lighting mounts to standardize image capture, and electronic locks on quarantine bins that only open when a supervisor scans their RFID badge.

#### 6. Handling Concurrency & Race Conditions in Production (Redis & PostgreSQL)
Right now, our prototype writes to local `.jsonl` and `.json` files, which would face file locks and lost-update race conditions if multiple stations wrote at the exact same millisecond. In production:
- We would replace `audit_log.jsonl` with **PostgreSQL**, using row-level locking (`SELECT ... FOR UPDATE`) for feedback updates and atomic conditional `UPDATE` statements so two supervisors cannot double-approve the same scrap action.
- And we would move `bandit_state.json` to **Redis**, using atomic hash counters (`HINCRBYFLOAT`) or distributed locks so concurrent feedback updates never overwrite each other.

---

### Contextual Bandit Action-Space Rationale

#### Why optimize RAG retrieval depth (`top_k`) instead of switching between OCR engines?
I chose to optimize `rag_top_k` (retrieving 1, 3, or 5 chunks) because it represents a direct, continuous trade-off between answer quality and token latency.

On edge hardware, Tesseract OCR is an on/off tool—it runs in ~15ms and either reads the stamped Work Order ID or it doesn't, so switching OCR engines does not offer a useful trade-off curve. In contrast, retrieving fewer chunks (`top_k=1`) speeds up generation on simple clean parts, while severe defects like cracks require `top_k=3` to capture both the zero-tolerance rule and the supervisor escalation contact. The bandit automatically learns to pick the fastest depth that still satisfies the operator.
