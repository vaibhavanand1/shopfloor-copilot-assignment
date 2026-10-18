# SOP-OPS-04: ERP Traveler Logging and Human Checkpoint Governance

## 1. Digital Traveler & Work Order Tracking
All machined parts traverse shop floor work centers accompanied by a standardized digital Work Order identifier (formatted as WO-XXXX, e.g., WO-1001 through WO-9999). Every operation, inspection sign-off, and scrap tally must be linked to the active Work Order number in the MES/ERP system.

## 2. Mandatory Human Approval Checkpoint
Under factory governance policies, automated inspection systems, vision classifiers, and AI copilot agents are classified as decision-support tools only. The API and copilot agents are strictly prohibited from mutating or finalizing scrap or escalation records directly in the ERP. All AI recommendations to scrap a part or escalate an incident must be generated with status PROPOSED_PENDING_HUMAN_APPROVAL. A qualified operator, certified inspector, or shift lead must physically verify the part and authenticate the change before the traveler status updates.

## 3. Red Quarantine Tagging and Physical Segregation
Whenever a part is proposed for scrap or pending escalation review:
1. The operator must print and attach a high-visibility Red Quarantine Tag inscribed with the Work Order ID, timestamp, and defect description.
2. The component must be physically moved to the locked Quarantine Cage at the end of the line.
3. The ERP traveler must show status HELD_FOR_QA until the Quality Lead or designated supervisor reviews and confirms the disposition.
