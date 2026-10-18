"""End-to-End Test Suite and 25-Iteration Online RL Convergence Simulation.

This script:
1. Validates all FastAPI endpoints (/health, /inspect JSON, /inspect multipart, /feedback, /transactions/{id}).
2. Resets the bandit to baseline Q-values (0.0).
3. Simulates 25 realistic worker inspection + feedback cycles across multiple defect states.
4. Demonstrates clear policy convergence:
   - Simple states ('no_defect', 'surface_scratch') converge to top_k=1 for lowest latency.
   - Complex states ('crack') converge to top_k=3 because top_k=1 misses secondary escalation contacts and receives unhelpful feedback (score=0).
5. Outputs a formatted Markdown comparison table of Q-values and counts.
"""

import json
import os
import sys
import time
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.main import app

client = TestClient(app)


def run_e2e_api_tests():
    """Runs functional sanity checks on all API endpoints."""
    print("=================================================================")
    print("STEP 1: RUNNING END-TO-END FUNCTIONAL API TESTS")
    print("=================================================================")

    # 1. Health check
    h = client.get("/health")
    assert h.status_code == 200, f"Health check failed: {h.text}"
    print(f"[*] GET /health -> OK: {h.json()['service']} (Model: {h.json()['llm_model']})")

    # 2. Inspect with JSON body & image path (Clean part)
    res_clean = client.post("/inspect", json={
        "query": "Is this part within tolerance?",
        "image_path": str(ROOT_DIR / "data" / "mock_images" / "part_clean.png"),
    })
    assert res_clean.status_code == 200, f"JSON inspect failed: {res_clean.text}"
    d_clean = res_clean.json()
    assert d_clean["defect_category"] == "no_defect"
    assert d_clean["extracted_ocr_text"] == "WO-1001"
    print(f"[*] POST /inspect (JSON, clean part) -> OK (Tx: {d_clean['transaction_id'][:8]}..., Defect: {d_clean['defect_category']}, WO: {d_clean['extracted_ocr_text']})")

    # 3. Inspect with Multipart Form upload (Crack part)
    crack_path = ROOT_DIR / "data" / "mock_images" / "part_crack.png"
    with open(crack_path, "rb") as f:
        res_crack = client.post(
            "/inspect",
            data={"query": "Found a fissure, who do I escalate to and should we scrap?"},
            files={"image": ("part_crack.png", f, "image/png")},
        )
    assert res_crack.status_code == 200, f"Multipart inspect failed: {res_crack.text}"
    d_crack = res_crack.json()
    assert d_crack["defect_category"] == "crack"
    assert d_crack["extracted_ocr_text"] == "WO-1003"
    assert d_crack["proposed_action"] is not None
    assert d_crack["proposed_action"]["status"] == "PROPOSED_PENDING_HUMAN_APPROVAL"
    assert d_crack["proposed_action"]["action_type"] == "SCRAP_PART"
    print(f"[*] POST /inspect (Multipart, crack part) -> OK (Defect: {d_crack['defect_category']}, Proposed Action: {d_crack['proposed_action']['action_type']})")

    # 4. Inspect without image (Text-only query)
    res_text = client.post("/inspect", json={
        "query": "What are the rules for red quarantine tagging?",
        "work_order_id": "WO-1002",
    })
    assert res_text.status_code == 200, f"Text-only inspect failed: {res_text.text}"
    d_text = res_text.json()
    assert d_text["defect_category"] == "no_image_supplied"
    assert d_text["work_order_status"]["work_order_id"] == "WO-1002"
    print(f"[*] POST /inspect (Text-only, WO-1002) -> OK (Defect: {d_text['defect_category']}, Batch: {d_text['work_order_status']['batch_status']})")

    # 5. Feedback submission
    tx_id = d_clean["transaction_id"]
    res_fb = client.post("/feedback", json={"transaction_id": tx_id, "feedback_score": 1})
    assert res_fb.status_code == 200, f"Feedback failed: {res_fb.text}"
    fb_data = res_fb.json()
    print(f"[*] POST /feedback -> OK (Tx: {tx_id[:8]}..., Reward: {fb_data['computed_reward']}, Score: 1)")

    # 6. Replay past transaction from audit log
    res_tx = client.get(f"/transactions/{tx_id}")
    assert res_tx.status_code == 200, f"Get transaction failed: {res_tx.text}"
    tx_record = res_tx.json()
    assert tx_record["transaction_id"] == tx_id
    assert tx_record["feedback"] is not None
    assert len(tx_record["decision_trace"]) > 0
    print(f"[*] GET /transactions/{tx_id[:8]}... -> OK (Replayed {len(tx_record['decision_trace'])} trace steps, verified audit trail)")

    print("\nALL FUNCTIONAL ENDPOINT TESTS PASSED WITH ZERO ERRORS.\n")


def simulate_rl_convergence():
    """Simulates 25 inspection + feedback cycles to demonstrate Contextual Bandit convergence."""
    print("=================================================================")
    print("STEP 2: RUNNING 25-ITERATION ONLINE RL CONVERGENCE SIMULATION")
    print("=================================================================")

    # Reset bandit state to clean slate
    reset_res = client.post("/bandit/reset")
    assert reset_res.status_code == 200
    print("[*] Reset Contextual Bandit Q-table to 0.0 baseline.\n")

    # Scenarios for simulation
    # 1. Clean parts (no_defect) - Simple tolerance query.
    #    Any top_k gives the tolerance rule, so top_k=1 wins on lower latency penalty.
    # 2. Surface scratch (surface_scratch) - Simple rework routing to B-4.
    #    top_k=1 gives Station B-4 instantly, so top_k=1 wins on latency.
    # 3. Structural crack (crack) - Complex dual-requirement query.
    #    Worker asks: "Found a fissure, who do I escalate to and do we scrap?"
    #    - top_k=1 retrieves only tolerance chunk, misses Marcus Vance contact -> feedback_score = 0 (unhelpful) -> severe negative reward!
    #    - top_k=3 retrieves both scrap rule and Marcus Vance contact chunk -> feedback_score = 1 (helpful) -> high reward!
    #    - top_k=5 retrieves chunks with higher latency -> feedback_score = 1 -> slightly lower reward than top_k=3.

    simulation_scenarios = [
        # 10 clean part queries
        {
            "state_type": "no_defect",
            "img_path": "data/mock_images/part_clean.png",
            "query": "Is this part acceptable for production?",
            "needs_multi_chunk": False,
        },
        # 8 scratch queries
        {
            "state_type": "surface_scratch",
            "img_path": "data/mock_images/part_scratch.png",
            "query": "Can this scratch be reworked and where should I route it?",
            "needs_multi_chunk": False,
        },
        # 7 crack queries
        {
            "state_type": "crack",
            "img_path": "data/mock_images/part_crack.png",
            "query": "Found a crack, who do I escalate to and do we scrap?",
            "needs_multi_chunk": True,
        },
    ]

    # Interleave 25 trials
    schedule = []
    # 10 clean, 8 scratch, 7 crack = 25
    for i in range(10):
        schedule.append(simulation_scenarios[0])
    for i in range(8):
        schedule.append(simulation_scenarios[1])
    for i in range(7):
        schedule.append(simulation_scenarios[2])

    # Interleave order to represent realistic mixed shop-floor stream
    mixed_trials = []
    c_idx, s_idx, k_idx = 0, 0, 0
    while len(mixed_trials) < 25:
        if c_idx < 10:
            mixed_trials.append(simulation_scenarios[0])
            c_idx += 1
        if s_idx < 8:
            mixed_trials.append(simulation_scenarios[1])
            s_idx += 1
        if k_idx < 7:
            mixed_trials.append(simulation_scenarios[2])
            k_idx += 1

    print(f"{'Iter':<5} | {'State (Defect)':<18} | {'Action (top_k)':<14} | {'Latency':<8} | {'Feedback':<8} | {'Reward':<8} | {'Best Action So Far'}")
    print("-" * 88)

    iteration_logs = []

    for i, trial in enumerate(mixed_trials, start=1):
        # 1. Post inspection
        inspect_res = client.post("/inspect", json={
            "query": trial["query"],
            "image_path": str(ROOT_DIR / trial["img_path"]),
        })
        assert inspect_res.status_code == 200
        data = inspect_res.json()
        tx_id = data["transaction_id"]
        state = data["defect_category"]
        chosen_action = data["pipeline_config"]["rag_top_k"]
        latency = data["execution_latency_seconds"]

        # 2. Determine realistic worker feedback based on shop-floor utility
        if trial["needs_multi_chunk"]:
            # Crack query needing escalation contact
            if chosen_action == 1:
                # Missing escalation contact: unhelpful
                feedback_score = 0
            else:
                # top_k=3 or 5 retrieves both scrap rule and escalation contact
                feedback_score = 1
        else:
            # Simple clean/scratch queries: top_k=1, 3, 5 all answer the question
            feedback_score = 1

        # 3. Submit feedback
        fb_res = client.post("/feedback", json={
            "transaction_id": tx_id,
            "feedback_score": feedback_score,
        })
        assert fb_res.status_code == 200
        fb_data = fb_res.json()
        reward = fb_data["computed_reward"]
        q_vals = fb_data["updated_q_values"]

        # Current argmax action
        best_a = max(q_vals.items(), key=lambda kv: kv[1])[0]

        iteration_logs.append({
            "iter": i,
            "state": state,
            "action": chosen_action,
            "latency": latency,
            "feedback": feedback_score,
            "reward": reward,
            "best_action": best_a,
        })

        print(f"{i:<5} | {state:<18} | top_k={chosen_action:<8} | {latency:<6.3f}s | {feedback_score:<8} | {reward:<8.3f} | top_k={best_a} (Q={q_vals[best_a]:.2f})")

    # Fetch final bandit state
    state_res = client.get("/bandit/state")
    final_bandit = state_res.json()
    q_table = final_bandit["q_table"]
    counts = final_bandit["counts"]

    print("\n" + "=" * 88)
    print("FINAL CONTEXTUAL BANDIT CONVERGENCE SUMMARY TABLE")
    print("=" * 88)

    md_table = generate_markdown_summary(q_table, counts)
    print(md_table)
    return md_table


def generate_markdown_summary(q_table, counts):
    """Generates a clean Markdown table showing Q-values, visit counts, and converged actions."""
    lines = []
    lines.append("| Defect State | Q(top_k=1) | Q(top_k=3) | Q(top_k=5) | Counts [1, 3, 5] | Converged Optimal Action | Dominant Driver |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---|")

    driver_map = {
        "no_defect": "Latency penalty minimization (single SOP tolerance chunk sufficient)",
        "surface_scratch": "Station B-4 routing retrieved in top 1 chunk; lower latency wins",
        "crack": "top_k=1 misses escalation contact (score=0 penalty); top_k=3 captures both",
        "dimensional_deviation": "Default baseline initialized; explores balanced top_k=3",
        "no_image_supplied": "Standard baseline fallback",
    }

    for state in ["no_defect", "surface_scratch", "crack", "dimensional_deviation", "no_image_supplied"]:
        q = q_table.get(state, {"1": 0.0, "3": 0.0, "5": 0.0})
        c = counts.get(state, {"1": 0, "3": 0, "5": 0})
        best_a = max(q.items(), key=lambda kv: kv[1])[0] if q else "1"
        counts_str = f"[{c.get('1', 0)}, {c.get('3', 0)}, {c.get('5', 0)}]"
        driver = driver_map.get(state, "Exploration")
        lines.append(
            f"| `{state}` | **{q.get('1', 0.0):.2f}** | **{q.get('3', 0.0):.2f}** | **{q.get('5', 0.0):.2f}** | {counts_str} | **`top_k={best_a}`** | {driver} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    run_e2e_api_tests()
    simulate_rl_convergence()
