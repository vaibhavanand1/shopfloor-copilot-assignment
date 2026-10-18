"""Epsilon-Greedy Contextual Bandit optimizing RAG Top-K retrieval for ShopFloor Copilot.

State Space (5 states):
- no_defect
- surface_scratch
- crack
- dimensional_deviation
- no_image_supplied

Action Space (1 axis optimized - RAG Top-K):
- 1 (ultra-fast, minimum token latency)
- 3 (balanced context)
- 5 (comprehensive multi-document retrieval)

Reward Formula:
- reward = (feedback_score * 10.0) - latency_in_seconds

Update Rule:
- Incremental sample average: Q(s, a) <- Q(s, a) + (1 / N(s, a)) * (reward - Q(s, a))
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BANDIT_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "bandit_state.json"

STATES = [
    "no_defect",
    "surface_scratch",
    "crack",
    "dimensional_deviation",
    "no_image_supplied",
]

ACTIONS = [1, 3, 5]
EPSILON_DEFAULT = 0.20


class ContextualBandit:
    def __init__(
        self,
        state_file: Path = BANDIT_STATE_FILE,
        epsilon: float = EPSILON_DEFAULT,
        actions: Optional[List[int]] = None,
    ):
        self.state_file = state_file
        self.epsilon = epsilon
        self.actions = actions or ACTIONS
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.counts: Dict[str, Dict[str, int]] = {}
        self._initialize_or_load()

    def _initialize_or_load(self):
        """Initializes empty Q-table and counts, or loads persisted state."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                self.q_table = data.get("q_table", {})
                self.counts = data.get("counts", {})
            except Exception:
                self._create_empty_state()
        else:
            self._create_empty_state()

        # Ensure all states and actions exist
        for s in STATES:
            if s not in self.q_table:
                self.q_table[s] = {str(a): 0.0 for a in self.actions}
            if s not in self.counts:
                self.counts[s] = {str(a): 0 for a in self.actions}
            for a in self.actions:
                if str(a) not in self.q_table[s]:
                    self.q_table[s][str(a)] = 0.0
                if str(a) not in self.counts[s]:
                    self.counts[s][str(a)] = 0

    def _create_empty_state(self):
        self.q_table = {s: {str(a): 0.0 for a in self.actions} for s in STATES}
        self.counts = {s: {str(a): 0 for a in self.actions} for s in STATES}
        self.save()

    def save(self):
        """Atomically persists Q-values and counts to disk."""
        data = {
            "q_table": self.q_table,
            "counts": self.counts,
            "epsilon": self.epsilon,
            "actions": self.actions,
            "states": STATES,
        }
        temp_file = self.state_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp_file.replace(self.state_file)

    def select_action(self, state: str, explore: bool = True) -> int:
        """Selects RAG top_k action using epsilon-greedy exploration/exploitation."""
        if state not in self.q_table:
            state = "no_image_supplied"

        # Epsilon exploration
        if explore and random.random() < self.epsilon:
            return random.choice(self.actions)

        # Exploitation: argmax_a Q(s, a)
        action_q = self.q_table[state]
        max_q = max(action_q.values())
        # Break ties randomly among best actions
        best_actions = [int(a_str) for a_str, q in action_q.items() if q == max_q]
        return random.choice(best_actions)

    def update(self, state: str, action: int, reward: float) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Updates Q(s, a) using incremental sample-average update rule."""
        if state not in self.q_table:
            state = "no_image_supplied"

        action_key = str(action)
        if action_key not in self.q_table[state]:
            action_key = str(self.actions[0])

        current_n = self.counts[state][action_key] + 1
        self.counts[state][action_key] = current_n

        old_q = self.q_table[state][action_key]
        # Incremental sample average formula: Q_new = Q_old + (1/n) * (R - Q_old)
        new_q = old_q + (1.0 / current_n) * (reward - old_q)
        self.q_table[state][action_key] = round(new_q, 4)

        self.save()
        return self.q_table[state], self.counts[state]

    def get_state_q_values(self, state: str) -> Dict[str, float]:
        if state not in self.q_table:
            state = "no_image_supplied"
        return self.q_table[state]

    def get_state_counts(self, state: str) -> Dict[str, int]:
        if state not in self.counts:
            state = "no_image_supplied"
        return self.counts[state]

    def reset(self):
        """Resets all Q-values and counts to 0.0."""
        self._create_empty_state()
