"""RL scheduling policy: Q-learning over expert placement actions (numpy-only)."""
from __future__ import annotations

import numpy as np

from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.state import ScheduleState


class RLScheduler(Scheduler):
    """Learn expert placement via Q-learning. Actions: load (GPU) vs execute_cpu.
    Resident experts always execute on GPU. Deterministic with fixed numpy seed."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict[tuple[str, str], float] = {}

    def _q(self, key: tuple[str, str]) -> float:
        return self.q_table.get(key, 0.0)

    def _choose_action(self, expert_id: str) -> str:
        if np.random.rand() < self.epsilon:
            return "execute_cpu" if np.random.rand() < 0.5 else "load"
        q_load = self._q((expert_id, "load"))
        q_cpu = self._q((expert_id, "execute_cpu"))
        return "load" if q_load >= q_cpu else "execute_cpu"

    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        actions: list[Action] = []
        for eid in state.requested:
            if eid in state.resident:
                actions.append(Action(kind="execute_gpu", expert_ids=(eid,)))
            else:
                action = self._choose_action(eid)
                actions.append(Action(kind=action, expert_ids=(eid,)))
        return actions

    def train(self, sim, episodes: int = 200, trace: list[list[str]] | None = None,
              epsilon_decay: float = 0.99) -> None:
        trace = trace or [["e0", "e1"]]
        for episode in range(episodes):
            prev_state = None
            for step in trace:
                sim._state.requested = tuple(step)
                actions = self.decide(sim._state, sim._clock)
                # reward proxy: negative of executed cpu_ms (cheaper placement rewarded)
                reward = -sum(
                    sim.profiles[eid].cpu_exec_ms
                    for a in actions if a.kind == "execute_cpu" for eid in a.expert_ids
                )
                for a in actions:
                    if a.kind in ("load", "execute_cpu"):
                        eid = a.expert_ids[0]
                        key = (eid, a.kind)
                        q = self._q(key)
                        self.q_table[key] = q + self.alpha * (reward - q)
                prev_state = step
            self.epsilon *= epsilon_decay
