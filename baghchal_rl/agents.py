"""Policies for Baghchal: a random baseline and tabular Q-learning."""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Optional

from .game import Action, Baghchal, StateKey

MODEL_FORMAT = "baghchal-qtable-v1"


class RandomAgent:
    """A uniform random legal-action policy."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def choose(
        self,
        game: Baghchal,
        epsilon: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> Action:
        rng = rng or self.rng
        actions = game.legal_actions()
        if not actions:
            raise ValueError("No legal actions available.")
        return rng.choice(actions)

    def update(self, *args, **kwargs) -> None:
        """RandomAgent does not learn; present so the trainer can call it."""
        return None

    @property
    def q(self) -> dict:
        return {}


class QAgent:
    """Tabular Q-learning agent.

    The Q-table is a nested dictionary::

        Q[state_key][action] = value

    Missing entries are treated as zero. States come from
    :meth:`Baghchal.key`, so they include the board layout, the number of
    goats placed, and the side to move.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        q: Optional[dict[StateKey, dict[Action, float]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q: dict[StateKey, dict[Action, float]] = q if q is not None else {}
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Q-table access
    # ------------------------------------------------------------------
    def q_value(self, state: StateKey, action: Action) -> float:
        return self.q.get(state, {}).get(action, 0.0)

    def max_q(self, state: StateKey, actions: list[Action]) -> float:
        if not actions:
            return 0.0
        return max(self.q_value(state, action) for action in actions)

    def _set_q(self, state: StateKey, action: Action, value: float) -> None:
        # Keeping exact zeros out of the table makes it materially smaller.
        if value == 0.0:
            values = self.q.get(state)
            if values is not None:
                values.pop(action, None)
                if not values:
                    del self.q[state]
            return
        self.q.setdefault(state, {})[action] = value

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def best_action(
        self,
        state: StateKey,
        actions: list[Action],
        rng: Optional[random.Random] = None,
    ) -> Optional[Action]:
        if not actions:
            return None
        rng = rng or self.rng

        values = [self.q_value(state, action) for action in actions]
        best_value = max(values)
        best_actions = [
            action for action, value in zip(actions, values) if value == best_value
        ]
        # Random tie-breaking is important early on, when every value is 0.
        return rng.choice(best_actions)

    def choose(
        self,
        game: Baghchal,
        epsilon: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> Action:
        actions = game.legal_actions()
        if not actions:
            raise ValueError("No legal actions available.")

        rng = rng or self.rng
        eps = self.epsilon if epsilon is None else epsilon
        if rng.random() < eps:
            return rng.choice(actions)

        action = self.best_action(game.key(), actions, rng)
        assert action is not None
        return action

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update(
        self,
        state: StateKey,
        action: Action,
        reward: float,
        next_state: Optional[StateKey] = None,
        next_actions: Optional[list[Action]] = None,
        done: bool = False,
    ) -> None:
        """One Q-learning update.

        When ``done`` is true the target is just the reward. Otherwise it is
        ``reward + gamma * max(Q(next_state, a))`` for the agent's next turn.
        """
        old_value = self.q_value(state, action)
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.max_q(next_state, next_actions or [])
        self._set_q(state, action, old_value + self.alpha * (target - old_value))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def num_entries(self) -> int:
        return sum(len(values) for values in self.q.values())

    def num_states(self) -> int:
        return len(self.q)


class FrozenQAgent(QAgent):
    """A greedy Q-table opponent that never updates its table.

    Useful for alternating training: train one side, freeze it, then train
    the other side against it.
    """

    def choose(
        self,
        game: Baghchal,
        epsilon: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> Action:
        return super().choose(game, epsilon=0.0, rng=rng)

    def update(self, *args, **kwargs) -> None:
        return None


def save_model(
    path: str | Path,
    tiger: Optional[QAgent] = None,
    goat: Optional[QAgent] = None,
    meta: Optional[dict] = None,
) -> Path:
    """Save one or both Q-tables to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": MODEL_FORMAT,
        "tiger": tiger.q if tiger is not None else None,
        "goat": goat.q if goat is not None else None,
        "meta": meta or {},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model(
    path: str | Path,
) -> tuple[
    Optional[dict[StateKey, dict[Action, float]]],
    Optional[dict[StateKey, dict[Action, float]]],
    dict,
]:
    """Load a Q-table model.

    Only load model files you created yourself: pickle can execute code.
    """
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict) or payload.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path} is not a Baghchal Q-table model.")

    return payload.get("tiger"), payload.get("goat"), payload.get("meta", {})
