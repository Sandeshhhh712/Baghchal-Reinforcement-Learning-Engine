"""A small Baghchal rules engine and tabular Q-learning agents."""

from .agents import FrozenQAgent, QAgent, RandomAgent, load_model, save_model
from .game import (
    Baghchal,
    EMPTY,
    GOAT,
    TIGER,
    TOTAL_GOATS,
    CAPTURES_TO_WIN,
    DEFAULT_MAX_PLIES,
)

__all__ = [
    "Baghchal",
    "EMPTY",
    "GOAT",
    "TIGER",
    "TOTAL_GOATS",
    "CAPTURES_TO_WIN",
    "DEFAULT_MAX_PLIES",
    "QAgent",
    "FrozenQAgent",
    "RandomAgent",
    "save_model",
    "load_model",
]

