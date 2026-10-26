"""Episode runner plus simple training and evaluation loops."""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .agents import FrozenQAgent, QAgent, RandomAgent
from .game import (
    DEFAULT_MAX_PLIES,
    GOAT,
    TIGER,
    Baghchal,
)

Policy = RandomAgent | QAgent


def linear_schedule(start: float, end: float, episode: int, total: int) -> float:
    """Linearly interpolate from ``start`` to ``end`` over ``total`` episodes."""
    if total <= 1:
        return end
    fraction = (episode - 1) / (total - 1)
    fraction = min(1.0, max(0.0, fraction))
    return start + (end - start) * fraction


def run_episode(
    game: Baghchal,
    policies: dict[int, Policy],
    *,
    epsilon: float = 0.0,
    rng: Optional[random.Random] = None,
    learn: bool = False,
    capture_bonus: float = 0.0,
) -> tuple[Optional[int], int]:
    """Play one episode, optionally updating Q-tables.

    Each learner's transition runs from one of its turns to its next turn.
    The reward is normally 0, except for the optional capture bonus and the
    final ``+1`` win / ``-1`` loss. When a player's transition is complete,
    its Q entry is bootstrapped from the best action available in the new
    state.

    Returns ``(winner, plies)``. ``winner`` is ``None`` for a draw.
    """
    rng = rng or random.Random()

    # player -> [state, action, reward accumulated so far]
    pending: dict[int, list] = {}

    while not game.is_terminal():
        player = game.current_player
        actions = game.legal_actions()
        if not actions:
            raise RuntimeError("Non-terminal state has no legal actions.")

        policy = policies[player]
        state = game.key()
        action = policy.choose(game, epsilon=epsilon, rng=rng)
        pending[player] = [state, action, 0.0]

        captured_before = game.goats_captured
        game.apply(action)
        captured = game.goats_captured - captured_before

        # Optional shaping: a tiger capture is good for the tiger and bad
        # for the goats. The total shaping over both players is still zero.
        if learn and capture_bonus and captured:
            if player == TIGER:
                pending[player][2] += capture_bonus * captured
                if GOAT in pending:
                    pending[GOAT][2] -= capture_bonus * captured

        if game.is_terminal():
            winner = game.winner()
            if learn:
                for pending_player, (state, action, reward) in pending.items():
                    if winner == pending_player:
                        reward += 1.0
                    elif winner is not None:
                        reward -= 1.0
                    policies[pending_player].update(
                        state, action, reward, done=True
                    )
            pending.clear()
            break

        # The game is still running. The player who is about to move has
        # now completed the transition that started on its previous turn.
        current = game.current_player
        if current in pending:
            state, action, reward = pending.pop(current)
            if learn:
                next_actions = game.legal_actions()
                policies[current].update(
                    state,
                    action,
                    reward,
                    game.key(),
                    next_actions,
                    done=False,
                )

    return game.winner(), game.plies


def evaluate(
    policies: dict[int, Policy],
    *,
    episodes: int = 500,
    seed: int = 0,
    max_plies: int = DEFAULT_MAX_PLIES,
) -> dict:
    """Play greedy/random games and return win counts."""
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    rng = random.Random(seed)
    tiger_wins = 0
    goat_wins = 0
    draws = 0
    total_plies = 0

    for _ in range(episodes):
        game = Baghchal(max_plies=max_plies)
        winner, plies = run_episode(
            game, policies, epsilon=0.0, rng=rng, learn=False
        )
        total_plies += plies
        if winner == TIGER:
            tiger_wins += 1
        elif winner == GOAT:
            goat_wins += 1
        else:
            draws += 1

    return {
        "episodes": episodes,
        "tiger_wins": tiger_wins,
        "goat_wins": goat_wins,
        "draws": draws,
        "tiger_win_rate": tiger_wins / episodes,
        "goat_win_rate": goat_wins / episodes,
        "draw_rate": draws / episodes,
        "avg_plies": total_plies / episodes,
    }


@dataclass
class TrainConfig:
    mode: str = "self-play"  # self-play, tiger, or goat
    episodes: int = 20_000
    alpha_start: float = 0.2
    alpha_end: float = 0.05
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    capture_bonus: float = 0.0
    max_plies: int = DEFAULT_MAX_PLIES
    seed: int = 42
    log_every: int = 500
    log: bool = True
    # Optional existing Q-tables, used by --load to continue training.
    tiger_q: Optional[dict] = None
    goat_q: Optional[dict] = None


@dataclass
class TrainResult:
    tiger: Optional[QAgent]
    goat: Optional[QAgent]
    episodes: int
    tiger_wins: int
    goat_wins: int
    draws: int
    total_plies: int
    elapsed: float


def train(config: TrainConfig) -> TrainResult:
    """Train one or both sides against random opponents or each other."""
    if config.mode not in {"self-play", "tiger", "goat"}:
        raise ValueError("mode must be 'self-play', 'tiger', or 'goat'.")
    if config.episodes <= 0:
        raise ValueError("episodes must be positive.")
    if config.log_every <= 0:
        raise ValueError("log_every must be positive.")
    if config.max_plies <= 0:
        raise ValueError("max_plies must be positive.")

    rng = random.Random(config.seed)

    tiger: Optional[QAgent] = None
    goat: Optional[QAgent] = None

    if config.mode in {"self-play", "tiger"}:
        tiger = QAgent(
            alpha=config.alpha_start,
            gamma=config.gamma,
            epsilon=config.epsilon_start,
            q=config.tiger_q,
            seed=config.seed,
        )
    if config.mode in {"self-play", "goat"}:
        goat = QAgent(
            alpha=config.alpha_start,
            gamma=config.gamma,
            epsilon=config.epsilon_start,
            q=config.goat_q,
            seed=config.seed + 1,
        )

    # In single-side modes, use a loaded Q-table as a frozen opponent when
    # one is available. This makes alternating training easy:
    #   train tiger -> save -> train goat against that tiger -> save -> ...
    tiger_opponent: Policy
    goat_opponent: Policy
    if config.mode == "tiger":
        assert tiger is not None
        tiger_opponent = tiger
        goat_opponent = (
            FrozenQAgent(q=config.goat_q)
            if config.goat_q is not None
            else RandomAgent(seed=config.seed + 3)
        )
    elif config.mode == "goat":
        assert goat is not None
        tiger_opponent = (
            FrozenQAgent(q=config.tiger_q)
            if config.tiger_q is not None
            else RandomAgent(seed=config.seed + 2)
        )
        goat_opponent = goat
    else:
        assert tiger is not None and goat is not None
        tiger_opponent = tiger
        goat_opponent = goat

    policies: dict[int, Policy] = {
        TIGER: tiger_opponent,
        GOAT: goat_opponent,
    }

    tiger_wins = 0
    goat_wins = 0
    draws = 0
    total_plies = 0
    recent: deque[tuple[Optional[int], int]] = deque(maxlen=config.log_every)
    started = time.time()

    for episode in range(1, config.episodes + 1):
        alpha = linear_schedule(
            config.alpha_start, config.alpha_end, episode, config.episodes
        )
        epsilon = linear_schedule(
            config.epsilon_start, config.epsilon_end, episode, config.episodes
        )
        if tiger is not None:
            tiger.alpha = alpha
            tiger.epsilon = epsilon
        if goat is not None:
            goat.alpha = alpha
            goat.epsilon = epsilon

        game = Baghchal(max_plies=config.max_plies)
        winner, plies = run_episode(
            game,
            policies,
            epsilon=epsilon,
            rng=rng,
            learn=True,
            capture_bonus=config.capture_bonus,
        )

        if winner == TIGER:
            tiger_wins += 1
        elif winner == GOAT:
            goat_wins += 1
        else:
            draws += 1
        total_plies += plies
        recent.append((winner, plies))

        if config.log and episode % config.log_every == 0:
            window = len(recent)
            window_tiger = sum(1 for w, _ in recent if w == TIGER)
            window_goat = sum(1 for w, _ in recent if w == GOAT)
            window_draw = window - window_tiger - window_goat
            window_plies = sum(p for _, p in recent) / window
            tiger_entries = tiger.num_entries() if tiger is not None else 0
            goat_entries = goat.num_entries() if goat is not None else 0
            elapsed = time.time() - started
            print(
                f"ep {episode:>7}/{config.episodes} | "
                f"eps {epsilon:0.3f} alpha {alpha:0.3f} | "
                f"last {window}: T {window_tiger / window:5.1%} "
                f"G {window_goat / window:5.1%} "
                f"D {window_draw / window:5.1%} | "
                f"plies {window_plies:5.1f} | "
                f"q T:{tiger_entries} G:{goat_entries} | "
                f"{elapsed:6.1f}s"
            )

    if config.log and config.episodes % config.log_every != 0:
        elapsed = time.time() - started
        print(
            f"ep {config.episodes:>7}/{config.episodes} | "
            f"T {tiger_wins / config.episodes:5.1%} "
            f"G {goat_wins / config.episodes:5.1%} "
            f"D {draws / config.episodes:5.1%} | "
            f"avg plies {total_plies / config.episodes:5.1f} | "
            f"q T:{tiger.num_entries() if tiger else 0} "
            f"G:{goat.num_entries() if goat else 0} | "
            f"{elapsed:6.1f}s"
        )

    return TrainResult(
        tiger=tiger,
        goat=goat,
        episodes=config.episodes,
        tiger_wins=tiger_wins,
        goat_wins=goat_wins,
        draws=draws,
        total_plies=total_plies,
        elapsed=time.time() - started,
    )
