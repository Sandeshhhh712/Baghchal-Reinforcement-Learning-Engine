#!/usr/bin/env python3
"""Play Baghchal against a trained Q-table agent, or watch it play.

Examples:
    python play.py
    python play.py --model models/baghchal_q.pkl --human goat
    python play.py --human tiger
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

from baghchal_rl.agents import QAgent, RandomAgent, load_model
from baghchal_rl.game import GOAT, TIGER, Action, Baghchal, action_text
from baghchal_rl.ui import (
    describe_action,
    help_text,
    legal_action_lines,
    parse_action,
    player_name,
    render,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play Baghchal against a tabular Q-learning agent."
    )
    parser.add_argument(
        "--human",
        choices=["goat", "tiger"],
        default="goat",
        help="side you play (default: goat)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/baghchal_q.pkl",
        help="Q-table model created by train.py",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="seed for tie-breaking/random play"
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=300,
        help="draw after this many plies; 0 means no limit",
    )
    return parser


def prompt_human(game: Baghchal) -> Optional[Action]:
    while True:
        try:
            text = input(f"Your move ({player_name(game.current_player)}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if not text:
            continue

        command = text.lower()
        if command in {"q", "quit", "exit"}:
            return None
        if command in {"h", "help", "?"}:
            print(help_text())
            continue
        if command in {"moves", "legal", "l"}:
            print("Legal actions: " + ", ".join(legal_action_lines(game)))
            continue
        if command in {"board", "b"}:
            print(render(game))
            continue

        action = parse_action(text)
        if action is None:
            print("Could not parse that. Type 'help' for commands.")
            continue

        if game.phase == "movement" and action[0] == -1:
            print("All goats are placed; use 'src dst' to move a goat.")
            continue

        if not game.is_legal(action):
            print(
                f"Illegal action: {action_text(action)}. "
                "Type 'moves' for legal actions."
            )
            continue

        return action


def main() -> int:
    args = build_parser().parse_args()

    human = GOAT if args.human == "goat" else TIGER
    agent_side = TIGER if human == GOAT else GOAT
    rng = random.Random(args.seed)

    tiger_q = None
    goat_q = None
    model_path = Path(args.model) if args.model else None
    if model_path is not None and model_path.exists():
        try:
            tiger_q, goat_q, meta = load_model(model_path)
        except Exception as exc:  # noqa: BLE001 - show a friendly CLI error
            print(f"Could not load model {model_path}: {exc}")
            print("Falling back to a random agent.")
            tiger_q = None
            goat_q = None
            meta = {}
        else:
            print(f"Loaded model: {model_path}")
            if meta.get("mode"):
                print(
                    f"  trained mode: {meta['mode']}, "
                    f"episodes: {meta.get('episodes', '?')}"
                )
    else:
        if model_path is not None:
            print(
                f"Model not found: {model_path}\n"
                "Falling back to a random agent. Run train.py first to create one."
            )

    agent_q = tiger_q if agent_side == TIGER else goat_q
    if agent_q is None:
        print(
            f"No trained Q-table for the {player_name(agent_side).lower()}; "
            "using a random agent."
        )
        agent = RandomAgent(seed=args.seed + 1)
    else:
        agent = QAgent(q=agent_q)
        print(
            f"Agent plays {player_name(agent_side).lower()} "
            f"({len(agent_q)} states in its Q-table)."
        )

    max_plies = None if args.max_plies <= 0 else args.max_plies
    game = Baghchal(max_plies=max_plies)

    print()
    print(render(game))
    print()

    while not game.is_terminal():
        player = game.current_player

        if player == human:
            action = prompt_human(game)
            if action is None:
                print("Game stopped.")
                return 0
        else:
            action = agent.choose(game, epsilon=0.0, rng=rng)
            print(
                f"Agent ({player_name(player)}) chooses: "
                f"{describe_action(game, action)}"
            )

        captured_before = game.goats_captured
        game.apply(action)
        captured = game.goats_captured - captured_before
        if captured:
            src, dst = action
            print(f"  Goat at {(src + dst) // 2} was captured!")

        print()
        print(render(game))
        print()

    winner = game.winner()
    if winner == TIGER:
        print("Tigers win!")
    elif winner == GOAT:
        print("Goats win!")
    else:
        print("Draw (move limit reached).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
