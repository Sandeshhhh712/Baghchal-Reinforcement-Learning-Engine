#!/usr/bin/env python3
"""Train tabular Q-learning agents for Baghchal.

Examples:
    python train.py --mode self-play --episodes 20000
    python train.py --mode tiger --episodes 10000 --save models/tiger.pkl
    python train.py --load models/baghchal_q.pkl --episodes 5000
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from baghchal_rl.agents import (
    FrozenQAgent,
    QAgent,
    RandomAgent,
    load_model,
    save_model,
)
from baghchal_rl.game import DEFAULT_MAX_PLIES, GOAT, TIGER
from baghchal_rl.trainer import TrainConfig, evaluate, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a simple tabular Q-learning agent for Baghchal."
    )
    parser.add_argument(
        "--mode",
        choices=["self-play", "tiger", "goat"],
        default="self-play",
        help=(
            "self-play trains both sides against each other; tiger trains "
            "only tigers against a random goat (or a frozen loaded goat); "
            "goat trains only goats against a random tiger (or a frozen "
            "loaded tiger) (default: self-play)."
        ),
    )
    parser.add_argument(
        "--episodes", type=int, default=20_000, help="number of training games"
    )
    parser.add_argument(
        "--alpha-start", type=float, default=0.2, help="initial learning rate"
    )
    parser.add_argument(
        "--alpha-end", type=float, default=0.05, help="final learning rate"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor"
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="initial exploration rate",
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=0.05, help="final exploration rate"
    )
    parser.add_argument(
        "--capture-bonus",
        type=float,
        default=0.0,
        help=(
            "optional reward for each goat captured (tigers get +bonus, "
            "goats get -bonus); 0 disables shaping"
        ),
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=DEFAULT_MAX_PLIES,
        help="draw after this many plies",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--log-every", type=int, default=500, help="progress log interval"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="models/baghchal_q.pkl",
        help="where to save the Q-table model",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="do not write a model file"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="existing model to continue training from",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=500,
        help="number of evaluation games after training",
    )
    parser.add_argument(
        "--no-eval", action="store_true", help="skip evaluation games"
    )
    parser.add_argument("--quiet", action="store_true", help="hide progress logs")
    return parser


def print_evaluation(label: str, stats: dict) -> None:
    print(
        f"  {label:<34} "
        f"T {stats['tiger_win_rate']:6.1%}  "
        f"G {stats['goat_win_rate']:6.1%}  "
        f"D {stats['draw_rate']:6.1%}  "
        f"avg plies {stats['avg_plies']:5.1f}"
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.eval_episodes < 0:
        parser.error("--eval-episodes cannot be negative")
    if args.max_plies <= 0:
        parser.error("--max-plies must be positive")

    tiger_q = None
    goat_q = None
    loaded_meta: dict = {}

    if args.load:
        load_path = Path(args.load)
        if not load_path.exists():
            print(f"Model not found: {load_path}")
            return 1
        try:
            tiger_q, goat_q, loaded_meta = load_model(load_path)
        except Exception as exc:  # noqa: BLE001 - show a friendly CLI error
            print(f"Could not load {load_path}: {exc}")
            return 1
        print(
            f"Loaded {load_path} "
            f"(tiger entries: {sum(len(v) for v in tiger_q.values()) if tiger_q else 0}, "
            f"goat entries: {sum(len(v) for v in goat_q.values()) if goat_q else 0})"
        )

    config = TrainConfig(
        mode=args.mode,
        episodes=args.episodes,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        capture_bonus=args.capture_bonus,
        max_plies=args.max_plies,
        seed=args.seed,
        log_every=args.log_every,
        log=not args.quiet,
        tiger_q=tiger_q,
        goat_q=goat_q,
    )

    result = train(config)

    print(
        f"\nTraining finished in {result.elapsed:.1f}s "
        f"({result.episodes} games, avg {result.total_plies / result.episodes:.1f} plies)."
    )
    print(
        f"Results: tigers {result.tiger_wins}, goats {result.goat_wins}, "
        f"draws {result.draws}."
    )
    if result.tiger is not None:
        print(
            f"Tiger Q-table: {result.tiger.num_states()} states, "
            f"{result.tiger.num_entries()} state-action entries."
        )
    if result.goat is not None:
        print(
            f"Goat Q-table:  {result.goat.num_states()} states, "
            f"{result.goat.num_entries()} state-action entries."
        )

    if not args.no_eval and args.eval_episodes > 0:
        print("\nEvaluation (greedy trained agents):")
        if result.tiger is not None:
            print_evaluation(
                "tiger (trained) vs random goat",
                evaluate(
                    {
                        TIGER: result.tiger,
                        GOAT: RandomAgent(seed=args.seed + 10),
                    },
                    episodes=args.eval_episodes,
                    seed=args.seed + 11,
                    max_plies=args.max_plies,
                ),
            )
        if result.goat is not None:
            print_evaluation(
                "random tiger vs goat (trained)",
                evaluate(
                    {
                        TIGER: RandomAgent(seed=args.seed + 12),
                        GOAT: result.goat,
                    },
                    episodes=args.eval_episodes,
                    seed=args.seed + 13,
                    max_plies=args.max_plies,
                ),
            )
        if result.tiger is not None and result.goat is not None:
            print_evaluation(
                "tiger (trained) vs goat (trained)",
                evaluate(
                    {TIGER: result.tiger, GOAT: result.goat},
                    episodes=args.eval_episodes,
                    seed=args.seed + 14,
                    max_plies=args.max_plies,
                ),
            )
        if args.mode == "tiger" and result.tiger is not None and goat_q is not None:
            print_evaluation(
                "tiger (trained) vs goat (loaded)",
                evaluate(
                    {TIGER: result.tiger, GOAT: FrozenQAgent(q=goat_q)},
                    episodes=args.eval_episodes,
                    seed=args.seed + 18,
                    max_plies=args.max_plies,
                ),
            )
        if args.mode == "goat" and result.goat is not None and tiger_q is not None:
            print_evaluation(
                "tiger (loaded) vs goat (trained)",
                evaluate(
                    {TIGER: FrozenQAgent(q=tiger_q), GOAT: result.goat},
                    episodes=args.eval_episodes,
                    seed=args.seed + 19,
                    max_plies=args.max_plies,
                ),
            )
        print_evaluation(
            "random vs random (baseline)",
            evaluate(
                {
                    TIGER: RandomAgent(seed=args.seed + 15),
                    GOAT: RandomAgent(seed=args.seed + 16),
                },
                episodes=args.eval_episodes,
                seed=args.seed + 17,
                max_plies=args.max_plies,
            ),
        )

    if not args.no_save:
        # In single-side modes, preserve the frozen loaded opponent so
        # alternating training does not throw away the other side's table.
        tiger_to_save = result.tiger
        if tiger_to_save is None and tiger_q is not None:
            tiger_to_save = QAgent(q=tiger_q)
        goat_to_save = result.goat
        if goat_to_save is None and goat_q is not None:
            goat_to_save = QAgent(q=goat_q)

        meta = {
            "mode": args.mode,
            "episodes": args.episodes,
            "alpha_start": args.alpha_start,
            "alpha_end": args.alpha_end,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "capture_bonus": args.capture_bonus,
            "max_plies": args.max_plies,
            "seed": args.seed,
            "tiger_wins": result.tiger_wins,
            "goat_wins": result.goat_wins,
            "draws": result.draws,
            "elapsed_seconds": round(result.elapsed, 3),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "loaded_from": args.load,
            "loaded_meta": loaded_meta,
            "preserved_tiger": result.tiger is None and tiger_q is not None,
            "preserved_goat": result.goat is None and goat_q is not None,
        }
        path = save_model(args.save, tiger_to_save, goat_to_save, meta)
        print(f"\nSaved model to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
