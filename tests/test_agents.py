import random
import tempfile
import unittest
from pathlib import Path

from baghchal_rl.agents import (
    FrozenQAgent,
    QAgent,
    RandomAgent,
    load_model,
    save_model,
)
from baghchal_rl.game import GOAT, TIGER, Baghchal
from baghchal_rl.trainer import (
    TrainConfig,
    evaluate,
    linear_schedule,
    run_episode,
    train,
)
from baghchal_rl.ui import describe_action, parse_action, render


class TestQAgent(unittest.TestCase):
    def test_terminal_update(self):
        agent = QAgent(alpha=0.5, gamma=0.9)
        state = Baghchal().key()
        action = (-1, 12)
        agent.update(state, action, 1.0, done=True)
        self.assertAlmostEqual(agent.q_value(state, action), 0.5)
        agent.update(state, action, 1.0, done=True)
        self.assertAlmostEqual(agent.q_value(state, action), 0.75)

    def test_bootstrap_update(self):
        agent = QAgent(alpha=0.5, gamma=0.9)
        state = Baghchal().key()
        next_state = (b"next", 1, TIGER)
        action = (-1, 12)
        next_actions = [(0, 1), (0, 5)]
        agent._set_q(next_state, (0, 1), 0.4)
        agent._set_q(next_state, (0, 5), 0.8)
        agent.update(state, action, 0.0, next_state, next_actions, done=False)
        # target = 0 + 0.9 * 0.8 = 0.72; q = 0 + 0.5 * 0.72 = 0.36
        self.assertAlmostEqual(agent.q_value(state, action), 0.36)

    def test_best_action_is_greedy(self):
        agent = QAgent()
        state = Baghchal().key()
        actions = [(-1, 12), (-1, 13)]
        agent._set_q(state, (-1, 13), 1.0)
        for _ in range(20):
            self.assertEqual(agent.best_action(state, actions), (-1, 13))

    def test_best_action_random_tie_break(self):
        agent = QAgent()
        state = Baghchal().key()
        actions = [(-1, 12), (-1, 13)]
        chosen = {agent.best_action(state, actions, random.Random(i)) for i in range(50)}
        self.assertEqual(chosen, set(actions))

    def test_choose_returns_legal_action(self):
        agent = QAgent(epsilon=1.0)
        game = Baghchal()
        action = agent.choose(game, rng=random.Random(1))
        self.assertIn(action, game.legal_actions())

    def test_unknown_state_is_zero(self):
        agent = QAgent()
        self.assertEqual(agent.q_value((b"x", 0, GOAT), (-1, 0)), 0.0)
        self.assertEqual(agent.max_q((b"x", 0, GOAT), []), 0.0)

    def test_num_entries_skips_zero_values(self):
        agent = QAgent()
        state = Baghchal().key()
        agent._set_q(state, (-1, 12), 0.0)
        self.assertEqual(agent.num_entries(), 0)
        agent._set_q(state, (-1, 12), 1.0)
        self.assertEqual(agent.num_entries(), 1)

    def test_frozen_agent_does_not_update(self):
        agent = FrozenQAgent()
        state = Baghchal().key()
        agent.update(state, (-1, 12), 1.0, done=True)
        self.assertEqual(agent.num_entries(), 0)
        self.assertEqual(agent.q, {})

    def test_frozen_agent_chooses_greedily(self):
        agent = FrozenQAgent(epsilon=1.0)
        state = Baghchal().key()
        agent._set_q(state, (-1, 13), 1.0)
        game = Baghchal()
        for _ in range(20):
            self.assertEqual(agent.choose(game, epsilon=1.0), (-1, 13))


class TestModelPersistence(unittest.TestCase):
    def test_round_trip(self):
        tiger = QAgent()
        goat = QAgent()
        state = Baghchal().key()
        tiger._set_q(state, (-1, 12), 0.5)
        goat._set_q(state, (0, 1), -0.5)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            save_model(path, tiger, goat, {"mode": "test"})
            loaded_tiger, loaded_goat, meta = load_model(path)

        self.assertEqual(loaded_tiger, tiger.q)
        self.assertEqual(loaded_goat, goat.q)
        self.assertEqual(meta["mode"], "test")

    def test_rejects_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pkl"
            path.write_bytes(b"not a pickle")
            with self.assertRaises(Exception):
                load_model(path)


class TestTrainer(unittest.TestCase):
    def test_linear_schedule(self):
        self.assertAlmostEqual(linear_schedule(1.0, 0.0, 1, 11), 1.0)
        self.assertAlmostEqual(linear_schedule(1.0, 0.0, 11, 11), 0.0)
        self.assertAlmostEqual(linear_schedule(1.0, 0.0, 6, 11), 0.5)
        self.assertAlmostEqual(linear_schedule(1.0, 0.0, 1, 1), 0.0)

    def test_episode_with_random_agents(self):
        rng = random.Random(7)
        game = Baghchal(max_plies=200)
        winner, plies = run_episode(
            game,
            {
                TIGER: RandomAgent(seed=1),
                GOAT: RandomAgent(seed=2),
            },
            rng=rng,
        )
        self.assertIn(winner, {TIGER, GOAT, None})
        self.assertGreater(plies, 0)
        self.assertTrue(game.is_terminal())

    def test_episode_with_learning(self):
        rng = random.Random(8)
        tiger = QAgent(alpha=0.2, epsilon=1.0, seed=1)
        goat = QAgent(alpha=0.2, epsilon=1.0, seed=2)
        run_episode(
            Baghchal(max_plies=100),
            {TIGER: tiger, GOAT: goat},
            epsilon=1.0,
            rng=rng,
            learn=True,
        )
        self.assertGreater(tiger.num_entries(), 0)
        self.assertGreater(goat.num_entries(), 0)

    def test_training_smoke(self):
        config = TrainConfig(
            mode="self-play",
            episodes=3,
            epsilon_start=1.0,
            epsilon_end=1.0,
            log=False,
            max_plies=100,
        )
        result = train(config)
        self.assertIsNotNone(result.tiger)
        self.assertIsNotNone(result.goat)
        self.assertGreater(result.tiger.num_entries(), 0)
        self.assertGreater(result.goat.num_entries(), 0)
        self.assertEqual(result.episodes, 3)

    def test_training_single_side_with_frozen_opponent(self):
        frozen_tiger = {Baghchal().key(): {(-1, 12): 1.0}}
        config = TrainConfig(
            mode="goat",
            episodes=2,
            epsilon_start=1.0,
            epsilon_end=1.0,
            log=False,
            max_plies=30,
            tiger_q=frozen_tiger,
        )
        result = train(config)
        self.assertIsNone(result.tiger)
        self.assertIsNotNone(result.goat)
        # The frozen opponent's table must not be modified.
        self.assertEqual(frozen_tiger, {Baghchal().key(): {(-1, 12): 1.0}})

    def test_evaluate_counts(self):
        stats = evaluate(
            {
                TIGER: RandomAgent(seed=1),
                GOAT: RandomAgent(seed=2),
            },
            episodes=5,
            seed=3,
            max_plies=100,
        )
        self.assertEqual(
            stats["tiger_wins"] + stats["goat_wins"] + stats["draws"], 5
        )
        self.assertGreater(stats["avg_plies"], 0)


class TestUI(unittest.TestCase):
    def test_parse_action_forms(self):
        self.assertEqual(parse_action("5"), (-1, 5))
        self.assertEqual(parse_action("place 5"), (-1, 5))
        self.assertEqual(parse_action("3 7"), (3, 7))
        self.assertEqual(parse_action("3->7"), (3, 7))
        self.assertEqual(parse_action("move 3 7"), (3, 7))
        self.assertIsNone(parse_action("nonsense"))

    def test_render(self):
        text = render(Baghchal())
        self.assertIn("Turn: Goats", text)
        self.assertIn("T = tiger", text)

    def test_describe_capture(self):
        board = [0] * 25
        for pos in (0, 4, 20, 24):
            board[pos] = TIGER
        board[1] = GOAT
        game = Baghchal(board, goats_placed=1, current_player=TIGER)
        self.assertIn("captures the goat at 1", describe_action(game, (0, 2)))


if __name__ == "__main__":
    unittest.main()
