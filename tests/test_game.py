import unittest

from baghchal_rl.game import (
    CAPTURES_TO_WIN,
    EMPTY,
    GOAT,
    TIGER,
    TOTAL_GOATS,
    Baghchal,
)


def boarded(tigers=(0, 4, 20, 24), goats=(), empty=(), goats_placed=0, player=GOAT,
            max_plies=200):
    board = [EMPTY] * 25
    for pos in tigers:
        board[pos] = TIGER
    for pos in goats:
        board[pos] = GOAT
    for pos in empty:
        board[pos] = EMPTY
    return Baghchal(
        board,
        goats_placed=goats_placed,
        current_player=player,
        max_plies=max_plies,
    )


class TestSetup(unittest.TestCase):
    def test_initial_state(self):
        game = Baghchal()
        self.assertEqual(game.current_player, GOAT)
        self.assertEqual(game.goats_placed, 0)
        self.assertEqual(game.goats_alive, TOTAL_GOATS)
        self.assertEqual(game.board.count(TIGER), 4)
        self.assertEqual(game.board.count(GOAT), 0)
        self.assertEqual(game.phase, "placement")
        self.assertEqual(len(game.legal_actions()), 21)
        self.assertFalse(game.is_terminal())
        self.assertIsNone(game.winner())

    def test_goat_placement_switches_turn(self):
        game = Baghchal()
        action = (-1, 12)
        self.assertTrue(game.is_legal(action))
        game.apply(action)
        self.assertEqual(game.board[12], GOAT)
        self.assertEqual(game.goats_placed, 1)
        self.assertEqual(game.current_player, TIGER)
        self.assertFalse(game.is_legal((-1, 11)))
        self.assertEqual(game.plies, 1)

    def test_goat_cannot_move_before_placement_is_done(self):
        game = boarded(goats=(12,), goats_placed=1, player=GOAT)
        self.assertFalse(game.is_legal((12, 11)))
        with self.assertRaises(ValueError):
            game.apply((12, 11))
        # Only placements are available.
        self.assertTrue(all(action[0] == -1 for action in game.legal_actions()))

    def test_goat_movement_phase(self):
        board = [GOAT] * 25
        for pos in (0, 4, 20, 24):
            board[pos] = TIGER
        board[6] = EMPTY
        game = Baghchal(board, goats_placed=20, current_player=GOAT)
        self.assertEqual(game.phase, "movement")
        self.assertTrue(game.is_legal((1, 6)))
        game.apply((1, 6))
        self.assertEqual(game.board[1], EMPTY)
        self.assertEqual(game.board[6], GOAT)
        self.assertEqual(game.current_player, TIGER)


class TestTigers(unittest.TestCase):
    def test_simple_tiger_move(self):
        game = boarded(player=TIGER)
        self.assertTrue(game.is_legal((0, 1)))
        game.apply((0, 1))
        self.assertEqual(game.board[0], EMPTY)
        self.assertEqual(game.board[1], TIGER)
        self.assertEqual(game.goats_captured, 0)
        self.assertEqual(game.current_player, GOAT)

    def test_tiger_capture(self):
        game = boarded(goats=(1,), goats_placed=1, player=TIGER)
        self.assertIn((0, 2), game.legal_actions())
        game.apply((0, 2))
        self.assertEqual(game.board[0], EMPTY)
        self.assertEqual(game.board[1], EMPTY)
        self.assertEqual(game.board[2], TIGER)
        self.assertEqual(game.goats_captured, 1)
        self.assertEqual(game.goats_alive, 19)

    def test_tiger_cannot_jump_empty_or_tiger(self):
        game = boarded(player=TIGER)
        self.assertFalse(game.is_legal((0, 2)))
        with self.assertRaises(ValueError):
            game.apply((0, 2))

    def test_tiger_cannot_move_to_occupied_square(self):
        game = boarded(goats=(1,), goats_placed=1, player=TIGER)
        with self.assertRaises(ValueError):
            game.apply((0, 1))

    def test_tiger_wins_after_five_captures(self):
        game = Baghchal(goats_placed=CAPTURES_TO_WIN)
        self.assertEqual(game.goats_captured, CAPTURES_TO_WIN)
        self.assertEqual(game.winner(), TIGER)
        self.assertTrue(game.is_terminal())

    def test_capture_list_matches_two_step_jumps(self):
        # Every action two steps away over a goat must be generated.
        game = boarded(goats=(1,), goats_placed=1, player=TIGER)
        self.assertIn((0, 2), game.legal_actions())

    def test_goats_are_not_generated_as_capture_actions(self):
        only_tigers = boarded(player=TIGER)
        self.assertTrue(all(action[0] != -1 for action in only_tigers.legal_actions()))


class TestTerminalConditions(unittest.TestCase):
    def test_goats_win_when_tigers_are_trapped(self):
        board = [GOAT] * 25
        for pos in (0, 4, 20, 24):
            board[pos] = TIGER
        board[7] = EMPTY
        game = Baghchal(board, goats_placed=20, current_player=TIGER)
        self.assertEqual(game.winner(), GOAT)
        self.assertTrue(game.is_terminal())

    def test_tigers_win_when_goats_have_no_move(self):
        # Tigers at 1,4,5,6 surround the only empty point (0). No goat can
        # move into it, so the goat player has no legal move.
        board = [GOAT] * 25
        for pos in (1, 4, 5, 6):
            board[pos] = TIGER
        board[0] = EMPTY
        game = Baghchal(board, goats_placed=20, current_player=GOAT)
        self.assertEqual(game.legal_actions(), [])
        self.assertEqual(game.winner(), TIGER)

    def test_draw_at_move_limit(self):
        game = Baghchal(max_plies=5, plies=5)
        self.assertIsNone(game.winner())
        self.assertTrue(game.is_terminal())

    def test_no_draw_without_limit(self):
        game = Baghchal(max_plies=None, plies=1000)
        self.assertFalse(game.is_terminal())


class TestStateUtilities(unittest.TestCase):
    def test_key_changes_after_action(self):
        game = Baghchal()
        before = game.key()
        game.apply((-1, 12))
        self.assertNotEqual(before, game.key())

    def test_clone_is_independent(self):
        game = Baghchal()
        clone = game.clone()
        clone.apply((-1, 12))
        self.assertEqual(game.board[12], EMPTY)
        self.assertEqual(clone.board[12], GOAT)
        self.assertEqual(game.goats_placed, 0)
        self.assertEqual(clone.goats_placed, 1)

    def test_apply_on_terminal_game_raises(self):
        game = Baghchal(goats_placed=CAPTURES_TO_WIN)
        with self.assertRaises(ValueError):
            game.apply((0, 1))

    def test_invalid_board_size(self):
        with self.assertRaises(ValueError):
            Baghchal([EMPTY] * 24)

    def test_goats_placed_must_cover_goats_on_board(self):
        board = [EMPTY] * 25
        board[12] = GOAT
        with self.assertRaises(ValueError):
            Baghchal(board, goats_placed=0)


if __name__ == "__main__":
    unittest.main()
