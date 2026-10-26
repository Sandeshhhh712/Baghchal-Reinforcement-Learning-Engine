"""Core Baghchal rules and state.

The board topology is imported from the original command line game in
``board.py`` so the RL engine and the CLI agree on legal moves.

Actions are two-tuples:

* ``(-1, position)`` places a goat on ``position`` during the placement phase.
* ``(source, destination)`` moves the piece on ``source`` to ``destination``.
  A tiger move that jumps two points over a goat is a capture.
"""

from __future__ import annotations

from typing import Optional

from board import Board as _LegacyBoard

EMPTY = 0
TIGER = 1
GOAT = 2

TOTAL_GOATS = 20
CAPTURES_TO_WIN = 5
DEFAULT_MAX_PLIES = 200

# Reuse the exact adjacency used by the existing CLI game.
MOVE_CONNECTIONS: dict[int, list[int]] = _LegacyBoard._move_connections
CAPTURE_CONNECTIONS: dict[int, list[int]] = _LegacyBoard._capture_connections

Action = tuple[int, int]
StateKey = tuple[bytes, int, int]


def action_text(action: Action) -> str:
    """Return a compact human-readable form of an action."""
    src, dst = action
    if src == -1:
        return f"place {dst}"
    return f"{src}->{dst}"


class Baghchal:
    """A mutable Baghchal game state.

    ``board`` is a 25-element list using ``EMPTY``, ``TIGER`` and ``GOAT``.
    Goats place first. Once all 20 goats have been placed, goats move one
    step along a connection. Tigers may either move one step or jump over a
    single goat to an empty point.

    The game ends when five goats have been captured (tigers win), when no
    tiger has a legal move (goats win), or when the current player has no
    legal move (that player loses). A move-limit draw can be configured with
    ``max_plies``.
    """

    __slots__ = ("board", "goats_placed", "current_player", "plies", "max_plies")

    def __init__(
        self,
        board: Optional[list[int]] = None,
        goats_placed: int = 0,
        current_player: int = GOAT,
        plies: int = 0,
        max_plies: Optional[int] = DEFAULT_MAX_PLIES,
    ) -> None:
        if board is None:
            self.board = [EMPTY] * 25
            for pos in (0, 4, 20, 24):
                self.board[pos] = TIGER
        else:
            self.board = list(board)

        if len(self.board) != 25:
            raise ValueError("A Baghchal board must have 25 positions.")
        if current_player not in (TIGER, GOAT):
            raise ValueError("current_player must be TIGER or GOAT.")
        if not 0 <= goats_placed <= TOTAL_GOATS:
            raise ValueError("goats_placed must be between 0 and 20.")
        if self.board.count(GOAT) > goats_placed:
            raise ValueError(
                "goats_placed cannot be smaller than the number of goats on the board."
            )

        self.goats_placed = goats_placed
        self.current_player = current_player
        self.plies = plies
        self.max_plies = max_plies

    # ------------------------------------------------------------------
    # State information
    # ------------------------------------------------------------------
    @property
    def goats_on_board(self) -> int:
        return self.board.count(GOAT)

    @property
    def goats_captured(self) -> int:
        return self.goats_placed - self.goats_on_board

    @property
    def goats_alive(self) -> int:
        return TOTAL_GOATS - self.goats_captured

    @property
    def goats_to_place(self) -> int:
        return TOTAL_GOATS - self.goats_placed

    @property
    def phase(self) -> str:
        return "placement" if self.goats_placed < TOTAL_GOATS else "movement"

    def key(self) -> StateKey:
        """Hashable state key used by the Q-table.

        ``goats_placed`` matters because it decides whether goats may place
        or must move, and it lets us derive how many goats have been
        captured. The side to move is included as well.
        """
        return (bytes(self.board), self.goats_placed, self.current_player)

    def clone(self) -> "Baghchal":
        return Baghchal(
            self.board,
            self.goats_placed,
            self.current_player,
            self.plies,
            self.max_plies,
        )

    def __repr__(self) -> str:
        return (
            f"Baghchal(goats_placed={self.goats_placed}, "
            f"current_player={self.current_player}, plies={self.plies})"
        )

    # ------------------------------------------------------------------
    # Legal moves
    # ------------------------------------------------------------------
    def legal_actions(self) -> list[Action]:
        """Return every legal action for the player to move."""
        board = self.board
        actions: list[Action] = []

        if self.current_player == GOAT:
            if self.goats_placed < TOTAL_GOATS:
                for pos, piece in enumerate(board):
                    if piece == EMPTY:
                        actions.append((-1, pos))
                return actions

            for src, piece in enumerate(board):
                if piece != GOAT:
                    continue
                for dst in MOVE_CONNECTIONS[src]:
                    if board[dst] == EMPTY:
                        actions.append((src, dst))
            return actions

        # Tiger: simple moves first, then captures.
        for src, piece in enumerate(board):
            if piece != TIGER:
                continue
            for dst in MOVE_CONNECTIONS[src]:
                if board[dst] == EMPTY:
                    actions.append((src, dst))
            for dst in CAPTURE_CONNECTIONS[src]:
                if board[dst] == EMPTY and board[(src + dst) // 2] == GOAT:
                    actions.append((src, dst))

        return actions

    def is_legal(self, action: Action) -> bool:
        return action in self.legal_actions()

    def _tigers_have_move(self) -> bool:
        board = self.board
        for src, piece in enumerate(board):
            if piece != TIGER:
                continue
            for dst in MOVE_CONNECTIONS[src]:
                if board[dst] == EMPTY:
                    return True
            for dst in CAPTURE_CONNECTIONS[src]:
                if board[dst] == EMPTY and board[(src + dst) // 2] == GOAT:
                    return True
        return False

    def _goats_have_move(self) -> bool:
        if self.goats_placed < TOTAL_GOATS and EMPTY in self.board:
            return True

        board = self.board
        for src, piece in enumerate(board):
            if piece != GOAT:
                continue
            for dst in MOVE_CONNECTIONS[src]:
                if board[dst] == EMPTY:
                    return True
        return False

    # ------------------------------------------------------------------
    # Game result
    # ------------------------------------------------------------------
    def winner(self) -> Optional[int]:
        """Return ``TIGER``, ``GOAT`` or ``None`` if the game is not over."""
        if self.goats_captured >= CAPTURES_TO_WIN:
            return TIGER

        # Goats win by removing every legal tiger move, no matter whose
        # turn it is.
        if not self._tigers_have_move():
            return GOAT

        # A player with no legal move loses. This only matters for goats;
        # the tiger case is covered by the check above.
        if self.current_player == GOAT and not self._goats_have_move():
            return TIGER

        return None

    def is_terminal(self) -> bool:
        if self.winner() is not None:
            return True
        return self.max_plies is not None and self.plies >= self.max_plies

    # ------------------------------------------------------------------
    # Applying moves
    # ------------------------------------------------------------------
    def apply(self, action: Action) -> None:
        """Apply a legal action, mutating the state.

        Raises ``ValueError`` if the action is not legal in this state.
        """
        if self.is_terminal():
            raise ValueError("Cannot apply an action to a finished game.")

        src, dst = action
        if not 0 <= dst < 25:
            raise ValueError(f"Invalid destination position: {dst}")

        board = self.board

        if src == -1:
            if self.current_player != GOAT:
                raise ValueError("Only goats may place a piece.")
            if self.goats_placed >= TOTAL_GOATS:
                raise ValueError("All 20 goats have already been placed.")
            if board[dst] != EMPTY:
                raise ValueError(f"Position {dst} is not empty.")

            board[dst] = GOAT
            self.goats_placed += 1
        else:
            if not 0 <= src < 25:
                raise ValueError(f"Invalid source position: {src}")

            piece = board[src]
            if piece != self.current_player:
                raise ValueError(
                    "The source position does not contain the current player's piece."
                )
            if board[dst] != EMPTY:
                raise ValueError(f"Position {dst} is not empty.")

            if piece == TIGER:
                if dst in CAPTURE_CONNECTIONS[src] and board[(src + dst) // 2] == GOAT:
                    # Remove the jumped goat.
                    board[(src + dst) // 2] = EMPTY
                elif dst not in MOVE_CONNECTIONS[src]:
                    raise ValueError(f"Illegal tiger move {src}->{dst}.")
            else:
                if self.goats_placed < TOTAL_GOATS:
                    raise ValueError(
                        "Goats cannot move until all 20 goats have been placed."
                    )
                if dst not in MOVE_CONNECTIONS[src]:
                    raise ValueError(f"Illegal goat move {src}->{dst}.")

            board[src] = EMPTY
            board[dst] = piece

        self.plies += 1
        self.current_player = TIGER if self.current_player == GOAT else GOAT
