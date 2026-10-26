"""Small terminal UI helpers shared by the play script and tests."""

from __future__ import annotations

from typing import Optional

from .game import (
    CAPTURE_CONNECTIONS,
    GOAT,
    TIGER,
    TOTAL_GOATS,
    Action,
    Baghchal,
)


def piece_name(piece: int) -> str:
    if piece == TIGER:
        return "Tiger"
    if piece == GOAT:
        return "Goat"
    return "Empty"


def player_name(player: int) -> str:
    return "Tigers" if player == TIGER else "Goats"


def render(game: Baghchal) -> str:
    """Return a text rendering of the board and status."""
    lines: list[str] = []
    lines.append(
        f"Turn: {player_name(game.current_player):<7} "
        f"Phase: {game.phase:<9} Plies: {game.plies}"
    )
    lines.append(
        f"Goats placed: {game.goats_placed}/{TOTAL_GOATS}   "
        f"Goats alive: {game.goats_alive}   "
        f"Captured: {game.goats_captured}"
    )
    lines.append("")
    lines.append("       " + "   ".join(str(column) for column in range(5)))
    lines.append("    +" + "---+" * 5)

    for row in range(5):
        cells: list[str] = []
        for column in range(5):
            piece = game.board[row * 5 + column]
            if piece == TIGER:
                cells.append(" T ")
            elif piece == GOAT:
                cells.append(" g ")
            else:
                cells.append(" . ")
        lines.append(f" {row}  |" + "|".join(cells) + "|")
        lines.append("    +" + "---+" * 5)

    lines.append("")
    lines.append("T = tiger, g = goat, . = empty. Position = row * 5 + column.")
    return "\n".join(lines)


def describe_action(game: Baghchal, action: Action) -> str:
    """Describe an action using the state *before* it is applied."""
    src, dst = action
    if src == -1:
        return f"place a goat at {dst}"

    piece = game.board[src]
    if (
        piece == TIGER
        and dst in CAPTURE_CONNECTIONS[src]
        and game.board[(src + dst) // 2] == GOAT
    ):
        return (
            f"tiger {src}->{dst} "
            f"(captures the goat at {(src + dst) // 2})"
        )
    return f"{piece_name(piece).lower()} {src}->{dst}"


def format_action(game: Baghchal, action: Action) -> str:
    """Short form used when listing legal moves."""
    src, dst = action
    if src == -1:
        return f"place {dst}"
    piece = game.board[src]
    if (
        piece == TIGER
        and dst in CAPTURE_CONNECTIONS[src]
        and game.board[(src + dst) // 2] == GOAT
    ):
        return f"{src}->{dst} (capture)"
    return f"{src}->{dst}"


def legal_action_lines(game: Baghchal) -> list[str]:
    return [format_action(game, action) for action in game.legal_actions()]


def help_text() -> str:
    return "\n".join(
        [
            "Commands:",
            "  5            place a goat on position 5 (placement phase)",
            "  3 7          move the piece on 3 to 7",
            "  3->7         same as above",
            "  place 5      explicit placement",
            "  move 3 7     explicit move",
            "  moves        list all legal actions",
            "  board        redraw the board",
            "  help         show this help",
            "  q            quit",
        ]
    )


def parse_action(text: str) -> Optional[Action]:
    """Parse user input into an action tuple, or return ``None``.

    Accepted forms include ``5``, ``place 5``, ``3 7``, ``3->7`` and
    ``move 3 7``. The action may still be illegal; the caller should check.
    """
    cleaned = text.strip().lower().replace("->", " ")
    if not cleaned:
        return None

    parts = cleaned.split()
    try:
        if len(parts) == 1:
            return (-1, int(parts[0]))
        if len(parts) == 2 and parts[0] in {"place", "p"}:
            return (-1, int(parts[1]))
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        if len(parts) == 3 and parts[0] in {"move", "m", "capture", "c"}:
            return (int(parts[1]), int(parts[2]))
    except ValueError:
        return None
    return None


def short_state(game: Baghchal) -> str:
    """A compact one-line state summary for logs."""
    return (
        f"{'T' if game.current_player == TIGER else 'G'} "
        f"g={game.goats_alive} captured={game.goats_captured} "
        f"plies={game.plies}"
    )


__all__ = [
    "piece_name",
    "player_name",
    "render",
    "describe_action",
    "format_action",
    "legal_action_lines",
    "help_text",
    "parse_action",
    "short_state",
]
