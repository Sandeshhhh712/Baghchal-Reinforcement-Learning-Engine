from enum import Enum
from typing import Optional


class PlayerEnum(Enum):
    T = 0
    G = 1


class Board:
    # valid movesets
    _move_connections = {
        0: [1, 5, 6],
        1: [2, 0, 6],
        2: [3, 1, 7, 6, 8],
        3: [4, 2, 8],
        4: [3, 9, 8],
        5: [6, 10, 0],
        6: [7, 5, 11, 1, 10, 2, 12, 0],
        7: [8, 6, 12, 2],
        8: [9, 7, 13, 3, 12, 4, 14, 2],
        9: [8, 14, 4],
        10: [11, 15, 5, 6, 16],
        11: [12, 10, 16, 6],
        12: [13, 11, 17, 7, 16, 8, 18, 6],
        13: [14, 12, 18, 8],
        14: [13, 19, 9, 18, 8],
        15: [16, 20, 10],
        16: [17, 15, 21, 11, 20, 12, 22, 10],
        17: [18, 16, 22, 12],
        18: [19, 17, 23, 13, 22, 14, 24, 12],
        19: [18, 24, 14],
        20: [21, 15, 16],
        21: [22, 20, 16],
        22: [23, 21, 17, 18, 16],
        23: [24, 22, 18],
        24: [23, 19, 18],
    }

    _capture_connections = {
        0: [2, 10, 12],
        1: [3, 11],
        2: [4, 0, 12, 10, 14],
        3: [1, 13],
        4: [2, 14, 12],
        5: [7, 15],
        6: [8, 16, 18],
        7: [9, 5, 17],
        8: [6, 18, 16],
        9: [7, 19],
        10: [12, 20, 0, 2, 22],
        11: [13, 21, 1],
        12: [14, 10, 22, 2, 20, 4, 24, 0],
        13: [11, 23, 3],
        14: [12, 24, 4, 22, 2],
        15: [17, 5],
        16: [18, 6, 8],
        17: [19, 15, 7],
        18: [16, 8, 6],
        19: [17, 9],
        20: [22, 10, 12],
        21: [23, 11],
        22: [24, 20, 12, 14, 10],
        23: [21, 13],
        24: [22, 14, 12],
    }

    def __init__(self):
        # board state initialization
        self.state: list[Optional[PlayerEnum]] = [None] * 25

        self.state[0] = PlayerEnum.T
        self.state[4] = PlayerEnum.T
        self.state[20] = PlayerEnum.T
        self.state[24] = PlayerEnum.T

    def display(self):
        # Using number based index instead of a matrix
        print("\nBoard\n")

        for row in range(25):
            cell = self.state[row]
            if cell is None:
                piece = "."
            else:
                piece = cell.name
            print(f"{piece:2}", end=" ")

            if (row + 1) % 5 == 0:
                print()

        print()

    def valid(self, pos):

        return 0 <= pos < 25

    def is_empty(self, pos):

        return self.valid(pos) and self.state[pos] is None

    def place(self, pos):

        if not self.valid(pos):
            return False

        if not self.is_empty(pos):
            return False

        self.state[pos] = PlayerEnum.G

        return True

    def move(self, src, dst, player):

        if not (self.valid(src) and self.valid(dst)):
            return False

        if self.state[src] != player:
            return False

        if dst not in self._move_connections[src]:
            return False

        if not self.is_empty(dst):
            return False

        self.state[src] = None
        self.state[dst] = player

        return True

    def capture(self, src, dst):

        if not (self.valid(src) and self.valid(dst)):
            return False

        if self.state[src] != PlayerEnum.T:
            return False

        if dst not in self._capture_connections[src]:
            return False

        if not self.is_empty(dst):
            return False

        middle = (src + dst) // 2

        if self.state[middle] != PlayerEnum.G:
            return False

        self.state[src] = None
        self.state[middle] = None
        self.state[dst] = PlayerEnum.T

        return True

    def tiger_has_move(self, pos):
        # depedency
        for i in self._move_connections[pos]:
            if self.is_empty(i):
                return True

        for i in self._capture_connections[pos]:

            middle = (pos + i) // 2

            if self.state[middle] == PlayerEnum.G and self.is_empty(i):
                return True

        return False

    def trapped_tigers(self):

        trapped = set()

        for i in range(25):

            if self.state[i] == PlayerEnum.T:

                if not self.tiger_has_move(i):
                    trapped.add(i)

        return trapped


def print_status(goats_alive, goats_placed, goat_total, board, active):

    print("------ Game Status ------")

    print(f"Turn: {active.name}")

    phase = "PLACEMENT" if goats_placed < goat_total else "MOVEMENT"

    print(f"Phase: {phase}")

    print(f"Goats placed: {goats_placed}/{goat_total}")

    print(f"Goats alive: {goats_alive}")

    print(f"Goats captured: {goat_total - goats_alive}")

    trapped = board.trapped_tigers()

    print(f"Tigers trapped: {len(trapped)}")

    print("-------------------------\n")


def main():

    board = Board()

    goat_total = 20
    goats_placed = 0
    goats_alive = 20

    active = PlayerEnum.G

    print("\nBaghchal\n")

    board.display()

    while True:

        print_status(goats_alive, goats_placed, goat_total, board, active)

        trapped = board.trapped_tigers()

        if len(trapped) == 4:
            print("Goats win")
            break

        if goats_alive <= 15:
            print("Tigers win")
            break

        if active == PlayerEnum.G and goats_placed < goat_total:

            user = input("Place goat (0-24) or q: ")

            if user == "q":
                break

            try:

                pos = int(user)

                if board.place(pos):

                    goats_placed += 1

                    board.display()

                    active = PlayerEnum.T

                else:

                    print("Illegal placement")

            except:

                print("Invalid input")

            continue

        user = input("Move src dst or q: ")

        if user == "q":
            break

        try:

            src, dst = map(int, user.split())

        except:

            print("Invalid input")

            continue

        if active == PlayerEnum.T:

            if board.capture(src, dst):

                goats_alive -= 1

                print("Tiger captured a goat!")

                board.display()

                active = PlayerEnum.G

                continue

        if board.move(src, dst, active):

            board.display()

            active = PlayerEnum.T if active == PlayerEnum.G else PlayerEnum.G

        else:

            print("Illegal move")


if __name__ == "__main__":

    main()
