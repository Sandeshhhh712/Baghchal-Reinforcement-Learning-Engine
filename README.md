# Baghchal Reinforcement Learning Engine

A small, dependency-free Baghchal (Bagh-chal / Tigers and Goats) rules engine
with tabular Q-learning agents. It reuses the board topology from the original
`board.py` CLI, so the RL engine and the existing game agree on legal moves.

The project intentionally keeps the learning side simple:

- no neural networks,
- no MCTS/search,
- one Q-table per side,
- full board states as table keys.

## Quick start

Python 3.10+ is required. There are no third-party packages.

Train a pair of self-play agents (about 30–45 seconds on a typical laptop):

```bash
python3 train.py --mode self-play --episodes 30000 --capture-bonus 0.5
```

Play against the trained tiger (the default: you are the goats):

```bash
python3 play.py --model models/baghchal_q.pkl --human goat
```

Play against the trained goat:

```bash
python3 play.py --model models/baghchal_q.pkl --human tiger
```

Without a model, `play.py` falls back to a random agent.

## Board and input

Positions are row-major:

```text
 0   1   2   3   4
 5   6   7   8   9
10  11  12  13  14
15  16  17  18  19
20  21  22  23  24
```

During the placement phase, enter a position number to place a goat. During
the movement phase, enter `source destination` (for example `3 7`). Tigers
capture by jumping two points over a goat; the engine recognizes a capture
automatically.

Useful commands inside `play.py`:

```text
moves      list every legal action
board      redraw the board
help       show the input help
q          quit
```

## Training modes

```bash
# Both sides learn against each other.
python3 train.py --mode self-play --episodes 30000

# Only tigers learn; goats play randomly.
python3 train.py --mode tiger --episodes 10000

# Only goats learn; tigers play randomly.
python3 train.py --mode goat --episodes 10000
```

`--load` continues from an existing model. In single-side modes, the loaded
other side is used as a frozen opponent, which makes alternating training easy:

```bash
python3 train.py --mode tiger --episodes 10000 --save models/q.pkl
python3 train.py --mode goat  --episodes 10000 --load models/q.pkl --save models/q.pkl
python3 train.py --mode tiger --episodes 10000 --load models/q.pkl --save models/q.pkl
```

Other useful options:

```text
--alpha-start / --alpha-end     learning-rate schedule
--gamma                         discount factor
--epsilon-start / --epsilon-end exploration schedule
--capture-bonus                 optional reward shaping for captures
--max-plies                     draw limit during training
--eval-episodes                 number of greedy evaluation games
--no-save / --no-eval / --quiet
--seed                          reproducibility
```

Run `python3 train.py --help` for the full list.

## How the Q-table works

### State

A state key is:

```python
(bytes(board), goats_placed, current_player)
```

`board` has 25 entries: empty, tiger, or goat. `goats_placed` determines
whether goats are still placing or have started moving, and it lets the engine
derive how many goats have been captured. The side to move is included for
clarity, although each side has its own Q-table.

### Action

```python
(-1, position)         # place a goat
(source, destination)  # move a piece
```

A tiger move two points away over a goat is a capture. Illegal actions are
never added to the table.

### Reward

- Win: `+1`
- Loss: `-1`
- Draw (move limit): `0`
- With `--capture-bonus B`: tigers get `+B` per captured goat and goats get
  `-B` per captured goat. The default is `0` (no shaping).

### Update

Each learner's transition runs from one of its turns to its next turn. When
the transition finishes, the standard Q-learning update is applied:

```text
Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
```

Terminal transitions use the terminal reward directly. During self-play both
sides update their own tables. Against a random or frozen opponent, only the
learning side updates.

The exploration rate and learning rate decay linearly over training.

## Example results

With `--mode self-play --episodes 30000 --capture-bonus 0.5 --seed 42`, a
greedy evaluation over 500 games looked like this on one machine:

```text
tiger (trained) vs random goat     T  99.8%  G   0.2%  D  0.0%
random tiger vs goat (trained)     T  96.6%  G   3.4%  D  0.0%
tiger (trained) vs goat (trained)  T  98.2%  G   1.8%  D  0.0%
random vs random (baseline)        T  98.8%  G   1.2%  D  0.0%
```

The tiger side is much easier to learn: random goat play loses almost every
game already. Self-play and capture shaping improve the goat side compared to
the random baseline, but a full-board tabular Q-table is very sample-hungry.
Expect many more episodes (or a different representation) for strong goat
play. This is a learning/demo engine, not a solved-game solver.

Known limitations:

- the table only generalizes by exact state match;
- board symmetries are not canonicalized;
- there is no opponent modelling or lookahead;
- the 200-ply training limit can turn long defensive games into draws.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests cover legal move generation, captures, both win conditions, the
no-legal-move rule, Q-learning updates, model save/load, a short training run,
and CLI parsing.

## Project layout

```text
board.py                  original CLI game (unchanged)
baghchal_rl/
  game.py                 rules, state, legal actions, result detection
  agents.py               RandomAgent, QAgent, FrozenQAgent, save/load
  trainer.py              episode runner, training loop, evaluation
  ui.py                   board rendering and input parsing
train.py                  training CLI
play.py                   human-vs-agent CLI
tests/                    unittest suite
models/                   generated Q-table models (git-ignored)
```

The original game still runs independently:

```bash
python3 board.py
```

Model files are Python pickles. Only load models you created yourself; pickle
can execute arbitrary code.
