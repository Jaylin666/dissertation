# Name-free processed game data

`association_croquet_games_1985_2025_no_names.csv.gz` is the checked
game-level dataset used in the dissertation. It contains 456,382 recorded
games from 1985-2025, including 11,379 games in 2025 and 5,143 unique coded
players.

Player names and direct personal fields have been removed. Stable coded player
identifiers are retained because longitudinal rating histories and the fixed
Player-A orientation depend on them. The original raw source files and the
player-name lookup table are not redistributed.

Load the file directly with pandas:

```python
import pandas as pd

games = pd.read_csv(
    "data/processed/association_croquet_games_1985_2025_no_names.csv.gz",
    low_memory=False,
)
```
