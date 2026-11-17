# Valorant Match Win Predictor

A Python machine-learning project that predicts whether a Valorant team will
win a match using recent team and player performance from a VCT DuckDB dataset.

The predictor uses only data available before each match. It does not use the
match being predicted when calculating player form or team win rate.

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment if desired,
then install the project dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Dataset requirements

Pass the path to a DuckDB file that has these two tables:

- `matches`: `match_id`, `utc_timestamp`, `team0_id`, `team1_id`, `score0`, and
  `score1`
- `player_map`: `match_id`, `team_idx`, `player_id`, `acs_all`, `rating_all`,
  `kills_all`, `deaths_all`, `assists_all`, `fk_all`, and `fd_all`

`player_map` is expected to contain one row per player per map. The script
aggregates those records into one row per player per match before training.

## Run the model

```bash
python main.py path/to/vct.duckdb
```

Optional parameters let you tune the history window:

```bash
python main.py path/to/vct.duckdb --lookback 10 --min-history 3
```

- `--lookback`: number of prior matches included in a rolling average (default:
  `10`)
- `--min-history`: prior matches required before a player or team can be used
  (default: `3`)

## How it works

The program turns historical player and team performance into a single training
row for each match. That row answers this question: based on what was known
before this match started, which team was more likely to win?

### 1. Combine map statistics into match statistics

The source data has a player record for every map. Since the prediction target
is the winner of an entire match, `main.py` groups those map records by player
and match. It averages map-based values such as ACS and rating, and totals
kills, deaths, and assists.

It then calculates a few useful derived values:

- **K/D** = kills divided by deaths
- **KDA** = (kills + assists) divided by deaths
- **First-blood net** = first bloods minus first deaths

Deaths are treated as at least one while calculating these ratios, preventing a
division-by-zero error.

### 2. Build pre-match player form

For every player, the script calculates a rolling average over their earlier
matches. By default it uses the previous 10 matches, and requires at least 3
earlier matches before a player has usable history.

The rolling features are ACS, rating, K/D, KDA, first bloods, first deaths, and
first-blood net. For example, a player's `hist_rating` for a match is their
average rating from matches played before that match—not their rating in the
current match.

This is important because using the current match's statistics would leak the
answer into the input features. The code uses a one-match shift before each
rolling average to prevent that leakage.

### 3. Convert player form into team form

The five players' historical feature values are averaged to represent the
team's pre-match form. The script also calculates each team's recent win rate
from its previous matches using the same rolling-window approach.

At this point, every team in a match has a set of features such as:

```text
average recent player rating
average recent player ACS
average recent player K/D and KDA
recent first-blood advantage
recent team win rate
```

### 4. Compare the two teams

A classifier needs one row per match, so the program subtracts Team B's form
from Team A's form. For example:

```text
diff_recent_winrate = Team A recent win rate - Team B recent win rate
diff_hist_rating    = Team A player rating history - Team B player rating history
```

A positive difference means Team A was stronger in that category before the
match. The original team ordering in the dataset is randomly flipped for about
half of the matches. This keeps the model from treating the dataset's
`team_idx = 0` position as meaningful.

### 5. Train on the past and test on the future

The dataset is ordered by match date. The oldest 80% of usable matches are used
for training and the newest 20% are held out for testing. This chronological
split simulates the real use case: training on past events and predicting later
ones.

Feature values are standardized before logistic regression is trained. This
puts measurements with different scales, such as ACS and win rate, on a
comparable scale and makes the model's weights easier to compare.

The target is `team_a_win`:

- `1` means Team A won
- `0` means Team B won

## Output

The program reports:

- a recent-win-rate baseline accuracy
- logistic-regression accuracy
- ROC-AUC when the test set contains both outcomes
- a classification report
- feature weights, where a positive weight favors Team A winning

Because matches are split chronologically, the reported test performance better
approximates predictions on future matches than a random train/test split.

### Reading the metrics

- **Baseline accuracy** selects the team with the better recent win rate. It is
  a simple benchmark the model should try to beat.
- **Model accuracy** is the fraction of held-out matches where the model chose
  the correct winner.
- **ROC-AUC** measures how well predicted win probabilities rank winners ahead
  of losers. `0.5` is roughly random ranking and `1.0` is perfect ranking.
- **Feature weights** show the direction and relative strength of each input.
  A positive `diff_hist_rating` weight, for example, means a Team A rating
  advantage increases the predicted chance that Team A wins.

These values evaluate the held-out portion of the provided dataset; they do
not guarantee the same performance for future tournaments or roster changes.

## Project files

- `main.py` — data preparation, feature engineering, training, and evaluation
- `requirements.txt` — Python dependencies
