"""Train and evaluate a Valorant match-win predictor from a VCT DuckDB file.

The model predicts whether a randomly assigned Team A wins a match. Every
feature is based only on matches played before the one being predicted.

Usage: python main.py path/to/vct.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler


LOOKBACK = 10
MIN_HISTORY = 3
TEST_FRACTION = 0.20
RANDOM_SEED = 42
PLAYER_STATS = ("acs", "rating", "kd", "kda", "first_bloods", "first_deaths", "fb_net")

# Add'l feature Elo constants:
# - Every team starts with the same rating of 1500.
# - TEAM_ELO_K controls how much a match result changes the rating.
# - A K value of 32 produces moderate, gradual rating changes.
TEAM_ELO_INITIAL = 1500.0
TEAM_ELO_K = 32.0


# Parse command-line arguments, including the database path and history settings.
def parse_args() -> argparse.Namespace:
    default_database = Path(__file__).parent / "data" / "vct.duckdb"
    #defined default database path as vct.duckdb in the same directory as the script

    parser = argparse.ArgumentParser(description=__doc__)

    #nargs for optional path argument, default to vct.duckdb if not provided
    parser.add_argument("database", type=Path,nargs='?', default=default_database, help="Path to the VCT DuckDB database")
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="Prior matches to average")
    parser.add_argument("--min-history", type=int, default=MIN_HISTORY, help="Prior matches required")
    return parser.parse_args()


# Load match data from DuckDB and aggregate map statistics by player and match.
def load_player_matches(database: Path) -> pd.DataFrame:
    """Aggregate map-level player stats to one player row per match."""
    if not database.is_file():
        raise FileNotFoundError(f"DuckDB file not found: {database}")
    query = """
        SELECT
            m.match_id, m.utc_timestamp AS match_date, pm.team_idx,
            CASE WHEN pm.team_idx = 0 THEN m.team0_id ELSE m.team1_id END AS team_id,
            CAST((pm.team_idx = 0) = (m.score0 > m.score1) AS INTEGER) AS team_won,
            pm.player_id, AVG(pm.acs_all) AS acs, AVG(pm.rating_all) AS rating,
            SUM(pm.kills_all) AS kills, SUM(pm.deaths_all) AS deaths,
            SUM(pm.assists_all) AS assists, AVG(pm.fk_all) AS first_bloods,
            AVG(pm.fd_all) AS first_deaths
        FROM player_map AS pm
        JOIN matches AS m USING (match_id)
        WHERE m.utc_timestamp IS NOT NULL
        GROUP BY ALL
    """
    with duckdb.connect(str(database), read_only=True) as connection:
        players = connection.execute(query).df()
    if players.empty:
        raise ValueError("No player-match rows were returned from the database.")
    return players.sort_values(["match_date", "match_id"]).reset_index(drop=True)


# Calculate derived player statistics such as K/D, KDA, and first-blood difference.
def add_derived_stats(players: pd.DataFrame) -> pd.DataFrame: 
    """Add rate statistics, guarding against zero deaths."""
    players = players.copy()
    deaths = players["deaths"].clip(lower=1)
    players["kd"] = players["kills"] / deaths
    players["kda"] = (players["kills"] + players["assists"]) / deaths
    players["fb_net"] = players["first_bloods"] - players["first_deaths"]
    return players


# Calculate rolling averages using only matches before the current match.
def past_rolling_average(values: pd.Series, lookback: int, min_history: int) -> pd.Series:
    """Average preceding observations only; shift prevents target leakage."""
    return values.shift(1).rolling(lookback, min_periods=min_history).mean()


# Add historical performance features for every player.
def add_player_history(players: pd.DataFrame, lookback: int, min_history: int) -> pd.DataFrame:
    players = players.copy()
    for stat in PLAYER_STATS:
        players[f"hist_{stat}"] = players.groupby("player_id", sort=False)[stat].transform(
            lambda values: past_rolling_average(values, lookback, min_history)
        )
    return players


# Aggregate individual player histories into team-level statistics.
def build_team_rows(players: pd.DataFrame, lookback: int, min_history: int) -> pd.DataFrame:
    """Reduce player history to team-level features for each match."""
    history_columns = [f"hist_{stat}" for stat in PLAYER_STATS]

    teams = (
        players.groupby(
            ["match_id", "match_date", "team_idx", "team_id"],
            as_index=False,
        )
        .agg(
            **{column: (column, "mean") for column in history_columns},
            team_won=("team_won", "first"),
            lineup=("player_id", lambda values: frozenset(values.dropna().unique())),
        )
        .sort_values(["match_date", "match_id", "team_idx"])
        .reset_index(drop=True)
    )

    # Existing team win rate remains a team-level feature. This uses a shift
    # internally, so the result for each match uses only previous matches.
    teams["recent_winrate"] = teams.groupby("team_id", sort=False)["team_won"].transform(
        lambda values: past_rolling_average(values, lookback, min_history)
    )

    # Elo changes below are match outcomes that occur after both teams receive
    # their team_elo value. The following match will observe the new rating.
    
    # Elo must be updated by match date. Sorting the match IDs here ensures
    # ratings are created only from matches that occurred earlier.
    # Elo must be precoessed in chronological match order for feature to work
    ordered_matches = (
        teams[["match_id", "match_date"]]
        .drop_duplicates()
        .sort_values(["match_date", "match_id"])
    )

    # Elo ratings are stored per team ID. Teams not seen before start at 1500.
    # previous_lineups tracks consecutive rosters for roster continuity.
    elo = {}
    previous_lineups = {}

    # Elo is written before each match outcome is applied. Therefore, the
    # feature will never contain information from the match being predicted.
    teams["team_elo"] = TEAM_ELO_INITIAL
    teams["roster_continuity"] = 1.0

    for match_id in ordered_matches["match_id"]:
        match = teams.loc[teams["match_id"] == match_id]
        rows = match.set_index("team_idx")

        # Elo requires exactly one Team 0 row and one Team 1 row.
        if set(rows.index) != {0, 1}:
            continue

        # Assign Elo and roster continuity before processing the match outcome.
        # Neither feature is allowed to use the current match as history.
        for team_idx in (0, 1):
            row = rows.loc[team_idx]

            # This is the original row index, not team_idx.
            team_row_index = match.index[match["team_idx"] == team_idx][0]

            team_id = int(row["team_id"])
            elo.setdefault(team_id, TEAM_ELO_INITIAL)

            teams.loc[team_row_index, "team_elo"] = elo[team_id]

            previous_lineup = previous_lineups.get(team_id)
            current_lineup = row["lineup"]

            if previous_lineup is None:
                continuity = 1.0
            else:
                union = previous_lineup | current_lineup
                continuity = (
                    len(previous_lineup & current_lineup) / len(union)
                    if union
                    else 1.0
                )

            teams.loc[team_row_index, "roster_continuity"] = continuity
            previous_lineups[team_id] = current_lineup

        team0 = rows.loc[0]
        team1 = rows.loc[1]
        team0_id = int(team0["team_id"])
        team1_id = int(team1["team_id"])

        team0_elo = elo[team0_id]
        team1_elo = elo[team1_id]

        # Convert Elo difference into the probability that Team 0 wins.
        team0_expected = 1.0 / (
            1.0 + 10 ** ((team1_elo - team0_elo) / 400.0)
        )
        team1_expected = 1.0 - team0_expected

        team0_score = float(team0["team_won"])
        team1_score = float(team1["team_won"])

        # Update ratings only after assigning the pre-match values above.
        elo[team0_id] = (
            team0_elo
            + TEAM_ELO_K * (team0_score - team0_expected)
        )
        elo[team1_id] = (
            team1_elo
            + TEAM_ELO_K * (team1_score - team1_expected)
        )
        #end of ELO chunk for calculation of team Elo based on match outcome from inital elo given

    return teams


# Convert two team rows into one match row using Team A minus Team B differences.
def build_match_rows(teams: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Create match-level differences: Team A feature minus Team B feature."""
    team0 = teams.loc[teams["team_idx"] == 0].set_index("match_id")
    team1 = teams.loc[teams["team_idx"] == 1].set_index("match_id")
    match_ids = team0.index.intersection(team1.index)
    team0, team1 = team0.loc[match_ids], team1.loc[match_ids]
    rng = np.random.default_rng(RANDOM_SEED)
    flip = rng.random(len(match_ids)) < 0.5
    columns = feature_names + ["team_won"]
    team_a, team_b = team0[columns].copy(), team1[columns].copy()
    team_a.loc[flip], team_b.loc[flip] = team1.loc[flip, columns].to_numpy(), team0.loc[flip, columns].to_numpy()

    matches = pd.DataFrame({"match_date": team0["match_date"], "team_a_win": team_a["team_won"]})
    for feature in feature_names:
        matches[f"diff_{feature}"] = team_a[feature] - team_b[feature]
    return matches.dropna().sort_values("match_date").reset_index(drop=True)


# Train the logistic-regression model and print evaluation results. #REFER TO
def train_and_evaluate(matches: pd.DataFrame) -> None:
    feature_columns = [column for column in matches if column.startswith("diff_")]
    cutoff = int(len(matches) * (1 - TEST_FRACTION))
    train, test = matches.iloc[:cutoff], matches.iloc[cutoff:]
    if len(train) == 0 or len(test) == 0 or train["team_a_win"].nunique() < 2:
        raise ValueError("Not enough ordered matches with both outcomes to train and test.")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_columns])
    x_test = scaler.transform(test[feature_columns])
    model = LogisticRegression(C=0.5, max_iter=1_000, random_state=RANDOM_SEED)
    model.fit(x_train, train["team_a_win"])
    predicted, probabilities = model.predict(x_test), model.predict_proba(x_test)[:, 1]
    baseline = (test["diff_recent_winrate"] > 0).astype(int)

    #print evaluation metrics HERE
    print(f"Matches used: {len(matches)} | train: {len(train)} | test: {len(test)}")
    print(f"Date range: {matches['match_date'].min():%Y-%m-%d} to {matches['match_date'].max():%Y-%m-%d}")
    print(f"Baseline accuracy: {accuracy_score(test['team_a_win'], baseline):.3f}")
    print(f"Model accuracy:    {accuracy_score(test['team_a_win'], predicted):.3f}")
    if test["team_a_win"].nunique() == 2:
        print(f"Model ROC AUC:     {roc_auc_score(test['team_a_win'], probabilities):.3f}")
    print("\nClassification report:\n", classification_report(test["team_a_win"], predicted, zero_division=0))
    weights = pd.Series(model.coef_[0], index=feature_columns).sort_values(ascending=False)
    print("Feature weights (positive favors Team A):\n", weights.round(3).to_string())


# Run the complete data-processing, training, and evaluation pipeline.
def main() -> None:
    args = parse_args()

    if args.lookback < 1 or args.min_history < 1 or args.min_history > args.lookback:
        raise ValueError("min-history must be at least 1 and no greater than lookback.")

    players = add_derived_stats(load_player_matches(args.database))
    players = add_player_history(players, args.lookback, args.min_history)
    teams = build_team_rows(players, args.lookback, args.min_history)

    feature_names = (
        [f"hist_{stat}" for stat in PLAYER_STATS]
        + ["recent_winrate", "team_elo"]
    )

    matches = build_match_rows(teams, feature_names)
    train_and_evaluate(matches)


if __name__ == "__main__":
    main()
