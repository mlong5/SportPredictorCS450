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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="Path to the VCT DuckDB database")
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="Prior matches to average")
    parser.add_argument("--min-history", type=int, default=MIN_HISTORY, help="Prior matches required")
    return parser.parse_args()


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


def add_derived_stats(players: pd.DataFrame) -> pd.DataFrame:
    """Add rate statistics, guarding against zero deaths."""
    players = players.copy()
    deaths = players["deaths"].clip(lower=1)
    players["kd"] = players["kills"] / deaths
    players["kda"] = (players["kills"] + players["assists"]) / deaths
    players["fb_net"] = players["first_bloods"] - players["first_deaths"]
    return players


def past_rolling_average(values: pd.Series, lookback: int, min_history: int) -> pd.Series:
    """Average preceding observations only; shift prevents target leakage."""
    return values.shift(1).rolling(lookback, min_periods=min_history).mean()


def add_player_history(players: pd.DataFrame, lookback: int, min_history: int) -> pd.DataFrame:
    players = players.copy()
    for stat in PLAYER_STATS:
        players[f"hist_{stat}"] = players.groupby("player_id", sort=False)[stat].transform(
            lambda values: past_rolling_average(values, lookback, min_history)
        )
    return players


def build_team_rows(players: pd.DataFrame, lookback: int, min_history: int) -> pd.DataFrame:
    """Reduce player history to team form for each match."""
    history_columns = [f"hist_{stat}" for stat in PLAYER_STATS]
    teams = (
        players.groupby(["match_id", "match_date", "team_idx", "team_id"], as_index=False)
        .agg(**{column: (column, "mean") for column in history_columns}, team_won=("team_won", "first"))
        .sort_values(["match_date", "match_id"])
    )
    teams["recent_winrate"] = teams.groupby("team_id", sort=False)["team_won"].transform(
        lambda values: past_rolling_average(values, lookback, min_history)
    )
    return teams


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

    print(f"Matches used: {len(matches)} | train: {len(train)} | test: {len(test)}")
    print(f"Date range: {matches['match_date'].min():%Y-%m-%d} to {matches['match_date'].max():%Y-%m-%d}")
    print(f"Baseline accuracy: {accuracy_score(test['team_a_win'], baseline):.3f}")
    print(f"Model accuracy:    {accuracy_score(test['team_a_win'], predicted):.3f}")
    if test["team_a_win"].nunique() == 2:
        print(f"Model ROC AUC:     {roc_auc_score(test['team_a_win'], probabilities):.3f}")
    print("\nClassification report:\n", classification_report(test["team_a_win"], predicted, zero_division=0))
    weights = pd.Series(model.coef_[0], index=feature_columns).sort_values(ascending=False)
    print("Feature weights (positive favors Team A):\n", weights.round(3).to_string())


def main() -> None:
    args = parse_args()
    if args.lookback < 1 or args.min_history < 1 or args.min_history > args.lookback:
        raise ValueError("min-history must be at least 1 and no greater than lookback.")
    players = add_derived_stats(load_player_matches(args.database))
    players = add_player_history(players, args.lookback, args.min_history)
    teams = build_team_rows(players, args.lookback, args.min_history)
    feature_names = [f"hist_{stat}" for stat in PLAYER_STATS] + ["recent_winrate"]
    train_and_evaluate(build_match_rows(teams, feature_names))


if __name__ == "__main__":
    main()
