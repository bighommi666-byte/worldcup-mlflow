"""
Data loading and feature engineering for FIFA World Cup match prediction.

Target: outcome of the match from the 'home' (listed-first) team's perspective
    0 = Home/listed-first team WINS
    1 = DRAW
    2 = Away/listed-second team WINS

NOTE on home/away: most World Cup matches are at neutral venues, so "home"
here mostly means "the team listed first in the fixture". To give the models
real signal beyond position, we add historical team-strength features
(win rate, avg goals scored/conceded) computed ONLY from the training split
to avoid data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def _clean_team_name(name):
    """Fix the encoding artifacts and inconsistent names in the dataset."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    # Remove the 'rn">' HTML artifact that prefixes several names
    name = name.replace('rn">', '').replace('rn"', '').strip()
    # Normalise a few well-known variants
    replacements = {
        "Germany FR": "Germany",
        "IR Iran": "Iran",
        "Korea Republic": "South Korea",
        "Korea DPR": "North Korea",
        "Soviet Union": "Russia",
    }
    return replacements.get(name, name)


def _outcome(row):
    if row["Home Team Goals"] > row["Away Team Goals"]:
        return 0  # home win
    if row["Home Team Goals"] < row["Away Team Goals"]:
        return 2  # away win
    return 1      # draw


def load_raw():
    """Load and clean the raw matches into a tidy DataFrame of real matches."""
    df = pd.read_csv("data/WorldCupMatches.csv")

    # Drop the ~3700 empty junk rows
    df = df.dropna(subset=["Home Team Name", "Away Team Name",
                           "Home Team Goals", "Away Team Goals"]).copy()

    # The raw Kaggle file contains some duplicate rows (e.g. several 2014
    # matches appear twice). A World Cup fixture is unique per
    # (Year, Stage, Home, Away), so drop exact duplicates on those keys.
    df = df.drop_duplicates(
        subset=["Year", "Stage", "Home Team Name", "Away Team Name",
                "Home Team Goals", "Away Team Goals"]
    ).copy()

    # Clean team names
    df["HomeTeam"] = df["Home Team Name"].apply(_clean_team_name)
    df["AwayTeam"] = df["Away Team Name"].apply(_clean_team_name)

    df["Home Team Goals"] = df["Home Team Goals"].astype(int)
    df["Away Team Goals"] = df["Away Team Goals"].astype(int)

    # Target
    df["Outcome"] = df.apply(_outcome, axis=1)

    # Knockout flag
    df["Is_Knockout"] = df["Stage"].apply(
        lambda x: 1 if any(s in str(x) for s in
        ["Final", "Semi", "Quarter", "Round of", "Third place",
         "Play-off", "eighth"]) else 0
    )

    # Normalised year (helps tree models split eras)
    yr_min, yr_max = df["Year"].min(), df["Year"].max()
    df["Year_Norm"] = (df["Year"] - yr_min) / (yr_max - yr_min)

    return df.reset_index(drop=True)


def _build_team_stats(train_df):
    """Compute per-team historical stats from the TRAINING split only."""
    stats = {}
    # Accumulate appearances, wins, goals for/against across home+away rows
    for _, r in train_df.iterrows():
        for team, gf, ga, won in [
            (r["HomeTeam"], r["Home Team Goals"], r["Away Team Goals"],
             r["Outcome"] == 0),
            (r["AwayTeam"], r["Away Team Goals"], r["Home Team Goals"],
             r["Outcome"] == 2),
        ]:
            s = stats.setdefault(team, {"games": 0, "wins": 0,
                                        "gf": 0, "ga": 0})
            s["games"] += 1
            s["wins"] += int(won)
            s["gf"] += gf
            s["ga"] += ga

    # Convert to rates
    table = {}
    for team, s in stats.items():
        g = max(s["games"], 1)
        table[team] = {
            "win_rate": s["wins"] / g,
            "avg_gf": s["gf"] / g,
            "avg_ga": s["ga"] / g,
        }
    # Global fallback for teams unseen in training
    table["__GLOBAL__"] = {
        "win_rate": np.mean([v["win_rate"] for v in table.values()]),
        "avg_gf": np.mean([v["avg_gf"] for v in table.values()]),
        "avg_ga": np.mean([v["avg_ga"] for v in table.values()]),
    }
    return table


def _apply_team_stats(df, table):
    g = table["__GLOBAL__"]

    def feats(team_col, prefix):
        wr, gf, ga = [], [], []
        for t in df[team_col]:
            s = table.get(t, g)
            wr.append(s["win_rate"])
            gf.append(s["avg_gf"])
            ga.append(s["avg_ga"])
        df[f"{prefix}_win_rate"] = wr
        df[f"{prefix}_avg_gf"] = gf
        df[f"{prefix}_avg_ga"] = ga

    feats("HomeTeam", "home")
    feats("AwayTeam", "away")
    return df


FEATURES = [
    "home_win_rate", "home_avg_gf", "home_avg_ga",
    "away_win_rate", "away_avg_gf", "away_avg_ga",
    "Is_Knockout", "Year_Norm",
]


def load_data(return_meta=False):
    """
    Returns: X_train, X_test, y_train, y_test, scaler, team_table
    Set return_meta=True to also get the cleaned raw matches DataFrame.
    """
    df = load_raw()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["Outcome"]
    )

    # Build team stats on TRAIN only (no leakage), apply to both splits
    table = _build_team_stats(train_df)
    train_df = _apply_team_stats(train_df.copy(), table)
    test_df = _apply_team_stats(test_df.copy(), table)

    X_train = train_df[FEATURES].values
    X_test = test_df[FEATURES].values
    y_train = train_df["Outcome"].values
    y_test = test_df["Outcome"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if return_meta:
        return X_train, X_test, y_train, y_test, scaler, table, df
    return X_train, X_test, y_train, y_test, scaler, table


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, sc, tbl = load_data()
    print("Train shape:", Xtr.shape, "Test shape:", Xte.shape)
    print("Features:", FEATURES)
    print("Class counts (train):", np.bincount(ytr))
    print("Teams in stats table:", len(tbl) - 1)
