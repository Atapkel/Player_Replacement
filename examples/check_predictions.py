"""Test the growth model on data it has never seen.

We pretend it is summer 2021 (end of FIFA 21): the model is trained only on seasons whose
"two years later" was known by then. It predicts every player's rating two seasons ahead,
and we compare with the real ratings in FIFA 23 (January 2023).

Examples (run from the project folder):
    python examples/check_predictions.py
    python examples/check_predictions.py --player "Pedri" --player "J. Bellingham" --player "E. Camavinga"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from finder import GROWTH_FEATURES, growth_pairs, load_players, train_growth_model

DECISION = 21   # summer 2021; outcomes are in FIFA 23


def main():
    parser = argparse.ArgumentParser(description="Compare growth predictions with what really happened.")
    parser.add_argument("--player", action="append", default=[], help="short name(s) to look at, can repeat")
    args = parser.parse_args()

    players = load_players()
    print(f"Training on data known in summer 20{DECISION} (pairs up to FIFA {DECISION - 2} -> {DECISION})...")
    model = train_growth_model(players, decision_version=DECISION)

    test = growth_pairs(players)
    test = test[(test.fifa_version == DECISION) & test.overall_2y.notna()].copy()
    test["predicted"] = model.predict(test[GROWTH_FEATURES])
    real_growth = test.overall_2y - test.overall

    print(f"\nTest set: {len(test):,} players from FIFA {DECISION} who are still in FIFA {DECISION + 2}\n")
    print("Average error (rating points, lower is better)")
    print(f"  'Nothing changes' baseline : {np.mean(np.abs(test.overall_2y - test.overall)):.2f}")
    print(f"  Our growth model           : {np.mean(np.abs(test.overall_2y - test.predicted)):.2f}")
    young = (test.age <= 23).to_numpy()
    print("\nWho will grow? Rank correlation with real growth, players aged 23 or younger (higher is better)")
    print(f"  EA's potential - overall   : {spearmanr((test.potential - test.overall)[young], real_growth[young]).statistic:.2f}")
    print(f"  Our growth model           : {spearmanr((test.predicted - test.overall)[young], real_growth[young]).statistic:.2f}")

    cols = ["short_name", "age", "club_name", "overall", "potential", "predicted", "overall_2y"]
    if args.player:
        view = test[test.short_name.isin(args.player)]
        missing = set(args.player) - set(view.short_name)
        if missing:
            print(f"\nNot found in both FIFA {DECISION} and {DECISION + 2}: {sorted(missing)}")
    else:
        view = test[test.overall >= 70].assign(g=lambda d: d.predicted - d.overall).nlargest(10, "g")
        print(f"\nThe 10 players (rated 70+) the model expected to grow most, and what happened:")
    print(view[cols].rename(columns={"overall": f"FIFA {DECISION}", "predicted": "predicted",
                                     "overall_2y": f"real FIFA {DECISION + 2}"}).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
