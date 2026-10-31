"""Replace several sold players at once with one shared budget, for the club's goal.

Examples (run from the project folder):
    python examples/plan_transfers.py --club "Brighton & Hove Albion" --sell "Y. Bissouma" --sell "Cucurella" --budget 45
    python examples/plan_transfers.py --club "Brighton & Hove Albion" --sell "Y. Bissouma" --sell "Cucurella" --budget 45 --focus midfield
    python examples/plan_transfers.py --club "Brighton & Hove Albion" --sell "Y. Bissouma" --sell "Cucurella" --budget 45 --goal profit
"""

import argparse
import difflib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from finder import FOCUS_GROUPS, GOALS, ReplacementFinder, load_players, risk_flags
from plot_style import euro


def main():
    parser = argparse.ArgumentParser(description="Plan signings for several sold players with one budget.")
    parser.add_argument("--club", required=True, help='club name as in FIFA, e.g. "Brighton & Hove Albion"')
    parser.add_argument("--sell", action="append", required=True, help='short name of a sold player; repeat for each')
    parser.add_argument("--budget", type=float, help="total budget in € millions (default: 70%% of their total value)")
    parser.add_argument("--goal", choices=list(GOALS), default="replace", help="; ".join(f"{k}: {v}" for k, v in GOALS.items()))
    parser.add_argument("--focus", choices=list(FOCUS_GROUPS), help="positions that count double in the plan")
    parser.add_argument("--season", type=int, default=22, help="FIFA version = decision at the end of that season (19-23)")
    parser.add_argument("--max-age", type=float, default=27)
    parser.add_argument("--min-fit", type=float, default=None, help="minimum style match 0-1")
    args = parser.parse_args()

    print(f"Training the models for FIFA {args.season} (about 20 seconds)...")
    finder = ReplacementFinder(load_players(), args.season)
    squad = finder.season[finder.season.club_name == args.club]
    if squad.empty:
        clubs = finder.season.club_name.dropna().astype(str).unique().tolist()
        sys.exit(f"No club '{args.club}'. Did you mean: {difflib.get_close_matches(args.club, clubs, n=5, cutoff=0.5)}")

    sold_ids = []
    for name in args.sell:
        match = squad[squad.short_name == name]
        if match.empty:
            hints = difflib.get_close_matches(name, squad.short_name.tolist(), n=5, cutoff=0.4)
            sys.exit(f"No player '{name}' at {args.club}. Did you mean: {hints}?")
        sold_ids.append(match.iloc[0].player_id)

    total_value = sum(finder.player(i).value_eur for i in sold_ids)
    budget = args.budget * 1e6 if args.budget else 0.7 * total_value
    print(f"\nSelling {', '.join(args.sell)} (total FIFA value {euro(total_value)}). Budget {euro(budget)}.")
    print(f"Goal: {GOALS[args.goal]}" + (f" · focus: {args.focus}" if args.focus else ""))

    try:
        plan = finder.plan(sold_ids, budget, goal=args.goal, focus=args.focus,
                           max_age=args.max_age, min_fit=args.min_fit)
    except ValueError as err:
        sys.exit(str(err))

    table = pd.DataFrame({
        "replaces": plan.sold_short_name, "player": plan.short_name, "age": plan.age.astype(int),
        "club": plan.club_name.astype(str), "pos": plan.main_position.astype(str),
        "now": plan.overall.astype(int), "in 2y": plan.pred_overall_2y.round(1),
        "price": plan.value_eur.map(euro), "value 2y": plan.pred_value_2y.map(euro),
        "stays": plan.p_stay.map("{:.0%}".format), "style": plan.fit.round(2),
        "risks": [risk_flags(r, finder.player(r.sold_id)) for _, r in plan.iterrows()],
    })
    print("\n" + table.to_string(index=False))
    print(f"\nSpent {euro(plan.value_eur.sum())} of {euro(budget)}; "
          f"expected profit in 2 years {euro(plan.expected_profit.sum())}; "
          f"average predicted rating in 2 years {plan.pred_overall_2y.mean():.1f}")


if __name__ == "__main__":
    main()
