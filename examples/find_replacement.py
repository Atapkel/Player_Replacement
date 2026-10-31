"""Ask the Replacement Finder for a shortlist from the command line.

Examples (run from the project folder):
    python examples/find_replacement.py --player "Y. Bissouma" --club "Brighton & Hove Albion" --budget 25
    python examples/find_replacement.py --player "Casemiro" --club "Real Madrid" --budget 44
    python examples/find_replacement.py --player "C. Eriksen" --club "Brentford" --budget 15 --max-age 28 --min-fit 0.6
    python examples/find_replacement.py --list-squad --club "Arsenal"
"""

import argparse
import difflib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from finder import ReplacementFinder, load_players, risk_flags
from plot_style import euro


def main():
    parser = argparse.ArgumentParser(description="Find cheap replacements for a player you sell.")
    parser.add_argument("--player", help='short name as in FIFA, e.g. "Y. Bissouma"')
    parser.add_argument("--club", required=True, help='club name as in FIFA, e.g. "Brighton & Hove Albion"')
    parser.add_argument("--season", type=int, default=22, help="FIFA version = decision at the end of that season (19-23)")
    parser.add_argument("--budget", type=float, help="transfer budget in € millions (default: 70%% of his value)")
    parser.add_argument("--wage", type=float, help="max wage in € thousands per week (default: his wage)")
    parser.add_argument("--max-age", type=float, default=27)
    parser.add_argument("--min-fit", type=float, default=None, help="minimum style match 0-1, for rare player types")
    parser.add_argument("--cheaper", type=float, default=0.0, help="0 = best future rating, up to 0.5 = prefer cheaper")
    parser.add_argument("-n", type=int, default=10, help="shortlist length")
    parser.add_argument("--list-squad", action="store_true", help="only print the club's squad (to find exact names)")
    args = parser.parse_args()

    print(f"Training the growth model for FIFA {args.season} (about 10 seconds)...")
    finder = ReplacementFinder(load_players(), args.season)
    season = finder.season

    squad = season[season.club_name == args.club]
    if squad.empty:
        clubs = season.club_name.dropna().astype(str).unique().tolist()
        hints = difflib.get_close_matches(args.club, clubs, n=5, cutoff=0.5)
        sys.exit(f"No club called '{args.club}' in FIFA {args.season}. Did you mean: {hints}")

    if args.list_squad or not args.player:
        cols = ["short_name", "main_position", "age", "overall", "value_eur"]
        print(squad.sort_values("overall", ascending=False)[cols].to_string(index=False))
        return

    match = squad[squad.short_name == args.player]
    if match.empty:
        hints = difflib.get_close_matches(args.player, squad.short_name.tolist(), n=5, cutoff=0.4)
        sys.exit(f"No player '{args.player}' at {args.club}. Did you mean: {hints}? (--list-squad shows everyone)")
    sold = match.iloc[0]

    budget = (args.budget * 1e6) if args.budget else 0.7 * sold.value_eur
    wage = (args.wage * 1e3) if args.wage else sold.wage_eur
    print(f"\nSelling {sold.short_name} ({sold.club_name}): {sold.main_position}, age {sold.age:.0f}, "
          f"rated {sold.overall:.0f}, value {euro(sold.value_eur)}")
    print(f"Budget {euro(budget)}, max wage {euro(wage)}/week, max age {args.max_age:.0f}\n")

    rec = finder.recommend(
        sold.player_id, budget=budget, n=args.n, min_fit=args.min_fit,
        weights={"fit": 0.0, "future": 1.0 - args.cheaper, "price": args.cheaper},
        wage_cap=wage, max_age=args.max_age, min_overall=sold.overall - 12,
    )
    if rec.empty:
        sys.exit("Nobody fits these limits. Try a bigger budget, a higher wage or a lower --min-fit.")

    table = pd.DataFrame({
        "player": rec.short_name, "age": rec.age.astype(int), "club": rec.club_name.astype(str),
        "pos": rec.main_position.astype(str), "now": rec.overall.astype(int),
        "in 2y": rec.pred_overall_2y.round(1), "value": rec.value_eur.map(euro),
        "style": rec.fit.round(2), "risks": [risk_flags(r, sold) for _, r in rec.iterrows()],
    })
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
