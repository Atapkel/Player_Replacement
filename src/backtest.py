"""Historical backtest: would the Replacement Finder have made better picks than simple rules?

Idea: go back to the end of an old season (e.g. FIFA 20, summer 2020), pretend that a club
sells one of its players, and let every method pick 5 replacements using only the data
known at that moment. Then look two seasons ahead (FIFA 22) and measure how the picks turned out.

Repeating this for hundreds of random "sold" players gives a fair comparison.
"""

import numpy as np
import pandas as pd

from finder import (DEFAULT_WEIGHTS, ReplacementFinder, growth_pairs, keep_good_fit,
                    pick_highest_overall, pick_highest_potential, pick_random, score)


def weighted(weights: dict, fit_filter: bool = False, min_fit: float | None = None):
    """A picking rule: optionally keep good style fits only, then rank by weighted score."""
    def pick(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        if fit_filter:
            kept = keep_good_fit(pool, min_fit=min_fit)
            pool = kept if len(kept) >= n else pool.nlargest(n, "fit")
        return score(pool, weights).nlargest(n, "score")
    return pick


SYSTEMS = {
    "Finder (style filter + growth model)": weighted(DEFAULT_WEIGHTS, fit_filter=True),
    "Finder, minimum style match 0.6": weighted(DEFAULT_WEIGHTS, fit_filter=True, min_fit=0.6),
    "Finder, prefer cheaper (price weight 0.3)": weighted({"fit": 0, "future": 0.7, "price": 0.3}, fit_filter=True),
    "Growth model only (no style filter)": weighted(DEFAULT_WEIGHTS),
    "First design (fit .4 / future .4 / price .2)": weighted({"fit": 0.4, "future": 0.4, "price": 0.2}),
    "Highest overall": pick_highest_overall,
    "Highest potential": pick_highest_potential,
    "Random": pick_random,
}
FINDER = "Finder (style filter + growth model)"


def sample_cases(season: pd.DataFrame, n_cases: int, seed: int) -> pd.DataFrame:
    """Realistic 'sold' players: established first-division players aged 22–30."""
    eligible = season[season.overall.between(70, 84) & season.age.between(22, 30)
                      & (season.league_level == 1) & (season.value_eur > 0)]
    return eligible.sample(min(n_cases, len(eligible)), random_state=seed)


def run_backtest(players: pd.DataFrame, decision_versions=(19, 20), n_cases: int = 250,
                 budget_factor: float = 0.6, top_n: int = 5, max_age: float = 28,
                 level_drop: float = 12, seed: int = 0, finders: dict | None = None,
                 systems: dict | None = None) -> pd.DataFrame:
    """Return one row per (case, system, pick) with the pick's outcome two seasons later.

    budget_factor: budget as a share of the sold player's value (0.6 = 60% of his price).
    level_drop: candidates may be at most this many points below the sold player today.
    finders: optional pre-built {version: ReplacementFinder} to avoid retraining.
    systems: optional subset of SYSTEMS to compare.
    """
    systems = systems or SYSTEMS
    future = growth_pairs(players).set_index(["player_id", "fifa_version"])[["overall_2y", "value_eur_2y"]]
    rows = []
    for version in decision_versions:
        finder = (finders or {}).get(version) or ReplacementFinder(players, version)
        for case_no, (_, sold) in enumerate(sample_cases(finder.season, n_cases, seed + version).iterrows()):
            pool = finder.candidates(sold.player_id, budget=budget_factor * sold.value_eur,
                                     wage_cap=sold.wage_eur, max_age=max_age,
                                     min_overall=sold.overall - level_drop)
            if len(pool) < top_n:
                continue
            for system, pick in systems.items():
                picks = pick(pool, top_n)
                for _, p in picks.iterrows():
                    rows.append({
                        "version": version, "case": f"{version}-{case_no}",
                        "sold_id": sold.player_id, "sold_overall": sold.overall,
                        "system": system, "player_id": p.player_id, "age": p.age,
                        "overall": p.overall, "value_eur": p.value_eur, "fit": p.fit,
                        "pool_size": len(pool),
                    })
    picks = pd.DataFrame(rows)
    picks = picks.join(future, on=["player_id", "version"])
    picks["still_in_game"] = picks.overall_2y.notna()
    # "Success": two years later he is still in the game and at most 3 points below the sold player
    picks["success"] = picks.still_in_game & (picks.overall_2y >= picks.sold_overall - 3)
    picks["value_change"] = np.log(picks.value_eur_2y / picks.value_eur)
    return picks


def summarize(picks: pd.DataFrame) -> pd.DataFrame:
    """Average outcome per system (each case counts equally)."""
    per_case = picks.groupby(["system", "case"]).agg(
        success=("success", "mean"),
        overall_2y=("overall_2y", "mean"),
        dropout=("still_in_game", lambda s: 1 - s.mean()),
        fit=("fit", "mean"),
        cost_m=("value_eur", lambda v: v.mean() / 1e6),
        value_change=("value_change", "median"),
    )
    table = per_case.groupby("system").mean()
    return table.loc[list(dict.fromkeys(s for s in SYSTEMS if s in table.index))]


def bootstrap_difference(picks: pd.DataFrame, metric: str, a: str, b: str,
                         n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean difference a − b of a per-case metric with a 95% bootstrap interval.

    Resampling whole cases (not single picks) respects that the 5 picks of one case
    are not independent.
    """
    per_case = picks.groupby(["case", "system"])[metric].mean().unstack()
    diff = (per_case[a] - per_case[b]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return diff.mean(), *np.percentile(boots, [2.5, 97.5])


# ---------------------------------------------------------------- club goals and multi-player plans

def pick_most_profit(pool: pd.DataFrame, n: int) -> pd.DataFrame:
    """The 'profit' goal: highest expected resale profit (value in 2 years × chance he still plays − price)."""
    return pool.nlargest(n, "expected_profit")


def realized_profit(picks: pd.DataFrame) -> pd.Series:
    """Value two seasons later minus the price paid, in €M.

    Conservative: a player who left the game counts as resold for nothing (his whole price is lost).
    """
    return (picks.value_eur_2y.fillna(0) - picks.value_eur) / 1e6


def sample_plan_cases(season: pd.DataFrame, n_cases: int, k: int, seed: int) -> list[list[int]]:
    """Clubs that sell k established players at once (same rules as `sample_cases`)."""
    eligible = season[season.overall.between(70, 84) & season.age.between(22, 30)
                      & (season.league_level == 1) & (season.value_eur > 0)]
    rng = np.random.default_rng(seed)
    clubs = eligible.groupby("club_team_id").player_id.apply(list)
    clubs = clubs[clubs.str.len() >= k]
    chosen = rng.choice(len(clubs), size=min(n_cases, len(clubs)), replace=False)
    return [list(rng.choice(clubs.iloc[i], size=k, replace=False)) for i in chosen]


def split_budget_picks(finder: ReplacementFinder, sold_ids: list[int], budget_factor: float,
                       goal: str, max_age: float, level_drop: float) -> list[int] | None:
    """The simple alternative to a joint plan: give every sold player his own share of the
    budget (budget_factor × his value) and pick the best candidate for each one separately."""
    picked = []
    budget = budget_factor * sum(finder.player(i).value_eur for i in sold_ids)
    for sid in sold_ids:
        sold = finder.player(sid)
        # Same candidate pool as the joint plan (built with the whole budget), then this player's share
        pool = finder.goal_pool(sid, budget, goal, wage_cap=sold.wage_eur,
                                max_age=max_age, min_overall=sold.overall - level_drop)
        pool = pool[(pool.value_eur <= budget_factor * sold.value_eur) & ~pool.player_id.isin(picked)]
        if pool.empty:
            return None
        picked.append(int(pool.gain.idxmax()))
    return picked


def run_plan_backtest(players: pd.DataFrame, decision_versions=(19, 20, 21), n_cases: int = 150,
                      k: int = 2, budget_factor: float = 0.6, goal: str = "replace", max_age: float = 28,
                      level_drop: float = 12, seed: int = 0, finders: dict | None = None) -> pd.DataFrame:
    """Clubs sell k players at once with one shared budget (budget_factor × their total value).

    Compares the joint plan (integer program) with splitting the budget and choosing separately.
    Only cases where both methods find a solution are kept, so they are compared on the same cases.
    Returns one row per (case, method, sold player) with the pick's outcome two seasons later.
    """
    future = growth_pairs(players).set_index(["player_id", "fifa_version"])[["overall_2y", "value_eur_2y"]]
    rows = []
    for version in decision_versions:
        finder = (finders or {}).get(version) or ReplacementFinder(players, version)
        for case_no, sold_ids in enumerate(sample_plan_cases(finder.season, n_cases, k, seed + version)):
            budget = budget_factor * sum(finder.player(i).value_eur for i in sold_ids)
            try:
                plan = finder.plan(sold_ids, budget, goal=goal, max_age=max_age, level_drop=level_drop)
            except ValueError:
                continue
            split = split_budget_picks(finder, sold_ids, budget_factor, goal, max_age, level_drop)
            if split is None:
                continue
            methods = {"Joint plan (optimizer)": plan.player_id.tolist(), "Split budget, choose separately": split}
            for method, picks in methods.items():
                for sid, pid in zip(sold_ids, picks):
                    rows.append({"version": version, "case": f"{version}-{case_no}", "method": method,
                                 "sold_id": sid, "sold_overall": finder.player(sid).overall,
                                 "player_id": pid, "value_eur": finder.player(pid).value_eur,
                                 "pred_overall_2y": finder.player(pid).pred_overall_2y, "budget": budget})
    picks = pd.DataFrame(rows).join(future, on=["player_id", "version"])
    picks["still_in_game"] = picks.overall_2y.notna()
    picks["success"] = picks.still_in_game & (picks.overall_2y >= picks.sold_overall - 3)
    picks["profit_m"] = realized_profit(picks)
    return picks
