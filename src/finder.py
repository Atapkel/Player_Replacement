"""Replacement Finder: the core logic shared by the notebooks and the Streamlit app.

The question it answers: "We are selling player X. Who can replace him cheaply?"

For every candidate it computes three signals:

    fit     - how similar his playing style is to the sold player (shape of the skill profile)
    future  - his predicted overall rating two seasons later (growth model)
    price   - how cheap he is inside the budget

By default it keeps only candidates whose style fit is in the top half of the pool and
ranks them by `future`. This design won the historical backtest (notebook 04); `fit` and
`price` can still be mixed into the score with weights. The growth model is trained only on
data that would have been available at the decision date, so the same code can be used for
historical backtests without leaking the future.

`value_residuals` (a model of EA's market value) is kept for notebook 03, which shows that
EA prices are almost a formula of rating, potential and age, so they hide no bargains.

Club goals and multi-player plans (notebook 06):

    goal="replace" - like-for-like: same style, best predicted rating in 2 years (the default)
    goal="squad"   - strengthen the squad: best predicted rating in 2 years, any style
    goal="profit"  - make money: highest expected resale profit, using two extra models
                     (market value in 2 years, and the chance the player is still playing)

`ReplacementFinder.plan` replaces several sold players at once with one shared budget. It is
a small integer program (scipy `milp`): one signing per sold player, nobody bought twice,
total cost within the budget, maximize the club's goal. `focus` ("attack", "midfield",
"defense") doubles the weight of those positions, so the optimizer spends more there.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "players.parquet"

# Roles are a bit finer than position groups: a full-back and a centre-back are both
# defenders, but nobody would replace one with the other.
ROLE = {
    "GK": "GK",
    "CB": "CB",
    "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "CDM": "DM", "CM": "CM", "CAM": "AM",
    "LM": "W", "RM": "W", "LW": "W", "RW": "W",
    "CF": "ST", "ST": "ST",
}
# Roles close enough that a player listed at the sold player's position as a secondary
# position can be considered (a CM who also plays CDM can replace a CDM; a CB cannot).
NEIGHBOR_ROLES = {
    "GK": set(), "CB": {"DM"}, "FB": {"W"}, "DM": {"CM"}, "CM": {"DM", "AM"},
    "AM": {"CM", "W", "ST"}, "W": {"AM", "ST", "FB"}, "ST": {"W", "AM"},
}
ROLE_NAMES = {
    "GK": "Goalkeeper", "CB": "Centre-back", "FB": "Full-back / wing-back",
    "DM": "Defensive midfielder", "CM": "Central midfielder", "AM": "Attacking midfielder",
    "W": "Winger", "ST": "Striker",
}

OUTFIELD_SKILLS = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys",
    "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance",
    "power_shot_power", "power_jumping", "power_stamina", "power_strength",
    "power_long_shots",
    "mentality_aggression", "mentality_interceptions", "mentality_positioning",
    "mentality_vision", "mentality_penalties", "mentality_composure",
    "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
]
GK_SKILLS = [
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes",
    "movement_reactions", "attacking_short_passing", "skill_long_passing", "mentality_composure",
]

GROWTH_FEATURES = [
    "overall", "potential", "age", "international_reputation", "league_level",
    "league_strength", "contract_years_left", "weak_foot", "skill_moves", "height_cm",
    "role",
] + OUTFIELD_SKILLS + GK_SKILLS[:5]
VALUE_FEATURES = ["overall", "potential", "age", "international_reputation",
                  "contract_years_left", "role"]

FUTURE_FEATURES = GROWTH_FEATURES + ["log_value"]   # for the value-in-2-years and stay models

DEFAULT_WEIGHTS = {"fit": 0.0, "future": 1.0, "price": 0.0}
DEFAULT_FIT_QUANTILE = 0.5   # keep candidates whose style fit is above the pool median

GOALS = {
    "replace": "Like-for-like: same style, best player in 2 years",
    "squad": "Strengthen the squad: best player in 2 years, any style",
    "profit": "Make money: highest expected resale profit",
}
FOCUS_GROUPS = {"attack": {"ST", "W", "AM"}, "midfield": {"DM", "CM"}, "defense": {"GK", "CB", "FB"}}
FOCUS_WEIGHT = 2.0   # a focus position's gain counts double in a transfer plan


# ---------------------------------------------------------------- data

def load_players(path: Path = DATA) -> pd.DataFrame:
    """End-of-season snapshot of every player, with role and league strength added."""
    players = pd.read_parquet(path)
    players = players[players.snapshot == "end"].drop(columns="snapshot")
    players = players[players.main_position.isin(ROLE)].copy()
    players["role"] = players.main_position.map(ROLE).astype("category")

    # League strength = average rating of the league's players in that season.
    # Rough (see notebook 01: some leagues only include their top clubs), but better than nothing.
    players["league_strength"] = (players.groupby(["fifa_version", "league_id"], observed=True)
                                  ["overall"].transform("mean"))
    return players.reset_index(drop=True)


def growth_pairs(players: pd.DataFrame) -> pd.DataFrame:
    """Join every player-season with the same player two seasons later.

    Players who left the game have NaN in `overall_2y`.
    """
    later = (players[["player_id", "fifa_version", "overall", "value_eur"]]
             .assign(fifa_version=lambda x: x.fifa_version - 2)
             .rename(columns={"overall": "overall_2y", "value_eur": "value_eur_2y"}))
    return players.merge(later, on=["player_id", "fifa_version"], how="left")


# ---------------------------------------------------------------- models

def train_growth_model(players: pd.DataFrame, decision_version: int) -> HistGradientBoostingRegressor:
    """Predict overall rating two seasons ahead.

    Only uses pairs whose "two seasons later" is already known at the decision date
    (version + 2 <= decision_version), so there is no information from the future.
    """
    pairs = growth_pairs(players)
    train = pairs[(pairs.fifa_version + 2 <= decision_version) & pairs.overall_2y.notna()]
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          categorical_features="from_dtype", random_state=0)
    return model.fit(train[GROWTH_FEATURES], train.overall_2y)


def with_log_value(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_value"] = np.log(df.value_eur.where(df.value_eur > 0))
    return df


def train_future_value_model(players: pd.DataFrame, decision_version: int) -> HistGradientBoostingRegressor:
    """Predict log(market value) two seasons ahead, for players who are still playing then.

    Same no-leakage rule as the growth model: only pairs whose outcome is already known.
    """
    pairs = with_log_value(growth_pairs(players))
    train = pairs[(pairs.fifa_version + 2 <= decision_version)
                  & (pairs.value_eur > 0) & (pairs.value_eur_2y > 0)]
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          categorical_features="from_dtype", random_state=0)
    return model.fit(train[FUTURE_FEATURES], np.log(train.value_eur_2y))


def train_stay_model(players: pd.DataFrame, decision_version: int) -> HistGradientBoostingClassifier:
    """Classify whether a player is still in the game two seasons later.

    About a third of players disappear within two years (retired, dropped to unlicensed
    leagues, released). For a club that wants to sell later, that is the main resale risk.
    """
    pairs = with_log_value(growth_pairs(players))
    train = pairs[pairs.fifa_version + 2 <= decision_version]
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           categorical_features="from_dtype", random_state=0)
    return model.fit(train[FUTURE_FEATURES], train.overall_2y.notna())


def value_residuals(season: pd.DataFrame) -> pd.Series:
    """How much cheaper (positive) or more expensive (negative) each player is than
    the value model expects, in log units. log 0.4 ≈ "about 50% cheaper than expected".

    The model is trained on the same season (all other players' prices are known at the
    decision date). Out-of-fold predictions make sure no player is judged by a model that
    saw his own price.
    """
    priced = season[season.value_eur > 0]
    model = HistGradientBoostingRegressor(max_iter=300, categorical_features="from_dtype",
                                          random_state=0)
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    expected = cross_val_predict(model, priced[VALUE_FEATURES], np.log(priced.value_eur), cv=folds)
    return pd.Series(expected - np.log(priced.value_eur), index=priced.index).reindex(season.index)


# ---------------------------------------------------------------- style similarity

def skill_columns(role: str) -> list[str]:
    return GK_SKILLS if role == "GK" else OUTFIELD_SKILLS


def style_similarity(pool: pd.DataFrame, target: pd.Series) -> np.ndarray:
    """Cosine similarity of skill-profile *shape* between the target and every pool row.

    Each player's own average skill is subtracted first, so a 65-rated copy of an
    85-rated player still counts as very similar. Level is scored separately.
    """
    cols = skill_columns(ROLE[target.main_position])
    X = pool[cols].to_numpy(dtype=float)
    t = target[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X - np.nanmean(X, axis=1, keepdims=True))
    t = np.nan_to_num(t - np.nanmean(t))
    return X @ t / (np.linalg.norm(X, axis=1) * np.linalg.norm(t) + 1e-9)


# ---------------------------------------------------------------- the finder

class ReplacementFinder:
    """Everything needed to recommend replacements at one decision date (one FIFA version)."""

    def __init__(self, players: pd.DataFrame, decision_version: int):
        self.decision_version = decision_version
        self.growth_model = train_growth_model(players, decision_version)

        self.future_value_model = train_future_value_model(players, decision_version)
        self.stay_model = train_stay_model(players, decision_version)

        season = with_log_value(players[players.fifa_version == decision_version])
        season["pred_overall_2y"] = self.growth_model.predict(season[GROWTH_FEATURES])
        season["pred_value_2y"] = np.exp(self.future_value_model.predict(season[FUTURE_FEATURES]))
        season["p_stay"] = self.stay_model.predict_proba(season[FUTURE_FEATURES])[:, 1]
        # A player who drops out of the game can't be sold on at a good price: count it as 0
        season["expected_resale"] = season.p_stay * season.pred_value_2y
        season["expected_profit"] = season.expected_resale - season.value_eur
        self.season = season.set_index("player_id", drop=False)

    def player(self, player_id: int) -> pd.Series:
        return self.season.loc[player_id]

    def candidates(self, sold_id: int, budget: float, wage_cap: float | None = None,
                   max_age: float | None = None, min_overall: float | None = None) -> pd.DataFrame:
        """Every player who could replace `sold_id` within the constraints, with `fit` added."""
        sold = self.player(sold_id)
        s = self.season
        role = ROLE[sold.main_position]
        same_role = s.role == role
        # Also accept players from a neighbouring role who list the sold player's position
        plays_position = (s.player_positions.str.contains(rf"\b{sold.main_position}\b", regex=True)
                          & s.role.isin(NEIGHBOR_ROLES[role]))
        mask = (
            (same_role | plays_position)
            & (s.club_team_id != sold.club_team_id)     # not our own players
            & (s.value_eur > 0) & (s.value_eur <= budget)
        )
        if wage_cap is not None:
            mask &= s.wage_eur <= wage_cap
        if max_age is not None:
            mask &= s.age <= max_age
        if min_overall is not None:
            mask &= s.overall >= min_overall

        pool = s[mask].copy()
        pool["fit"] = style_similarity(pool, sold)
        return pool

    def recommend(self, sold_id: int, budget: float, n: int = 10, weights: dict | None = None,
                  fit_quantile: float = DEFAULT_FIT_QUANTILE, min_fit: float | None = None,
                  goal: str = "replace", **constraints) -> pd.DataFrame:
        """Top-n candidates for the club's goal, with a short explanation for each.

        min_fit: optional absolute minimum style match, useful for rare player types
        (e.g. a classic playmaker), where even the top half of the pool plays differently.
        goal: "replace" (default), "squad" or "profit", see GOALS.
        """
        sold = self.player(sold_id)
        pool = self.goal_pool(sold_id, budget, goal, fit_quantile, min_fit, **constraints)
        if pool.empty:
            return pool
        if goal == "profit":
            pool["score"] = pool.gain.rank(pct=True)
        else:
            pool = score(pool, weights or DEFAULT_WEIGHTS)
        top = pool.nlargest(n, "score").copy()
        top["why"] = [explain(row, sold, goal) for _, row in top.iterrows()]
        return top

    def goal_pool(self, sold_id: int, budget: float, goal: str = "replace",
                  fit_quantile: float = DEFAULT_FIT_QUANTILE, min_fit: float | None = None,
                  **constraints) -> pd.DataFrame:
        """Candidates allowed for this goal, with `gain` = what the goal maximizes.

        replace: style filter on;  gain = predicted rating in 2 years − sold player's rating
        squad:   any style;        gain = the same rating gain
        profit:  any style;        gain = expected resale profit in €M
        `min_fit` applies to every goal when given.
        """
        if goal not in GOALS:
            raise ValueError(f"goal must be one of {list(GOALS)}")
        sold = self.player(sold_id)
        pool = self.candidates(sold_id, budget, **constraints)
        if pool.empty:
            return pool.assign(gain=pd.Series(dtype=float))
        if goal == "replace":
            pool = keep_good_fit(pool, fit_quantile, min_fit)
        elif min_fit is not None:
            pool = pool[pool.fit >= min_fit]
        if goal == "profit":
            gain = pool.expected_profit / 1e6
        else:
            gain = pool.pred_overall_2y - sold.overall
        return pool.assign(gain=gain)

    def plan(self, sold_ids: list[int], budget: float, goal: str = "replace", focus: str | None = None,
             options_per_player: int = 25, max_age: float | None = 27, level_drop: float = 12,
             min_fit: float | None = None, wage_caps: dict | None = None) -> pd.DataFrame:
        """Replace several sold players at once with one shared budget.

        Integer program: for every sold player choose exactly one candidate, never the same
        candidate twice, total cost ≤ budget, maximize Σ weight × gain. `focus` gives the
        positions in FOCUS_GROUPS[focus] weight FOCUS_WEIGHT, so more money goes there.

        Returns one row per sold player (columns `sold_*` describe him), with `weight` and
        `gain`. Raises ValueError if no combination fits the budget.
        """
        focus_roles = FOCUS_GROUPS.get(focus, set()) if focus else set()
        options = []
        for slot, sid in enumerate(sold_ids):
            sold = self.player(sid)
            cap = (wage_caps or {}).get(sid, sold.wage_eur)
            pool = self.goal_pool(sid, budget, goal, min_fit=min_fit, wage_cap=cap, max_age=max_age,
                                  min_overall=sold.overall - level_drop)
            pool = pool[~pool.club_team_id.isin([self.player(i).club_team_id for i in sold_ids])]
            if pool.empty:
                raise ValueError(f"No candidates for {sold.short_name} within these limits")
            # Options: the best by gain, plus every "best for its price" candidate (nobody cheaper
            # has a higher gain). That frontier contains the best pick for any budget share, so
            # the joint plan can never do worse than splitting the budget by hand.
            by_price = pool.sort_values(["value_eur", "gain"], ascending=[True, False])
            frontier = by_price[by_price.gain > by_price.gain.cummax().shift(fill_value=-np.inf)]
            keep = pd.concat([pool.nlargest(options_per_player, "gain"), frontier])
            keep = keep[~keep.index.duplicated()].copy()
            keep["slot"] = slot
            keep["weight"] = FOCUS_WEIGHT if ROLE[sold.main_position] in focus_roles else 1.0
            for col in ("short_name", "main_position", "overall", "value_eur", "age"):
                keep[f"sold_{col}"] = sold[col]
            keep["sold_id"] = sid
            options.append(keep)
        opts = pd.concat(options, ignore_index=True)

        k = len(opts)
        one_per_slot = np.zeros((len(sold_ids), k))
        one_per_slot[opts.slot.to_numpy(), np.arange(k)] = 1
        ids = opts.player_id.to_numpy()
        repeated = [pid for pid in np.unique(ids) if (ids == pid).sum() > 1]
        once = np.array([(ids == pid).astype(float) for pid in repeated]).reshape(len(repeated), k)
        constraints = [LinearConstraint(one_per_slot, 1, 1),
                       LinearConstraint(opts.value_eur.to_numpy()[None, :], 0, budget)]
        if repeated:
            constraints.append(LinearConstraint(once, 0, 1))
        result = milp(c=-(opts.weight * opts.gain).to_numpy(), constraints=constraints,
                      integrality=np.ones(k), bounds=Bounds(0, 1))
        if not result.success:
            raise ValueError("No combination of signings fits this budget. Try a bigger budget.")
        chosen = opts[result.x > 0.5].sort_values("slot")
        return chosen.set_index("slot")


def keep_good_fit(pool: pd.DataFrame, quantile: float = DEFAULT_FIT_QUANTILE,
                  min_fit: float | None = None) -> pd.DataFrame:
    """Drop candidates whose playing style is far from the sold player's."""
    threshold = pool.fit.quantile(quantile)
    if min_fit is not None:
        threshold = max(threshold, min_fit)
    return pool[pool.fit >= threshold]


def score(pool: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Turn each signal into a 0–1 percentile inside the pool and take a weighted sum.

    Percentiles make the three signals comparable even though they are measured in
    completely different units (cosine, rating points, euros).
    """
    pool = pool.copy()
    pool["fit_pct"] = pool.fit.rank(pct=True)
    pool["future_pct"] = pool.pred_overall_2y.rank(pct=True)
    pool["price_pct"] = (-pool.value_eur).rank(pct=True)   # cheaper = higher
    total = sum(weights.values())
    pool["score"] = sum(weights[k] * pool[f"{k}_pct"] for k in ("fit", "future", "price")) / total
    return pool


def explain(row: pd.Series, sold: pd.Series, goal: str = "replace") -> str:
    if goal == "profit":
        return (f"costs €{row.value_eur / 1e6:.1f}M · predicted value in 2 years €{row.pred_value_2y / 1e6:.1f}M · "
                f"{row.p_stay:.0%} chance he is still playing · expected profit €{row.expected_profit / 1e6:+.1f}M")
    return (f"{row.fit:.0%} style match · rated {row.overall:.0f} now, "
            f"predicted {row.pred_overall_2y:.0f} in 2 years · "
            f"costs {row.value_eur / sold.value_eur:.0%} of {sold.short_name}'s value")


def risk_flags(row: pd.Series, sold: pd.Series) -> str:
    """Simple warnings for a candidate, each tied to a known pitfall of football data."""
    flags = []
    if row.club_position in ("SUB", "RES"):
        flags.append("bench player")          # proxy for little playing time
    if row.league_strength < sold.league_strength - 3:
        flags.append("weaker league")         # ratings across leagues aren't fully comparable
    if row.contract_years_left <= 1:
        flags.append("contract ending")
    if row.age >= 27:
        flags.append("little growth left")
    if "p_stay" in row and row.p_stay < 0.5:
        flags.append("high drop-out risk")    # stay model: under 50% chance he is still playing in 2 years
    return ", ".join(flags) or "-"


# ---------------------------------------------------------------- simple baselines for the backtest

def pick_highest_overall(pool: pd.DataFrame, n: int) -> pd.DataFrame:
    """What a basic scout does: the best current rating he can afford."""
    return pool.nlargest(n, "overall")


def pick_highest_potential(pool: pd.DataFrame, n: int) -> pd.DataFrame:
    """The 'buy young talent' rule of thumb: the highest EA potential he can afford."""
    return pool.nlargest(n, "potential")


def pick_random(pool: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return pool.sample(min(n, len(pool)), random_state=seed)
