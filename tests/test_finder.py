"""Automatic checks that the Replacement Finder follows its own rules.

Run from the project folder:   pytest -v tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from finder import (GROWTH_FEATURES, NEIGHBOR_ROLES, ROLE, ReplacementFinder, load_players,
                    style_similarity, train_growth_model)

DECISION = 22
CASES = [  # (player, club, budget in €, max age)
    ("Y. Bissouma", "Brighton & Hove Albion", 25e6, 27),
    ("Casemiro", "Real Madrid", 44e6, 27),
    ("C. Eriksen", "Brentford", 15e6, 28),
    ("T. Courtois", "Real Madrid", 40e6, 30),
]


@pytest.fixture(scope="module")
def players():
    return load_players()


@pytest.fixture(scope="module")
def finder(players):
    return ReplacementFinder(players, DECISION)


def sold_player(finder, name, club):
    s = finder.season
    return s[(s.short_name == name) & (s.club_name == club)].iloc[0]


@pytest.mark.parametrize("name, club, budget, max_age", CASES)
def test_shortlist_respects_the_limits(finder, name, club, budget, max_age):
    sold = sold_player(finder, name, club)
    rec = finder.recommend(sold.player_id, budget=budget, n=20, wage_cap=sold.wage_eur,
                           max_age=max_age, min_overall=sold.overall - 12)
    assert len(rec) > 0
    assert (rec.value_eur <= budget).all(), "someone costs more than the budget"
    assert (rec.wage_eur <= sold.wage_eur).all(), "someone earns more than the wage cap"
    assert (rec.age <= max_age).all(), "someone is older than the age limit"
    assert (rec.overall >= sold.overall - 12).all(), "someone is too weak today"


@pytest.mark.parametrize("name, club, budget, max_age", CASES)
def test_never_recommends_own_players(finder, name, club, budget, max_age):
    sold = sold_player(finder, name, club)
    rec = finder.recommend(sold.player_id, budget=budget, n=20, max_age=max_age)
    assert sold.player_id not in rec.player_id.values
    assert (rec.club_team_id != sold.club_team_id).all()


@pytest.mark.parametrize("name, club, budget, max_age", CASES)
def test_candidates_play_a_compatible_role(finder, name, club, budget, max_age):
    sold = sold_player(finder, name, club)
    role = ROLE[sold.main_position]
    rec = finder.recommend(sold.player_id, budget=budget, n=20, max_age=max_age)
    for _, r in rec.iterrows():
        same_role = r.role == role
        neighbour = r.role in NEIGHBOR_ROLES[role] and sold.main_position in r.player_positions
        assert same_role or neighbour, f"{r.short_name} ({r.main_position}) can't replace a {sold.main_position}"


def test_minimum_style_match_is_respected(finder):
    sold = sold_player(finder, "C. Eriksen", "Brentford")
    rec = finder.recommend(sold.player_id, budget=15e6, n=20, max_age=28, min_fit=0.6)
    assert len(rec) > 0
    assert (rec.fit >= 0.6).all()


def test_a_player_is_most_similar_to_himself(finder):
    s = finder.season
    pool = s[s.role == "DM"].head(500)
    target = pool.iloc[0]
    sim = style_similarity(pool, target)
    assert np.isclose(sim[0], 1.0)
    assert sim.argmax() == 0


def test_no_data_leakage(players):
    """Deleting every season after the decision must not change a single prediction.

    If it did, the model would be learning from the future.
    """
    decision = 21
    past_only = players[players.fifa_version <= decision]
    season = players[players.fifa_version == decision]
    with_future = train_growth_model(players, decision).predict(season[GROWTH_FEATURES])
    without_future = train_growth_model(past_only, decision).predict(season[GROWTH_FEATURES])
    assert np.allclose(with_future, without_future)


def test_results_are_reproducible(players, finder):
    sold = sold_player(finder, "Y. Bissouma", "Brighton & Hove Albion")
    again = ReplacementFinder(players, DECISION)
    first = finder.recommend(sold.player_id, budget=25e6, n=10).player_id.tolist()
    second = again.recommend(sold.player_id, budget=25e6, n=10).player_id.tolist()
    assert first == second


# ---------------------------------------------------------------- club goals and multi-player plans

def brighton_sales(finder):
    return [sold_player(finder, "Y. Bissouma", "Brighton & Hove Albion").player_id,
            sold_player(finder, "Cucurella", "Brighton & Hove Albion").player_id]


def test_profit_columns_make_sense(finder):
    s = finder.season
    assert s.p_stay.between(0, 1).all()
    assert (s.pred_value_2y > 0).all()
    assert np.allclose(s.expected_resale, s.p_stay * s.pred_value_2y)
    assert np.allclose(s.expected_profit.dropna(), (s.expected_resale - s.value_eur).dropna())


@pytest.mark.parametrize("goal", ["replace", "squad", "profit"])
def test_plan_respects_budget_one_signing_per_player(finder, goal):
    sold_ids = brighton_sales(finder)
    plan = finder.plan(sold_ids, budget=45e6, goal=goal)
    assert len(plan) == len(sold_ids)
    assert plan.value_eur.sum() <= 45e6
    assert plan.player_id.is_unique, "the same player was bought twice"
    assert set(plan.sold_id) == set(sold_ids)


def test_plan_never_buys_same_player_twice_for_same_role(finder):
    s = finder.season
    first = s[(s.overall >= 75) & (s.league_level == 1) & (s.value_eur > 0)]
    club, role = first.groupby(["club_team_id", "role"], observed=True).size().pipe(lambda c: c[c >= 2]).index[0]
    sold_ids = first[(first.club_team_id == club) & (first.role == role)].player_id.tolist()[:2]
    plan = finder.plan(sold_ids, budget=0.6 * sum(finder.player(i).value_eur for i in sold_ids))
    assert plan.player_id.is_unique


def test_focus_moves_money_to_the_focus_position(finder):
    bissouma, cucurella = brighton_sales(finder)
    spend = lambda plan, sid: plan.loc[plan.sold_id == sid, "value_eur"].iat[0]
    neutral = finder.plan([bissouma, cucurella], budget=45e6)
    midfield = finder.plan([bissouma, cucurella], budget=45e6, focus="midfield")
    defense = finder.plan([bissouma, cucurella], budget=45e6, focus="defense")
    assert spend(midfield, bissouma) >= spend(neutral, bissouma)
    assert spend(defense, cucurella) >= spend(neutral, cucurella)


def test_joint_plan_is_never_worse_than_splitting_the_budget(finder):
    """The optimizer must reach at least the predicted gain of choosing each player separately."""
    from backtest import split_budget_picks
    sold_ids = brighton_sales(finder)
    budget = 0.6 * sum(finder.player(i).value_eur for i in sold_ids)
    plan = finder.plan(sold_ids, budget=budget, max_age=28)
    split = split_budget_picks(finder, sold_ids, 0.6, "replace", max_age=28, level_drop=12)
    predicted = lambda ids: sum(finder.player(i).pred_overall_2y for i in ids)
    assert predicted(plan.player_id) >= predicted(split) - 1e-9


def test_plan_reports_an_impossible_budget(finder):
    with pytest.raises(ValueError):
        finder.plan(brighton_sales(finder), budget=1e5)
