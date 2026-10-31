"""Replacement Finder: interactive demo for a sporting director.

Run from the project folder:
    streamlit run app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
import plot_style
from finder import FOCUS_GROUPS, GOALS, ROLE, ROLE_NAMES, ReplacementFinder, load_players, risk_flags
from plot_style import AQUA, BLUE, ORANGE, euro

plot_style.apply()
st.set_page_config(page_title="Replacement Finder", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner="Loading players…")
def get_players():
    return load_players()


@st.cache_resource(show_spinner="Training the models for this season (growth, value in 2 years, drop-out risk)…")
def get_finder(version: int) -> ReplacementFinder:
    return ReplacementFinder(get_players(), version)


def radar(rows, labels):
    stats = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
    angles = np.linspace(0, 2 * np.pi, len(stats), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={"polar": True})
    for row, label, color in zip(rows, labels, [BLUE, ORANGE, AQUA]):
        values = row[stats].astype(float).fillna(0).tolist()
        ax.plot(angles + angles[:1], values + values[:1], color=color, lw=2, label=label)
        ax.fill(angles + angles[:1], values + values[:1], color=color, alpha=0.08)
    ax.set_xticks(angles, [s.capitalize() for s in stats])
    ax.set_ylim(30, 95)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08))
    return fig


def player_table(rows: pd.DataFrame, sold_lookup) -> pd.DataFrame:
    """The columns a sporting director needs, for a shortlist or a plan."""
    return pd.DataFrame({
        "Player": rows.short_name, "Age": rows.age.astype(int), "Club": rows.club_name.astype(str),
        "Pos": rows.main_position.astype(str), "Rating": rows.overall.astype(int),
        "Predicted in 2y": rows.pred_overall_2y.round(1), "Price": rows.value_eur.map(euro),
        "Value in 2y": rows.pred_value_2y.map(euro), "Still playing in 2y": rows.p_stay.map("{:.0%}".format),
        "Style match": rows.fit.round(2),
        "Risks": [risk_flags(r, sold_lookup(r)) for _, r in rows.iterrows()],
    })


# ---------------------------------------------------------------- sidebar: the decision

st.sidebar.header("1 · Club and season")
version = st.sidebar.selectbox("Season (end of)", [22, 21, 20, 19, 23], format_func=lambda v: f"FIFA {v} (summer 20{v})")
finder = get_finder(version)
season = finder.season

clubs = season.dropna(subset=["club_name"]).groupby("club_name", observed=True).overall.mean().sort_values(ascending=False)
club = st.sidebar.selectbox("Our club", clubs.index,
                            index=int(np.flatnonzero(clubs.index == "Brighton & Hove Albion")[0])
                            if "Brighton & Hove Albion" in clubs.index else 0)
squad = season[(season.club_name == club) & (season.value_eur > 0)].sort_values("overall", ascending=False)
label = lambda pid: f"{squad.loc[pid, 'short_name']} · {squad.loc[pid, 'main_position']} · {squad.loc[pid, 'overall']:.0f}"

mode = st.sidebar.radio("What are we doing?", ["Replace one player", "Transfer plan: several players"])

st.sidebar.header("2 · Our goal")
goal = st.sidebar.radio("Club goal", list(GOALS), format_func=lambda g: GOALS[g],
                        help="Like-for-like keeps the style. Strengthen ignores style and maximizes the rating in 2 years. "
                             "Make money maximizes expected resale profit (value in 2 years × chance he is still playing − price).")

st.sidebar.header("3 · Our limits")
max_age = st.sidebar.slider("Max age", 18, 35, 27)
level_drop = st.sidebar.slider("Max rating below the sold player today", 0, 20, 12)
min_fit = st.sidebar.slider("Minimum style match", 0.0, 0.9, 0.0, step=0.05,
                            help="Raise for rare player types (e.g. a classic playmaker).")

st.title("⚽ Replacement Finder")
st.caption("We sell players. Who can replace them within our budget, for our goal, and still be good in two years? "
           "Ratings and values are from EA Sports FIFA, not real match data or real transfer fees.")

# ---------------------------------------------------------------- mode 1: one player

if mode == "Replace one player":
    sold_id = st.sidebar.selectbox("Player we sell", squad.player_id, format_func=label)
    sold = finder.player(sold_id)
    budget_m = st.sidebar.slider("Transfer budget (€M)", 0.5, 150.0,
                                 float(round(min(150.0, max(0.5, 0.7 * sold.value_eur / 1e6)), 1)), step=0.5)
    wage_k = st.sidebar.number_input("Max wage (€K / week)", 1, 600, int(min(600, max(1, sold.wage_eur / 1e3))))
    cheap = 0.0
    if goal != "profit":
        cheap = st.sidebar.slider("Prefer cheaper players", 0.0, 0.5, 0.0, step=0.05,
                                  help="In the backtest, 0.3 saved ~28% per player but lowered success from ~56% to ~41%.")
    n = st.sidebar.slider("Shortlist length", 3, 20, 10)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Selling", sold.short_name)
    c2.metric("Role", ROLE_NAMES[ROLE[sold.main_position]])
    c3.metric("Rating / age", f"{sold.overall:.0f} / {sold.age:.0f}")
    c4.metric("FIFA value", euro(sold.value_eur))
    c5.metric("Wage / week", euro(sold.wage_eur))

    rec = finder.recommend(
        sold.player_id, budget=budget_m * 1e6, n=n, goal=goal,
        weights={"fit": 0.0, "future": 1.0 - cheap, "price": cheap}, min_fit=min_fit or None,
        wage_cap=wage_k * 1e3, max_age=max_age, min_overall=sold.overall - level_drop,
    )
    if rec.empty:
        st.warning("No player fits these limits. Try a bigger budget, a higher wage cap or a lower minimum style match.")
        st.stop()

    table = player_table(rec, lambda r: sold)
    if goal == "profit":
        table.insert(9, "Expected profit", rec.expected_profit.map(lambda v: f"{euro(v)}" if v >= 0 else f"−{euro(-v)}"))
    st.subheader(f"Shortlist: {len(rec)} replacements for {sold.short_name} · {GOALS[goal]}")
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download shortlist (CSV)", table.to_csv(index=False), file_name=f"shortlist_{sold.short_name}.csv")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Compare styles")
        pick = st.selectbox("Candidate", rec.player_id, format_func=lambda pid: rec.loc[pid, "short_name"])
        st.pyplot(radar([sold, rec.loc[pick]], [f"{sold.short_name} (sold)", rec.loc[pick, "short_name"]]))
    with right:
        st.subheader("Why this player?")
        st.write(rec.loc[pick, "why"])
        st.markdown(
            "**How the Finder works**\n\n"
            "1. Candidates: same or neighbouring role, within budget, wage cap and age limit, not from our club.\n"
            "2. Like-for-like keeps the half whose skill-profile *shape* is most similar to the sold player.\n"
            "3. Ranking by the goal: predicted rating in two years, or expected resale profit.\n\n"
            "**Backtest (750 historical cases):** like-for-like ~56% of picks became real replacements "
            "vs ~43% for \"highest rating I can afford\". Make-money mode: ~€5.5M real profit per pick."
        )

    with st.expander("Check our own squad first: young players the growth model likes"):
        own = season[(season.club_team_id == sold.club_team_id) & (season.player_id != sold.player_id)
                     & (season.age <= 23)].copy()
        own["growth"] = own.pred_overall_2y - own.overall
        st.dataframe(
            own.nlargest(8, "pred_overall_2y")[["short_name", "age", "main_position", "overall", "pred_overall_2y", "growth"]]
            .round(1).rename(columns={"short_name": "Player", "age": "Age", "main_position": "Pos",
                                      "overall": "Rating", "pred_overall_2y": "Predicted in 2y", "growth": "Growth"}),
            hide_index=True, width="stretch",
        )

# ---------------------------------------------------------------- mode 2: transfer plan

else:
    default = squad.player_id.tolist()[:2]
    if club == "Brighton & Hove Albion" and version == 22:   # the real summer-2022 sales
        default = squad[squad.short_name.isin(["Y. Bissouma", "Cucurella"])].player_id.tolist()
    sold_ids = st.sidebar.multiselect("Players we sell (2–4)", squad.player_id.tolist(), default=default,
                                      format_func=label, max_selections=4)
    focus = st.sidebar.selectbox("Focus of the team's development", [None, *FOCUS_GROUPS],
                                 format_func=lambda f: "No focus" if f is None else f"Improve {f}",
                                 help="Positions in the focus count double, so the plan spends more money there.")
    total_value = sum(finder.player(i).value_eur for i in sold_ids) if sold_ids else 10e6
    budget_m = st.sidebar.slider("Total budget for all signings (€M)", 1.0, 300.0,
                                 float(round(min(300.0, max(1.0, 0.7 * total_value / 1e6)), 1)), step=0.5)

    if len(sold_ids) < 1:
        st.info("Choose the players you are selling in the sidebar.")
        st.stop()

    try:
        plan = finder.plan(sold_ids, budget=budget_m * 1e6, goal=goal, focus=focus,
                           max_age=max_age, level_drop=level_drop, min_fit=min_fit or None)
    except ValueError as err:
        st.warning(str(err))
        st.stop()

    spent = plan.value_eur.sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players sold", len(sold_ids))
    c2.metric("Spent", euro(spent), f"{spent / (budget_m * 1e6):.0%} of budget", delta_color="off")
    c3.metric("Left over", euro(budget_m * 1e6 - spent))
    if goal == "profit":
        c4.metric("Expected profit in 2 years", euro(plan.expected_profit.sum()))
    else:
        c4.metric("Avg predicted rating in 2y", f"{plan.pred_overall_2y.mean():.1f}",
                  f"{(plan.pred_overall_2y - plan.sold_overall).mean():+.1f} vs sold", delta_color="normal")

    st.subheader(f"Transfer plan · {GOALS[goal]}" + (f" · focus: {focus}" if focus else ""))
    sold_lookup = lambda r: finder.player(r.sold_id)
    table = player_table(plan, sold_lookup)
    table.insert(0, "Replaces", [f"{r.sold_short_name} ({r.sold_main_position}, {r.sold_overall:.0f}, {euro(r.sold_value_eur)})"
                                 for _, r in plan.iterrows()])
    if goal == "profit":
        table.insert(10, "Expected profit", plan.expected_profit.map(lambda v: f"{euro(v)}" if v >= 0 else f"−{euro(-v)}"))
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download plan (CSV)", table.to_csv(index=False), file_name=f"transfer_plan_{club}.csv")

    st.markdown(
        "**How the plan is made:** an optimizer (integer program) picks exactly one signing for every sold player, "
        "never the same player twice, keeps the total within the budget, and maximizes the goal. A focus position "
        "counts double, so more of the budget goes there. **Backtest:** on 450 clubs selling 3 players, the joint plan "
        "found real replacements 67% of the time, against 60% when the budget is split per player by hand."
    )

    with st.expander("Alternatives for each sold player"):
        for _, row in plan.iterrows():
            pool = finder.goal_pool(row.sold_id, budget_m * 1e6, goal, min_fit=min_fit or None,
                                    wage_cap=finder.player(row.sold_id).wage_eur, max_age=max_age,
                                    min_overall=row.sold_overall - level_drop)
            st.markdown(f"**Instead of {row.short_name} for {row.sold_short_name}**")
            alt = pool.nlargest(6, "gain")
            st.dataframe(player_table(alt, lambda r: finder.player(row.sold_id)), hide_index=True, width="stretch")
