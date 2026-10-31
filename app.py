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
from finder import ROLE, ROLE_NAMES, ReplacementFinder, load_players, risk_flags
from plot_style import AQUA, BLUE, ORANGE, euro

plot_style.apply()
st.set_page_config(page_title="Replacement Finder", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner="Loading players…")
def get_players():
    return load_players()


@st.cache_resource(show_spinner="Training the growth model for this season…")
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


# ---------------------------------------------------------------- sidebar: the decision

st.sidebar.header("1 · Who are we selling?")
version = st.sidebar.selectbox("Season (end of)", [22, 21, 20, 19, 23], format_func=lambda v: f"FIFA {v} (summer 20{v})")
finder = get_finder(version)
season = finder.season

clubs = season.dropna(subset=["club_name"]).groupby("club_name", observed=True).overall.mean().sort_values(ascending=False)
club = st.sidebar.selectbox("Our club", clubs.index,
                            index=int(np.flatnonzero(clubs.index == "Brighton & Hove Albion")[0])
                            if "Brighton & Hove Albion" in clubs.index else 0)
squad = season[season.club_name == club].sort_values("overall", ascending=False)
sold_name = st.sidebar.selectbox(
    "Player we sell", squad.player_id,
    format_func=lambda pid: f"{squad.loc[pid, 'short_name']} · {squad.loc[pid, 'main_position']} · {squad.loc[pid, 'overall']:.0f}",
)
sold = finder.player(sold_name)
sold_value = sold.value_eur if pd.notna(sold.value_eur) else 10e6   # free agents have no FIFA value
sold_wage = sold.wage_eur if pd.notna(sold.wage_eur) else 50e3

st.sidebar.header("2 · Our limits")
budget_m = st.sidebar.slider("Transfer budget (€M)", 0.5, 150.0,
                             float(round(min(150.0, max(0.5, 0.7 * sold_value / 1e6)), 1)), step=0.5)
wage_k = st.sidebar.number_input("Max wage (€K / week)", 1, 600, int(min(600, max(1, sold_wage / 1e3))))
max_age = st.sidebar.slider("Max age", 18, 35, 27)
level_drop = st.sidebar.slider("Max rating below the sold player today", 0, 20, 12)

st.sidebar.header("3 · What matters to us")
use_filter = st.sidebar.checkbox("Only players with a similar style (top half)", value=True,
                                 help="The backtest winner. Switch off to rank by future rating only.")
min_fit = st.sidebar.slider("Minimum style match", 0.0, 0.9, 0.0, step=0.05,
                            help="Raise for rare player types (e.g. a classic playmaker).")
cheap = st.sidebar.slider("Prefer cheaper players", 0.0, 0.5, 0.0, step=0.05,
                          help="0 = best future rating. Higher = cheaper players. In the backtest, 0.3 saved ~28% per player but lowered success from ~56% to ~41%.")
n = st.sidebar.slider("Shortlist length", 3, 20, 10)

# ---------------------------------------------------------------- main page

st.title("⚽ Replacement Finder")
st.caption("We sell a player. Who can replace him cheaply, in the same style, and still be good in two years? "
           "Ratings and values are from EA Sports FIFA, not real match data or real transfer fees.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Selling", sold.short_name)
c2.metric("Role", ROLE_NAMES[ROLE[sold.main_position]])
c3.metric("Rating / age", f"{sold.overall:.0f} / {sold.age:.0f}")
c4.metric("FIFA value", euro(sold.value_eur))
c5.metric("Wage / week", euro(sold.wage_eur))

rec = finder.recommend(
    sold.player_id, budget=budget_m * 1e6, n=n,
    weights={"fit": 0.0, "future": 1.0 - cheap, "price": cheap},
    fit_quantile=0.5 if use_filter else 0.0, min_fit=min_fit or None,
    wage_cap=wage_k * 1e3, max_age=max_age, min_overall=sold.overall - level_drop,
)

if rec.empty:
    st.warning("No player fits these limits. Try a bigger budget, a higher wage cap or a lower minimum style match.")
    st.stop()

rec["risks"] = [risk_flags(r, sold) for _, r in rec.iterrows()]
table = pd.DataFrame({
    "Player": rec.short_name, "Age": rec.age.astype(int), "Club": rec.club_name.astype(str),
    "League": rec.league_name.astype(str), "Pos": rec.main_position.astype(str),
    "Rating": rec.overall.astype(int), "Predicted in 2y": rec.pred_overall_2y.round(1),
    "Value": rec.value_eur.map(euro), "Wage/wk": rec.wage_eur.map(euro),
    "Style match": rec.fit.round(2), "Risks": rec.risks,
})
st.subheader(f"Shortlist: {len(rec)} replacements for {sold.short_name}")
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
        "2. Style filter: keep the half whose skill-profile *shape* is most similar to the sold player.\n"
        "3. Ranking: predicted rating in two years (gradient boosting trained only on past seasons).\n\n"
        "**Backtest (750 historical cases):** ~56% of picks became real replacements within two years, "
        "vs ~43% for \"highest rating I can afford\"."
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
