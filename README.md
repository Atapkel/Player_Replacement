# Replacement Finder: Moneyball scouting with machine learning

**We sell a player. Who can replace him cheaply, in the same playing style, and still be good in two years?**

A decision-support tool for a football club with a limited budget. Given the player a club is selling, the
budget, the wage cap and an age limit, it recommends replacements that (1) play in a similar style and (2) are
predicted to be the strongest in two years. It is validated by a historical backtest over 750 transfer decisions.

AI Essentials, Lab work #2, Scenario 3 (Moneyball 2.0) with elements of Scenario 5 (a product for a sporting director).

## Key results

| | Finder | "Highest rating I can afford" | "Highest potential" | Random |
|---|---|---|---|---|
| Picks that became real replacements within 2 years* | **56%** | 43% | 44% | 11% |
| Picks that dropped out of the FIFA database within 2 years | 13% | 20% | 7% | 32% |
| Style similarity to the sold player | **0.74** | 0.60 | 0.64 | 0.62 |
| Average price | €3.8M | €3.9M | €2.6M | €1.4M |

\*Still in the game and at most 3 rating points below the sold player, two seasons later. Budget = 60% of the sold
player's value. 750 cases from summers 2019–2021. The advantage over both simple rules is significant (95% bootstrap
interval for the difference vs. "highest rating": +10.9 to +15.4 points), and it holds in every season, under stricter
or looser success definitions, and with a 20% smaller budget.

**Showcase:** in summer 2019, replacing a Chicago Fire striker on a €2.3M budget, one of the Finder's five picks was a
17-year-old rated 68 for €1.8M: Erling Haaland (rated 87 two years later).

**Summer 2022 decisions** (notebook 05):

| Case | Finder's recommendation | What happened in reality |
|---|---|---|
| Brighton sells Bissouma, €25M | Pablo Rosario (Nice), best style match, €18.5M | Brighton promoted its own Caicedo (whom the growth model also liked) |
| Real Madrid sells Casemiro, €44M | Palhinha (best style match) or Højbjerg | Real paid ~€80M+ for Tchouaméni; **Fulham bought Palhinha that summer** |
| Brentford loses Eriksen, €15M | Denis Suárez or Kahveci (needed a stricter style filter) | No like-for-like replacement |

## Main findings, including the negative ones

1. **A growth model beats EA's "potential" at predicting who will improve.** Gradient boosting predicts the rating
   two seasons ahead with an error of 2.28 points vs 2.59 for "no change", and ranks growth better than EA potential,
   especially for players aged ≤ 23 (Spearman 0.42 vs 0.35).
2. **EA's market value is essentially a formula.** Rating, potential, age and role predict it within ~2%. So there are no
   "hidden bargains" in EA prices, and real-market undervaluation can't be measured with game data. The only way to beat
   a formula price is to predict the future better than the formula. That's what the growth model does.
3. **"Cheap leagues" are an illusion caused by confounding.** Players in some leagues look 2× cheaper at the same rating,
   but the gap disappears once age and potential are accounted for (±0.2%).
4. **Playing styles are a continuum, not clusters** (silhouette ≈ 0.22 at best). Clusters work as labels (Creator,
   Ball-winner, Positional defender), but replacement search uses continuous cosine similarity of skill-profile shapes. It
   recognizes the same player one season later as the #1 match among 6,000+ midfielders 86% of the time.
5. **Our first design was wrong, and the backtest showed it.** Mixing 40% style similarity into the score gave only 36%
   success. Using style as a *filter* and the growth model for *ranking* gave 56%.

## How it works

```
all players in the season (FIFA ratings, end of season)
        │  same or neighbouring role, within budget / wage cap / age limit, not our own players
        ▼
candidate pool
        │  style filter: keep the half most similar to the sold player
        │  (cosine similarity of skill-profile shape, so level doesn't matter)
        ▼
style-matching candidates
        │  rank by predicted rating in 2 years
        │  (gradient boosting, trained only on seasons known at the decision date)
        ▼
shortlist + explanation + risk flags (bench player, weaker league, contract ending, little growth left)
```

## Data

[FIFA 23 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset)
(Kaggle, Stefano Leone): EA Sports FIFA 15–23 ratings, about 10 million rows (one per player per weekly update),
5.6 GB. `src/prepare_data.py` reduces it to two snapshots per player per season (start and end), 393,866 rows, 29 MB.
The analysis uses the end-of-season snapshot, about 20–24k players per season.

## Project structure

```
src/
  download_data.py     download the Kaggle dataset (no login needed)
  prepare_data.py      5.6 GB CSV -> 29 MB parquet (chunked, ~2.5 min, <2 GB RAM)
  finder.py            the Replacement Finder: style similarity, growth model, scoring, risk flags
  backtest.py          historical backtest and bootstrap confidence intervals
  plot_style.py        shared chart style
notebooks/
  01_data_overview.ipynb           what's in the data, quality checks, pitfalls (selection bias, confounding, survivorship)
  02_playing_styles.ipynb          style profiles, clustering, similarity and its validation
  03_growth_and_value_models.ipynb growth model vs baselines; why EA prices hide no bargains
  04_backtest.ipynb                750 historical decisions vs simple rules, robustness checks
  05_recommendations.ipynb         summer 2022 decisions for Brighton, Real Madrid, Brentford
app.py                 Streamlit demo for a sporting director
examples/
  find_replacement.py  ask for a shortlist from the command line
  check_predictions.py test the growth model against real FIFA 23 ratings it never saw
tests/
  test_finder.py       16 automatic checks (limits, roles, no data leakage, reproducibility)
docs/
  how_it_works.md           the whole system explained from A to Z, with one example followed through
  how_it_works_ru.md        the same explanation in Russian
  Replacement_Finder_explained_ru.pptx   31-slide visual explanation in Russian (for learning, not presenting)
  speaker_scripts_ru.md     the speaker scripts in Russian
  defense_cheatsheet_ru.md  defense cheat sheet in Russian: all ML methods, metrics, numbers, answers
  how_to_test.md            how to use and test the model, with sample outputs
  Replacement_Finder.pptx   the presentation (13 slides, speaker notes included)
  speaker_scripts.md        the talk split into four speakers' scripts
  presentation_outline.md   15-minute talk plan
  defense_questions.md      likely questions and answers for the personal defense
```

## How to run

```bash
pip install -r requirements.txt
python src/download_data.py      # ~6 GB into data/raw/
python src/prepare_data.py       # -> data/processed/players.parquet
jupyter lab notebooks/           # run 01 -> 05 in order (04 caches its backtest, ~3 min first run)
streamlit run app.py             # interactive demo
python examples/find_replacement.py --player "Y. Bissouma" --club "Brighton & Hove Albion" --budget 25
python examples/check_predictions.py   # predictions vs reality on unseen data
pytest -v tests/                       # 16 automatic checks
```

See `docs/how_to_test.md` for a full guide with sample outputs.

## Pitfalls we found and how we handled them

| Pitfall (from the task) | Where | What we did |
|---|---|---|
| Data leakage | 03, 04 | Time-based splits: the growth model only learns from seasons whose outcome was known at the decision date |
| Different positions | 02, finder | Candidates only from the same or neighbouring role; goalkeepers use their own skills |
| Different leagues | 01, 03, 05 | Found selection bias (some leagues include only top clubs) and confounding (league price gaps = age mix); "weaker league" risk flag |
| Correlation ≠ causation | 01, 03 | League price gaps disappear once age and potential are controlled |
| Not enough playing time | 05 | No minutes data; "bench player" flag as a proxy |
| Outdated data | 03 | Growth model ~0.5–1 point optimistic in recent seasons (EA's rating style changed); ranking unaffected |
| Selection / survivorship bias | 01, 04 | 38% of players leave the game within 2 years; counted as failures in the backtest |
| Uncertainty | 04, 05 | Bootstrap intervals; early reality check (our Brighton pick dropped 2 points by Jan 2023) |
| Game ratings vs real market | 03, 05 | EA value ≈ formula, not a real fee (Tchouaméni: FIFA €52M vs ~€80M+ real); shortlist = input for human scouts |

## Limitations

- Ratings and values are EA's opinion, not match events or real fees.
- FIFA 23 ends in January 2023, so the last backtest season and the 2022 reality check are mid-season.
- The tool replaces one player at a time; it doesn't plan a whole squad or a long-term strategy.
- Rare player types (e.g. classic playmakers) need a stricter style filter, chosen by a human.

## Use of AI tools

This project was built with Claude (Anthropic) as a coding and analysis assistant.

- **What it was used for:** proposing the project design, writing the data pipeline, the `src/` modules, the notebooks
  and the Streamlit app, running experiments, and drafting explanations.
- **What it was given:** the assignment text, the Kaggle dataset (it worked directly with the files), and our decisions
  (scenario, product idea "sell a player, get cheap replacements", scope, that it should write the code with explanations).
- **How results were checked:** every number in the notebooks comes from executed code, not from the model's memory.
  Claims were checked against outputs, and several were corrected after checking: the "league affects price" claim in
  notebook 01 turned out to be confounding, the value model's error was 2% not 25%, and the first scoring design was
  rejected by the backtest. Real-world facts (e.g. Palhinha's move to Fulham, Bissouma's to Tottenham) were confirmed in
  the FIFA 23 data; reported transfer fees (Tchouaméni ~€80M+) are approximate and marked as such.
- **Risks noticed:** an AI assistant can state plausible but wrong interpretations. Several of its first interpretations
  were wrong and only the executed output revealed it. Football knowledge in the text (player roles, transfers) should be
  verified by the team.
- **Done or verified by the team:** _(fill in: who studied/re-ran which notebook, which parameters you changed, which
  results you checked by hand)_.
