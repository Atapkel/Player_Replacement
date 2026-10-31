# How to test and use the model

Our "model" is really two parts working together: a **style similarity** (who plays the same way) and a
**growth model** (gradient boosting that predicts a player's rating two seasons later). The growth model isn't saved
to a file. It's retrained in about 10 seconds every time you start a tool, always using only data that existed at the
chosen moment in time. That keeps every test honest.

There are four ways to test it, from the easiest to the most technical. All commands are run from the project folder
(`ai_essentials_4/`), with the project's Python environment active.

## Before you start (once)

```bash
pip install -r requirements.txt     # or: uv sync, from the parent folder
python src/download_data.py         # only if data/raw/ is empty (~6 GB)
python src/prepare_data.py          # only if data/processed/players.parquet is missing
```

## 1. The app: click and see

```bash
streamlit run app.py
```

A browser tab opens. On the left you choose the season, your club and the player you sell, then set the budget, wage
cap and age limit. The page shows the shortlist, a radar chart comparing styles, and young players in your own squad
the model likes.

**Things to try:**
- Brighton → Y. Bissouma, budget 25: the shortlist from our presentation.
- Move the budget to 20: Rosario moves to #1.
- Brentford → C. Eriksen: look at the low "Style match" values, then raise *Minimum style match* to 0.6.
- Your own favorite club and player. Does the list make football sense to you?

## 2. The command line: one question, one answer

`examples/find_replacement.py` does the same as the app, in the terminal.

```bash
# Who could replace Bissouma at Brighton in summer 2022, with €25M?
python examples/find_replacement.py --player "Y. Bissouma" --club "Brighton & Hove Albion" --budget 25

# Real Madrid sells Casemiro, €44M
python examples/find_replacement.py --player "Casemiro" --club "Real Madrid" --budget 44

# A rare playmaker: demand a closer style match
python examples/find_replacement.py --player "C. Eriksen" --club "Brentford" --budget 15 --max-age 28 --min-fit 0.6

# Don't know the exact names? List a squad first
python examples/find_replacement.py --club "Arsenal" --list-squad

# Go back in time: summer 2020 instead of 2022
python examples/find_replacement.py --player "B. Saka" --club "Arsenal" --season 20 --budget 20
```

Names must be written as in FIFA (`"B. Saka"`, not `"Saka"`). If you make a typo, the script suggests the closest names.

**Sample output:**

```
Selling Y. Bissouma (Brighton & Hove Albion): CDM, age 24, rated 81, value €35.5M
Budget €25M, max wage €57K/week, max age 27

        player  age                club pos  now  in 2y  value  style        risks
 N. Maksimović   26              Getafe CDM   79   79.0 €20.5M   0.61            -
      J. Weigl   25             Benfica  CM   79   78.7   €23M   0.63            -
        D. Sow   24 Eintracht Frankfurt  CM   79   78.5   €24M   0.67            -
      R. Tapia   25       Celta de Vigo CDM   79   78.4   €22M   0.60 bench player
    P. Rosario   24                Nice CDM   78   77.8 €18.5M   0.79            -
```

**How to read it:** `now` is today's rating; `in 2y` is the growth model's prediction; `style` is the style match with
the sold player (1.0 = identical shape of skills); `risks` are warnings (bench player, weaker league, contract ending,
little growth left). The list is sorted by `in 2y`, among players whose style is in the closest half.

| Option | Meaning | Default |
|---|---|---|
| `--season` | decision at the end of FIFA 19–23 | 22 (summer 2022) |
| `--budget` | transfer budget in € millions | 70% of the sold player's value |
| `--wage` | max wage in € thousands per week | the sold player's wage |
| `--max-age` | oldest player allowed | 27 |
| `--min-fit` | minimum style match (0–1) | none |
| `--cheaper` | 0 = best future rating, up to 0.5 = prefer cheaper players | 0 |
| `-n` | shortlist length | 10 |

## 2b. Transfer plans: several players, one budget, a goal and a focus

`examples/plan_transfers.py` replaces several sold players at once. An optimizer splits the shared budget.

```bash
# Brighton sold Bissouma and Cucurella in summer 2022. €45M for both replacements:
python examples/plan_transfers.py --club "Brighton & Hove Albion" --sell "Y. Bissouma" --sell "Cucurella" --budget 45

# Same, but the club wants to strengthen its midfield first:
python examples/plan_transfers.py --club "Brighton & Hove Albion" --sell "Y. Bissouma" --sell "Cucurella" --budget 45 --focus midfield

# Same, but the goal is to make money (buy players whose value should grow):
python examples/plan_transfers.py --club "Brighton & Hove Albion" --sell "Y. Bissouma" --sell "Cucurella" --budget 45 --goal profit
```

**Sample output (focus midfield):**

```
   replaces        player  age        club pos  now  in 2y  price value 2y stays  style        risks
Y. Bissouma   I. Bennacer   23       Milan CDM   81   80.9 €33.5M   €28.4M  100%   0.66 bench player
  Cucurella T. Livramento   18 Southampton  RB   74   76.9  €8.5M   €16.1M   98%   0.74 bench player

Spent €42M of €45M; expected profit in 2 years €1.98M; average predicted rating in 2 years 78.9
```

Goals: `replace` (same style, the default), `squad` (any style, best rating in 2 years), `profit` (expected resale
profit). Focus: `attack`, `midfield`, `defense` (those positions count double). In the app, choose
**"Transfer plan: several players"** in the sidebar.

## 3. Check the predictions against reality

```bash
python examples/check_predictions.py
python examples/check_predictions.py --player "Pedri" --player "J. Bellingham" --player "B. Saka"
```

This is the most important test. The script pretends it is **summer 2021**. It trains the growth model only on
what was known then, predicts every player's rating two seasons ahead, and compares with the **real** FIFA 23 ratings,
which the model has never seen.

**Sample output:**

```
Test set: 13,488 players from FIFA 21 who are still in FIFA 23

Average error (rating points, lower is better)
  'Nothing changes' baseline : 2.33
  Our growth model           : 2.03

Who will grow? Rank correlation with real growth, players aged 23 or younger (higher is better)
  EA's potential - overall   : 0.34
  Our growth model           : 0.42

   short_name  age         club_name  FIFA 21  potential  predicted  real FIFA 23
      B. Saka 18.0           Arsenal     79.0       88.0       81.7          83.0
 E. Camavinga 17.0            Rennes     78.0       89.0       79.7          79.0
        Pedri 17.0      FC Barcelona     77.0       91.0       79.2          85.0
J. Bellingham 17.0 Borussia Dortmund     76.0       88.0       78.5          85.0
```

**How to read it:** on unseen data the model is about 13% more accurate than assuming nothing changes, and it ranks
young players' growth better than EA's own potential rating. The same advantages appeared in notebook 03 for earlier
seasons, so the result is stable.

The table also shows the model's honest weakness: it was **too cautious about superstars**. Pedri and Bellingham grew
by 8–9 points, while the model expected about 2–3. The model learns from what *typical* players do, so exceptional
development is the hardest thing to predict. That's a good answer if the teacher asks "where does your model fail?".

## 4. Automatic checks (pytest)

```bash
pytest -v tests/
```

24 checks run in about a minute. They test that the Finder follows its own rules:

| Check | What it proves |
|---|---|
| Shortlist respects the limits (4 cases) | Nobody above the budget, the wage cap or the age limit, or too weak today |
| Never recommends own players (4 cases) | The sold player and his teammates are excluded |
| Compatible role (4 cases) | A centre-back is never offered as a defensive midfielder, a keeper only gets keepers |
| Minimum style match | With `min_fit=0.6`, every candidate really has a style match ≥ 0.6 |
| A player is most similar to himself | The similarity measure is mathematically sane (cosine = 1 with itself) |
| **No data leakage** | Deleting every season after the decision date doesn't change a single prediction. If the model used the future, this test would fail |
| Reproducible | Training twice gives exactly the same shortlist |
| Profit columns make sense | Stay probability between 0 and 1; expected profit = stay chance × value in 2 years − price |
| Plan respects the budget (3 goals) | One signing per sold player, total within budget, nobody bought twice |
| Same role sold twice | Two sold players of the same role never get the same replacement |
| Focus moves money | With focus "midfield", more is spent on the midfielder's replacement (and the same for defense) |
| Joint plan ≥ split budget | The optimizer never predicts worse than choosing each player with his own budget share |
| Impossible budget | A budget too small for any combination gives a clear error |

Expected result: `24 passed`.

## Experiments to prepare for the defense

The teacher may ask you to change a parameter and explain what happens. Try these yourself first:

1. **Budget −20%:** run the Bissouma case with `--budget 25` and `--budget 20`. Which names stay? (Rosario stays and moves to #1.)
2. **Age limit:** `--max-age 23` gives a younger list. Do the predicted ratings go up or down compared with today's ratings? Why?
3. **Prefer cheaper:** add `--cheaper 0.3`. Prices drop, and so do the predicted ratings. In the backtest this saved ~28% per player but lowered success from ~56% to ~41%.
4. **Style filter:** Eriksen with and without `--min-fit 0.6`. Why does a rare player type need a stricter filter?
5. **Another season:** `--season 20` for any player. The model is then trained only on data up to 2020.
6. **Your own player:** pick a player you know well and judge whether the shortlist makes football sense. If it doesn't, find out why (style match? league? bench player?).
