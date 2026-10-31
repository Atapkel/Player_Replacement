# How the Replacement Finder works, from A to Z

This guide explains the whole project in plain words, step by step, from the raw data file to the shortlist on the
screen and the test that proves it works. You don't need to read the code first. Where it helps, the guide names the
file and function that does each step, so you can open it afterwards and recognize what you see.

We follow **one real example** the whole way through: in summer 2022, Brighton sells its defensive midfielder Yves
Bissouma to Tottenham, and wants a replacement for at most €25M.

---

## The big picture in one paragraph

We start with ten years of FIFA player ratings. For the player a club is selling, we first collect every player who
could play his role and fits the club's money limits. From those, we keep only the ones whose **playing style** is
closest to the sold player's. That part uses unsupervised machine learning, because nobody tells the computer what a
"similar style" is. It measures it from the skills. Then we **predict how good each of them will be in two years**
with a model trained on the past. That part is supervised machine learning: the model learns from thousands of
examples where we already know the answer. We sort the candidates by that prediction and show the best ones, with
reasons and warnings. Finally, to know whether any of this deserves trust, we **replay 750 transfer decisions from
the past** and check what really happened to the players the system would have bought.

Here is the journey of our example, in numbers. Remember the shape; every section below explains one step of it:

```
23,803 players in FIFA 22
   → 3,055 can play Bissouma's role (not from Brighton, with a market value)
   →   535 fit the budget, the wage cap, the age limit and the minimum level
   →   268 play in the closest style (the better half)
   →     5 with the best predicted rating in two years = the shortlist
```

---

## Part 1 · The data

### A. What we start with

The data is the *FIFA 23 Complete Player Dataset* from Kaggle. EA Sports, the company that makes the FIFA video
games, gives every licensed professional player numbers from 0 to 100: an **overall** rating (how good he is now), a
**potential** rating (how good EA thinks he can become), six summary stats (pace, shooting, passing, dribbling,
defending, physical) and about 35 **detailed skills**, such as short passing, interceptions or standing tackle. The
data also contains his age, club, league, position, market value, weekly wage and when his contract ends.

The file covers FIFA 15 to FIFA 23, which is roughly the seasons 2014/15 to 2022/23. It is huge, 5.6 GB with ten
million rows, for a simple reason: EA updates the ratings almost every week during a season, and the file stores
every single update. The same player appears about sixty times per season.

### B. Making the data usable (`src/prepare_data.py`)

We don't need sixty weekly versions of each player, so the cleaning script keeps only **two snapshots per player per
season**: the **start** of the season (the first update, around the game's release in September) and the **end** of
the season (the last update, close to the summer transfer window).

The file is too big to load into memory in one go, so the script reads it in **chunks** of 500,000 rows. For each
chunk it keeps the earliest and the latest update of every player, and at the end it repeats this once more over all
the chunks together, because one player's updates can be split between two chunks. It also reads only the columns we
need (no photo links, for example) and stores ratings as small 32-bit numbers. Thanks to this, the whole job takes
about two and a half minutes and never needs more than 2 GB of memory.

While it's at it, the script adds a few columns of its own:

- `season_start`: the real year the season starts. FIFA 22 is the 2021/22 season, so 2021.
- `main_position`: the first position in the player's list. Bissouma is listed as "CDM, CM", so his main position is CDM (defensive midfielder).
- `contract_years_left`: the contract end year minus the season start.

The result is saved as `data/processed/players.parquet`: 393,866 rows in 29 MB. Parquet is a compact file format
that pandas reads in a second.

### C. Getting it ready for the model (`load_players` in `src/finder.py`)

Every tool in the project starts by calling `load_players()`. It does three things.

First, it keeps only the **end-of-season** snapshot, because that is what a club knows when the summer transfer
window opens. That leaves 196,933 rows: one per player per season.

Second, it turns the 15 FIFA positions into **8 roles**: goalkeeper, centre-back, full-back, defensive midfielder,
central midfielder, attacking midfielder, winger and striker. Left-back and right-back are the same job on different
sides, so they share the role "full-back"; centre-back is a different job, even though both are defenders.

Third, it computes **league strength**: the average rating of all players in that league in that season. It's a rough
measure (some leagues only have their best clubs in the game, as we saw in notebook 01), but it gives the model a hint
about the level a player is used to.

---

## Part 2 · Finding candidates

### D. The candidate pool (`ReplacementFinder.candidates`)

Now the club tells us who it sells and what its limits are. A candidate must pass all of these rules:

1. **He can play the role.** Either he has the same role as the sold player, or he comes from a neighbouring role and
   lists the sold player's exact position among his positions. For Bissouma (a defensive midfielder) that means all
   defensive midfielders, plus central midfielders who also play CDM. A centre-back who once played CDM is *not*
   accepted, because the two jobs are too different. The neighbouring roles are written down in `NEIGHBOR_ROLES`.
2. **He isn't one of our own players.** You can't buy your own player.
3. **He has a market value**, and it is **within the budget**: here at most €25M.
4. **His wage is at most the wage cap.** By default that's the sold player's wage: €57K per week for Bissouma.
5. **He is not older than the age limit** (27 here, so that he still has resale value).
6. **He is not much weaker today**: at most 12 rating points below the sold player. Bissouma is rated 81, so every candidate is rated at least 69.

In our example, 3,055 players can play the role and aren't from Brighton. After the money, age and level limits,
**535** are left. These rules are ordinary business logic, not machine learning, but they matter: they make sure the
model only ever ranks players the club could actually sign.

---

## Part 3 · Ingredient one: playing style (unsupervised learning)

### E. Turning skills into a "style profile"

Two players can both be defensive midfielders and still play very differently: one wins the ball with hard tackles,
another reads the game and intercepts passes, a third mostly passes. We want a replacement who does **the same job**.

Each player has 29 detailed outfield skills. If we compared these numbers directly, a great player and a weak player
would always look different, simply because all of the great player's numbers are higher. But a cheaper replacement
will almost always be a bit weaker, and we don't want that to hide a similar style.

So we look at the **shape** of the skills, not their height. For each player we take his average skill and subtract
it from every skill. What remains says what he is *relatively* good and bad at. This is the **style profile**.

Bissouma's average skill is 74.7. His style profile shows that, compared with his own average, he is strongest at
standing tackle (+10.3), shot power and interceptions (+8.3 each) and short passing (+7.3), and weakest at finishing
and penalties (−16.7 each) and curve (−12.7). That reads like a ball-winning midfielder who passes well and rarely
scores, which is how football people describe him.

### F. Measuring how similar two styles are: cosine similarity (`style_similarity`)

Picture each style profile as an arrow in a space with 29 directions, one direction per skill. Two players with the
same style have arrows pointing the same way. **Cosine similarity** measures the angle between two arrows:

- **1** means exactly the same direction: identical style;
- **0** means unrelated styles;
- **−1** means opposite styles: one player's strengths are the other's weaknesses.

A tiny example with only three skills (tackling, passing, shooting):

- Player A: 80, 70, 60. Average 70, so his profile is +10, 0, −10.
- Player B: 70, 60, 50. Average 60, so his profile is also +10, 0, −10. B is weaker, but the cosine with A is **1**: the same style.
- Player C: 60, 70, 80. Average 70, so his profile is −10, 0, +10. Cosine with A is **−1**: a shooter, not a tackler.

Because we subtracted each player's own average first, the length of the arrow (his level) doesn't count, only its
direction. That's exactly the "shape, not height" idea. Pablo Rosario of Nice has a lower average skill (70.2) than
Bissouma, but his cosine similarity with Bissouma is **0.79**, the highest on our final shortlist.

Goalkeepers are compared on their own set of skills (diving, handling, reflexes and so on), because outfield skills
mean little for a keeper.

### G. Why not clustering?

The obvious first idea was **clustering**: let K-Means sort midfielders into groups such as "ball-winners" and
"playmakers", and look for replacements inside the sold player's group. We tried it (notebook 02). The groups do make
football sense, but the **silhouette score** was only 0.22. The silhouette compares how close each player is to his
own group with how close he is to the nearest other group; near 0 means the groups overlap. Styles turned out to be a
**continuum**, not separate boxes. Two players at opposite edges of one group can play quite differently, so a group
label is too rough for choosing a replacement. A continuous score like cosine similarity can rank every candidate.

### H. Can we trust the similarity? The self-recognition test

A player's style changes little from one season to the next. So we took 500 midfielders' FIFA 22 profiles and searched
for the most similar player among **all 6,064 midfielders in FIFA 21**. If the measure works, the best match should be
the same player one year earlier. It was, **86% of the time** (and he was in the top 10 in 92% of cases). Plain
distance on the raw skills did worse (84%), even though it has the unfair help of the player's level, which barely
changes in a year.

### I. The style filter (`keep_good_fit`)

Every candidate in the pool now has a style score. We keep the **better half**: everyone whose score is at least the
pool's median. In our example the median is 0.58, so 268 of the 535 candidates stay.

Why a filter and not a weight in the final score? We tried both, and the backtest decided (Part 6): mixing style into
the score made the picks worse, because it traded too much quality for similarity. As a filter, style removes players
who would do a different job, and the quality of the picks barely suffers.

For rare player types the median isn't strict enough. For a classic playmaker like Eriksen, half of all candidates
have almost nothing in common with him (the median style score in his pool is about 0.10). So the user can also set a
**minimum style match** (`min_fit`), for example 0.6.

---

## Part 4 · Ingredient two: the future (supervised learning)

### J. What we want to predict

Style tells us who *fits*. It doesn't tell us who will be *good*. The most similar players to Casemiro, for example,
were mostly over 30, so they would get worse soon. A club buys a player for the next seasons, not for the last one.

So the second ingredient answers: **what will this player's overall rating be two seasons from now?**

### K. Learning from the past: training examples (`growth_pairs`)

This is **supervised learning**: we show the computer many examples where we already know the answer, and it learns
the pattern. Each example is one player in one season, together with his rating two seasons later. For instance:
"Mbappé in FIFA 17: 17 years old, rated 79, at Monaco, … → rated 88 in FIFA 19."

The function `growth_pairs` builds these examples by joining every player-season with the same player two seasons
later. Players who disappeared from the game have no "two seasons later", so they can't be used for training. The
model therefore learns "how good will he be, *if he's still playing*". The backtest deals with the players who
disappear (Part 6).

For each example the model sees **45 features**: overall, potential, age, international reputation, league level and
league strength, contract years left, weak foot, skill moves, height, role, and the detailed skills (including the
five goalkeeper skills, which matter for keepers).

### L. The golden rule: never use the future (data leakage)

Imagine we want to make a decision in summer 2022. At that moment we know the FIFA 22 ratings, but not FIFA 23's. If
the model were trained on an example like "FIFA 21 → FIFA 23", it would be using information from the future that no
real club could have. That's called **data leakage**, and it makes a model look brilliant in testing and useless in
real life.

So `train_growth_model` only uses examples whose "two seasons later" is already known at the decision date: the
starting season plus 2 must be at most the decision season. For a decision in summer 2022 (FIFA 22), that means
starting seasons FIFA 15 to FIFA 20, whose outcomes are FIFA 17 to FIFA 22. That gives **80,730 training examples**.

We even have an automatic test for this rule (`test_no_data_leakage` in `tests/`): it trains the model once with all
the data and once after deleting every season after the decision, and checks that every prediction is identical.

### M. The model: gradient boosting (`HistGradientBoostingRegressor`)

The model is **gradient boosting**, from scikit-learn. The idea in simple words:

1. Start with a very rough guess, the same number for everybody (roughly the average future rating).
2. Build a small **decision tree**: a set of yes/no questions such as "Is he younger than 21? Is his potential more
   than 10 points above his overall?". It doesn't predict the rating itself; it predicts **how wrong the current guess
   is** for each kind of player, and corrects it a little.
3. Build another small tree that corrects the mistakes that are *still* left. Then another, and another.

Our model builds 400 such trees, and each correction is shrunk to 5% of its size (the "learning rate"), so the model
improves slowly and carefully instead of over-reacting to a few unusual players. Together, the trees learn patterns no
single rule captures, for example that potential matters a lot at 19 but almost nothing at 29.

Why this model and not a neural network? Our data is a table with tens of thousands of rows. For that kind of data,
gradient boosting is the standard tool: it's accurate, trains in a few seconds, handles missing values by itself, and
we can check what it relies on. The task itself says model complexity is not a goal.

### N. How good is it?

We tested it on seasons it never saw (notebook 03, and `examples/check_predictions.py`). Three methods compete:

- **"Nothing changes"**: the rating in two years equals today's. Ratings change slowly, so this is a strong baseline.
- **Linear regression** on overall, potential and age.
- **Our gradient boosting model.**

On the 2019 and 2020 decisions, the average error was 2.59 rating points for "nothing changes", 2.57 for linear
regression and **2.28** for our model. On the 2021 decisions it was 2.33 against **2.03**. More important for scouting
is whether it puts players in the right **order** of growth. For players aged 23 or younger, the rank correlation
between predicted and real growth was **0.42**, against **0.35** for EA's own potential rating.

It has two honest weaknesses. In some test years it was about 0.5 to 1 point too optimistic for everybody, probably
because EA's rating style changed over time. That doesn't change the *order* of the candidates. And it is **too
cautious about superstars**: in summer 2021 it expected Pedri and Bellingham to reach about 79, and both reached 85.
The model learns from what typical players do, so exceptional development is the hardest thing to predict.

For Bissouma himself the model predicts 81.1 in two years: he was 24 and already at his level, so no big change.

---

## Part 5 · From scores to a decision

### O. Ranking the candidates (`score`)

Each of the 268 remaining candidates now has a predicted rating in two years. By default, we simply **sort by that
prediction**.

The code is a little more general. It converts each signal into a **percentile** inside the pool (0 = worst in the
pool, 1 = best), so that numbers in completely different units can be combined. There are three signals: style match,
predicted rating, and price (cheaper = higher). The final score is a weighted average of the percentiles. The default
weights are 0 for style (because style is already used as the filter), 1 for the prediction and 0 for price. That's
the same as sorting by the prediction.

The **"prefer cheaper"** slider in the app moves weight from the prediction to the price. At 0.3, the score is 70% future
rating and 30% cheapness. In the backtest that saved about 28% per player but lowered the success rate from about 56%
to 41%. The club can choose that trade-off; it isn't free.

### P. Explanations and risk flags (`explain`, `risk_flags`)

A list of names without reasons is useless to a sporting director. For every recommended player the Finder writes a
one-line explanation, for example: *"79% style match · rated 78 now, predicted 78 in 2 years · costs 52% of
Y. Bissouma's value"*.

It also adds simple **risk flags**, each tied to a known weakness of football data:

- **bench player**: his club lists him as a substitute or reserve. We have no minutes-played data, so this is our proxy for "maybe not enough playing time to trust his rating".
- **weaker league**: his league's strength is more than 3 points below the selling club's league. Ratings across leagues aren't fully comparable.
- **contract ending**: at most one year left. Good for the price, but other clubs will be interested too.
- **little growth left**: aged 27 or more.

### Q. The result for our example

```
        player  age  club                 now  in 2y   value   style  risks
 N. Maksimović   26  Getafe                79   79.0  €20.5M   0.61   -
      J. Weigl   25  Benfica               79   78.7    €23M   0.63   -
        D. Sow   24  Eintracht Frankfurt   79   78.5    €24M   0.67   -
      R. Tapia   25  Celta de Vigo         79   78.4    €22M   0.60   bench player
    P. Rosario   24  Nice                  78   77.8  €18.5M   0.79   -
```

All five are two or three points below Bissouma today and cost half to two thirds of his value. Our final
recommendation was Rosario: the closest style (0.79), the cheapest, a regular starter, and he moves to #1 if the budget
is cut by 20%. We also report honestly that by January 2023 his rating had dropped 2 points, which is why we name Sow
as plan B.

This is all the app shows (`app.py`). It calls exactly the same functions, with sliders instead of typed numbers.

---

## Part 6 · How we know it works: the backtest (`src/backtest.py`)

A shortlist always looks convincing. Any method produces names. The real question is: **would these players actually
have been good buys?** To answer it, we replay history.

### R. Replaying 750 decisions

We stand at the end of three past seasons: FIFA 19, FIFA 20 and FIFA 21 (summers 2019, 2020 and 2021). In each one we
pick **250 random "sold" players**: established first-division players rated 70 to 84 and aged 22 to 30. For each of
them we pretend his club sells him and has **60% of his value** to spend, his wage as the wage cap, an age limit of 28,
and a minimum level of 12 points below him.

For every case, the growth model is retrained using only what was known at that moment (the leakage rule from
section L). Then **eight methods** each pick five replacements from **exactly the same candidate pool**, so any
difference comes from the method and nothing else:

- our Finder (style filter + growth model);
- the Finder with a minimum style match of 0.6;
- the Finder preferring cheaper players;
- the growth model without the style filter;
- our first design (style mixed into the score);
- "highest rating I can afford": what a basic scout would do;
- "highest potential I can afford": the classic "buy young talent" rule;
- and random picks, as a floor.

### S. What counts as success

Two seasons later, a pick is a **success** if he is still in the game **and** rated at most 3 points below the player
who was sold. In other words, he became a real replacement.

Players who disappeared from the game count as **failures**. This matters: about 38% of players leave the game within
two years. If we only judged the players who stayed, every method would look better than it really is. That mistake is
called **survivorship bias**.

### T. The result

Over the 750 decisions, the Finder's picks succeeded **56%** of the time. "Highest rating I can afford" managed 43%,
"highest potential" 44%, and random picks 11%. The Finder's picks also dropped out less often (13% against 20%), matched
the sold player's style much better (0.74 against 0.60), and cost the same (about €3.8M against €3.9M). It doesn't win
by spending more.

### U. Is it luck? The bootstrap

Even with 750 cases, a difference could be partly luck. The **bootstrap** checks this. We draw 750 cases at random
from our 750, *with replacement* (some cases appear twice, some not at all), and recompute the Finder's advantage. We
repeat this 2,000 times. The middle 95% of those advantages is the **95% confidence interval**. For the Finder against
"highest rating", it is **+11 to +15 points**: entirely above zero, so the advantage is not luck. We resample whole
cases rather than single picks, because the five picks of one case share the same sold player and pool, so they aren't
independent.

### V. Trying to break it

We also changed the setup on purpose. The Finder is the best method in each season separately (2019, 2020 and 2021),
with a stricter definition of success (as good as the sold player) and a looser one (up to 5 points below), and with a
**20% smaller budget** (43% against 30%).

We also report where it is weakest. In 2021 its lead was only about 2 points, because those outcomes were measured in
January 2023, mid-season, so young players had less time to grow. And under the strictest definition it ties with
"highest potential".

### W. What the backtest taught us

The backtest didn't only confirm the final design; it **created** it. Our first design mixed 40% style, 40% future and
20% price into one score. It succeeded only 36% of the time, worse than the simple scout. That's why style became a
filter instead of a weight.

---

## Part 7 · The idea we dropped, and why that matters

### X. The value model: EA's price is a formula

The classic "Moneyball" idea is to find players who are **cheap for their quality**. We built a gradient boosting model
that predicts a player's market value from his rating, potential, age, reputation, contract and role. The plan was to
call players "undervalued" if their real price was far below the predicted one.

It predicted EA's price **within about 2%**. That means EA's market value is almost a **formula** of rating,
potential, age and role. There are no hidden bargains inside EA's prices, and the "undervalued" players didn't do any
better later. We also checked the famous "cheap leagues" idea: at the same rating, players in the Japanese league cost
half of what players in La Liga cost. But they are also older and have lower potential. Once those are taken into
account, the league changes the price by only about 0.2%. That was **confounding**: a hidden third factor (age and
potential) created a pattern that looked like cause and effect.

This negative result changed our whole strategy. If the price is a formula of today's rating, potential and age, the
only way to beat it is to **know the future better than the formula does**. That's exactly what the growth model does,
and why it became the core of the system.

It also marks the biggest limitation of the project: EA's prices are not real transfer fees. Tchouaméni was worth €52M
in FIFA, but Real Madrid reportedly paid around €80M or more. The shortlist is a starting point for human scouts, not a
final answer.

---

## Part 8 · Checking that the code obeys the rules

### Y. Automatic tests (`tests/test_finder.py`)

`pytest -v tests/` runs 16 checks in about 15 seconds. They check that:

- no recommended player breaks the budget, the wage cap, the age limit or the minimum level;
- the club's own players are never recommended;
- every candidate plays a compatible role;
- the minimum style match is respected;
- a player's style is most similar to his own (cosine = 1 with himself);
- deleting all future seasons changes no prediction (the leakage test);
- two runs give exactly the same shortlist.

---

## Part 9 · Club goals and transfer plans

So far the club sold one player and wanted a like-for-like replacement. Real clubs often sell several players in one
window, with one budget, and they don't always want the same thing. The project now handles both (notebook 06).

### The three goals

- **Like-for-like** (`replace`, the default): everything described above. Style filter, then rank by predicted rating.
- **Strengthen the squad** (`squad`): no style filter. Simply the candidate with the best predicted rating in two years.
  Useful when the sold player didn't fit the team anyway.
- **Make money** (`profit`): rank by **expected profit**. This needs two new models, trained with the same no-leakage rule
  as the growth model:
  - a **regression** model that predicts the player's **market value in two years** (typical error 39%, against 50% for
    "the value stays the same");
  - a **classification** model that predicts the **chance he is still in the game** in two years (AUC 0.78). A player who
    disappears can't be sold on, so we count his resale as €0. The model is well calibrated: when it says 30%, about 30%
    really stay; when it says 95%, about 95% do. Below 50% the Finder shows a "high drop-out risk" flag.

  Expected profit = chance of staying × predicted value in two years − price. In the 750-case backtest, profit mode earned
  about **€5.5M per signing** two years later. That ties with the simple rule "buy the highest potential" (€5.3M), so it
  isn't better on money alone, but its signings are better replacements (49% vs 44%) and disappear less often.

### Several players, one budget: the transfer plan

Imagine Brighton in summer 2022: they sold both Bissouma and Cucurella and have €45M for the two replacements. The simple
approach is to split the money by hand (a share for each position) and choose each replacement separately. But money can't
move between positions that way.

The **transfer plan** (`ReplacementFinder.plan`) solves the whole problem at once with **integer programming**, using
scipy's `milp`. For every pair (sold player, candidate) there is a variable that is 1 if we buy that candidate, 0 if not.
The rules are written as constraints: exactly one signing for each sold player; nobody bought twice; total price within
the budget. The solver then finds the combination with the largest total gain (for like-for-like: predicted rating minus
the sold player's rating).

For each sold player, the optimizer considers the best candidates plus every "best for its price" candidate (nobody
cheaper has a higher gain). That guarantees the plan is never worse, on predictions, than splitting the budget by hand.
In the backtest (450 clubs each), the joint plan produced real replacements **65% vs 61%** of the time when two players
were sold, and **67% vs 60%** with three. Both differences are significant, and the advantage grows with the number of
players. Our first optimizer lost to the simple method; the backtest showed it, which is how we found and fixed the bug.

### Focus: telling the plan what matters

A club can say "this summer, improve the midfield" (or attack, or defense). In the plan, the gain of the focus positions
counts double, so the optimizer is willing to spend more there. For Brighton with €45M: without a focus it buys
Maksimović (€20.5M) and David Raum (€23.5M); with a midfield focus, Bennacer (€33.5M) and a cheap 18-year-old left-back;
with a defense focus, Renan Lodi (€36.5M) at left-back and a cheaper midfielder. The focus is the club's wish, not a
prediction, so there is nothing to test against reality, only that the money really moves (which a test checks).

---

## Z. The whole thing in one breath

FIFA ratings (10 million weekly rows) are cleaned to one end-of-season row per player and season. When a club sells a
player, business rules keep the players who can play his role and fit the money and age limits. **Unsupervised
learning** (cosine similarity of skill-profile shapes) keeps the half that plays most like him. **Supervised learning**
(gradient boosting trained only on the past) predicts each one's rating in two years, and we rank by that, with
explanations and risk flags. A **backtest on 750 historical decisions**, with no future data, the same pool for every
method, drop-outs counted as failures and a bootstrap check, shows that this beats "buy the best rating you can afford"
by about 13 percentage points. Along the way, four of our six ideas were rejected by the data, and every rejection made
the system better.

---

## Summary tables

### Which file does what

| Step | What happens | Where |
|---|---|---|
| B | Clean 10M rows to 2 snapshots per player per season | `src/prepare_data.py` |
| C | Load end-of-season rows, map positions to roles, add league strength | `load_players` in `src/finder.py` |
| D | Build the candidate pool from the club's limits | `ReplacementFinder.candidates` |
| E–F | Style profile and cosine similarity | `style_similarity` |
| I | Keep the closest half (and optional minimum match) | `keep_good_fit` |
| K | Build training examples (player now → rating in 2 years) | `growth_pairs` |
| L–M | Train gradient boosting on the past only | `train_growth_model` |
| O | Turn signals into percentiles and a weighted score | `score` |
| P | Explanation line and risk flags | `explain`, `risk_flags` |
| Q | Shortlist on screen | `app.py`, `examples/find_replacement.py` |
| R–W | Replay 750 decisions, compare methods, bootstrap | `src/backtest.py`, notebook 04 |
| X | The dropped value model | `value_residuals`, notebook 03 |
| Y | Automatic rule checks | `tests/test_finder.py` |
| 9 | Value-in-2-years and stay models | `train_future_value_model`, `train_stay_model` |
| 9 | Goals (replace, squad, profit) | `ReplacementFinder.goal_pool`, `recommend(goal=...)` |
| 9 | Transfer plan with one budget and a focus | `ReplacementFinder.plan`, `examples/plan_transfers.py`, notebook 06 |

### Our example in numbers

| Stage | Players left |
|---|---|
| All players in FIFA 22 | 23,803 |
| Can play Bissouma's role, not from Brighton, with a market value | 3,055 |
| Within €25M, wage ≤ €57K/week, age ≤ 27, rated ≥ 69 | 535 |
| Style match at least the median (0.58) | 268 |
| Shortlist: best predicted rating in two years | 5 |

### Key numbers

| What | Number |
|---|---|
| Training examples for a 2022 decision | 80,730 (seasons FIFA 15–20 → 17–22) |
| Features the growth model uses | 45 |
| Growth model error vs "nothing changes" | 2.28 vs 2.59 points (2019–20); 2.03 vs 2.33 (2021) |
| Ranking growth of players ≤ 23: our model vs EA potential | 0.42 vs 0.35 |
| Style self-recognition (top match / top 10) | 86% / 92% |
| Backtest success: Finder / highest rating / highest potential / random | 56% / 43% / 44% / 11% |
| Advantage over "highest rating", 95% interval | +11 to +15 points |
| EA price predicted by a simple model within | ≈ 2% |

### Glossary

| Term | Meaning in this project |
|---|---|
| Supervised learning | Learning from examples with known answers (player now → his rating two years later) |
| Unsupervised learning | Finding structure without answers (which players play alike) |
| Style profile | Each skill minus the player's own average skill: the shape, not the height |
| Cosine similarity | How closely two style profiles point in the same direction (−1 to 1) |
| K-Means / silhouette | A clustering method / a score for how well-separated the clusters are |
| Gradient boosting | Many small decision trees, each correcting the previous ones' mistakes |
| Data leakage | Accidentally training on information from after the decision date |
| Baseline | A simple method a model must beat ("nothing changes", "highest rating") |
| MAE | Mean absolute error: the average size of a prediction's mistake |
| Rank correlation (Spearman) | How well the predicted order matches the real order (1 = perfect) |
| Backtest | Replaying past decisions with only past data, then checking the outcome |
| Survivorship bias | Looking only at players who stayed, which makes results look too good |
| Bootstrap | Resampling the cases many times to see how much a result could vary by chance |
| Confounding | A hidden factor that creates a misleading pattern (age behind "cheap leagues") |
