# Personal defense: likely questions and how to answer

Every team member should be able to answer all of these in their own words. Short, plain answers are best. If you
don't know something, say what you would check, and where.

## About the method

**Why did you choose gradient boosting for the growth model?**
It handles many features of different kinds (ratings, age, league, role) and finds non-linear patterns, such as "potential
matters a lot at 19 but not at 29", without us writing those rules. We compared it with two simpler methods: "no change"
(error 2.59 points) and linear regression (2.57). Boosting had 2.28. For ranking, linear regression was almost as good, so
the gain is mainly in the size of the prediction.

**Why cosine similarity and not clustering for style?**
Clustering gave weak groups (silhouette 0.22 at best, falling with more groups): styles are a continuum. Two players at
opposite ends of one cluster can play quite differently. Cosine similarity gives a continuous score for every pair. We
subtract each player's own average skill first, so we compare the *shape* (what he's relatively good at), not the level.

**What is the silhouette score?**
For each player: how close he is to his own group compared with the nearest other group. From −1 to 1; near 0 means the
groups overlap.

**How do you know the similarity works?**
Self-recognition test (notebook 02): take a player's FIFA 22 profile and search all FIFA 21 midfielders. The most similar one
is the same player 86% of the time (top 10: 92%), better than plain Euclidean distance (84% / 88%).

**Why is style a filter and not part of the score?**
We tried mixing it in (40% style, 40% future, 20% price): only 36% success in the backtest. Style as a filter (keep the
better half) and the growth model for ranking: 56%. The backtest showed us our first idea was wrong.

**What does "success" mean in the backtest?**
Two seasons later, the pick is still in the game and rated at most 3 points below the sold player. Players who disappeared
count as failures. We also checked 0 and 5 points; the Finder stays ahead of "highest overall" in all cases.

## About quality and trust

**How did you avoid data leakage?**
Time-based splits. For a decision at the end of season *v*, the growth model only learns from pairs (*v'*, *v'*+2) where
*v'*+2 ≤ *v*, so it never sees outcomes after the decision date. Example: for summer 2022 decisions, it learns from pairs up
to FIFA 20 → 22. In the value model, each player is priced by a model that didn't see his own price (out-of-fold).

**How do you know the result isn't luck?**
750 cases, and a bootstrap: resample the cases thousands of times and recompute the difference. The 95% interval of the
Finder's advantage over "highest overall" is +10.9 to +15.4 points, entirely above zero. We resample whole cases because
the 5 picks of one case aren't independent.

**What's the baseline, and why that one?**
"Highest overall rating I can afford", which is what a basic scout does. Also "highest potential" (the young-talent rule)
and random picks as a floor. The task asks to compare with a simple approach.

**Which limitation affects reliability the most?**
That ratings and values come from EA, not from matches or real transfer fees. We showed EA's value is almost a formula
(2% error), so we can't measure real-market bargains, and real fees are higher (Tchouaméni: FIFA €52M, reported real fee
~€80M+).

**What was the most unexpected result?**
Pick one: (a) EA's market value is almost a formula, so there are no bargains in EA prices; (b) the "cheap league" effect
disappears once age is controlled (confounding); (c) our first design was worse than the simple rules; (d) the Haaland case.

## "What if…" questions

**What changes if the budget is cut by 20%?**
Backtest: every method gets worse (the Finder goes from 56% to 43% success), but the Finder stays best: +13 points over
"highest overall" and +6 over "highest potential". Brighton case: with €20M, Rosario moves from #5 to #1 and is the only name
on both lists, so the recommendation is robust.
*In the app:* move the budget slider and show it live.

**What would you do with more data?**
(1) Real transfer fees (Transfermarkt), to measure real-market value. (2) Minutes played and match events (e.g. StatsBomb), to
replace "bench player" as a proxy and check ratings against performance. (3) FIFA 24+, to check the 2022 predictions properly.

**Change a parameter of the model.**
In the app: "Minimum style match" (try the Eriksen case at 0 and at 0.6), "Prefer cheaper", "Max age". In code:
`DEFAULT_FIT_QUANTILE` in `src/finder.py`, or `budget_factor` in `run_backtest`.

**Why did you recommend Rosario if he dropped 2 points by January 2023?**
We decided with information from summer 2022, when he was the best style match, the cheapest and robust to budget cuts. The
drop is an early warning after half a season, not a verdict. That's why we named Sow as plan B. Every prediction carries
uncertainty, and we show it instead of hiding it.

**Would Real Madrid have been better off with Palhinha?**
For "replace Casemiro now, for half the money": probably yes, since he's rated 82 and is the closest style match, and Fulham bought
him that summer. But Real bought a 21-year-old with potential 88 for the long term and for resale. Our tool answers a
different question.

## Questions about your own part

Each person: be ready to say what you studied, re-ran or changed, and one result you checked yourself. Suggested split:

| Person | Owns | Should be able to explain in detail |
|---|---|---|
| A | Data + app | prepare_data.py (why 10M → 393k rows), notebook 01, the Streamlit demo |
| B | Style | notebook 02: style profile, K-Means, silhouette, PCA, cosine, self-recognition test |
| C | Models | notebook 03: time split, baselines, MAE, Spearman, permutation importance, the value-model negative result |
| D | Backtest + decisions | notebooks 04–05: design, success definition, bootstrap, robustness, 2022 recommendations |

A good exercise before the defense: each person changes one thing in their notebook (e.g. *k* in K-Means, the success
threshold, the age limit), reruns it, and explains what changed and why.
