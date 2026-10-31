# Speaker scripts

The same text is in the speaker notes of `Replacement_Finder.pptx` (in PowerPoint: View → Notes, or Presenter View).
Speak in your own words. These scripts are a guide, not something to read aloud word for word.
Replace `[Name A]` … `[Name D]` with your real names (cover slide and the hand-over lines below).

| Speaker | Part | Slides | Speaking time |
|---|---|---|---|
| A | The question and the main ML idea | 1, 2, 3 | ≈ 2:20 |
| B | Data, methods and critical thinking | 4, 5, 6 | ≈ 3:38 |
| C | Validation: backtest, stress tests, the Haaland warning | 7, 8, 9, 10 | ≈ 3:25 |
| D | Demo, decisions, honesty, conclusion | 11, 12, 13, 14 | ≈ 3:46 |

Total ≈ 1845 words: about 13 minutes at a calm pace, plus about a minute of demo clicks.

Order: A (1–3) → B (4–6) → C (7–10) → D (11–14).

**The story in one line:** two ML models answer one decision (A); every method had to beat an alternative, and four of our six ideas were rejected (B); a time-split backtest on 750 decisions shows it works and where it's weak (C); here is what we'd buy, what we can't claim, and how we checked the AI (D).


---

## Speaker A: The question and the main ML idea

### Slide 1 · Cover (about 20 seconds)

Good afternoon. We are [team names], and our project is called Replacement Finder.

It combines two machine-learning models to support one real decision a football club makes every summer. And we spent as much effort trying to prove it wrong as we did building it.

### Slide 2 · The question we answer (about 1 minute)

Every summer, clubs sell players. Imagine Brighton in 2022: they have just sold Yves Bissouma, their defensive midfielder, to Tottenham.

So the question we answer is: which affordable players can do the same job, and will still be good in two years?

The person who decides is the sporting director, together with the scouts. The decision is concrete: whom to buy for this one role. The limits are real: less money than the sale brought in, a wage cap and an age limit.

And it's hard because there are around twenty thousand players in our data every season. No scouting department can watch all of them. They need to know where to look first.

### Slide 3 · The main idea: two ML models, one decision (about 1 minute 30 seconds)

Here is the whole idea on one slide.

We start from ten years of FIFA ratings. First, simple business rules: same role, within the budget, the wage cap and the age limit. That leaves a pool of often more than a thousand players.

Step three is unsupervised learning. We describe every player's style as the shape of his skill profile, and we keep the half of the pool whose shape is closest to the player we sold. Nobody labels the data; similarity does the work.

Step four is supervised learning. A gradient boosting model, trained on past seasons, predicts each candidate's rating two seasons from now, and we rank by that.

The result is a shortlist with reasons and risk flags, in an app for the sporting director.

So, in one sentence: unsupervised learning decides who fits, supervised learning decides who will be good, and a time-split backtest decides whether we should trust any of it.

[Name B] will explain the data and why we chose these methods.


---

## Speaker B: Data, methods and critical thinking

### Slide 4 · Data and features (about 1 minute 15 seconds)

Our data is the FIFA 23 complete player dataset from Kaggle: EA Sports ratings from FIFA 15 to 23. The raw file has ten million rows, because EA updates ratings almost every week, so the same player appears around sixty times per season. We kept two snapshots per player per season, which leaves about 394 thousand rows.

We built our own features, most importantly the style profile, each skill minus the player's own average, and the target: the rating two seasons later.

We also found problems. Goalkeepers have no outfield stats, so they're compared on their own skills. League names repeat across countries: "Serie A" is both Italy and Brazil, so we group by IDs. And there is selection bias: Ukraine has only its two best clubs in the game, so its league looks stronger than England's.

### Slide 5 · Methods and the alternatives we tested (about 1 minute 30 seconds)

For every part of the system we asked: is there a simpler alternative, and does our method beat it?

To decide whether two players play alike, we first tried K-Means clustering. But the silhouette score is only 0.22: styles are a continuum, not boxes. We use cosine similarity of skill shapes instead, and we validated it: given a player's profile, it finds the same player one season earlier as the top match 86 percent of the time, better than plain Euclidean distance.

To predict the future, we compared gradient boosting with three baselines: "nothing changes", linear regression, and EA's own potential rating. Boosting has the smallest error, 2.28 rating points against 2.59, and it ranks young players' growth better than EA's potential.

The third idea, finding players who are cheap for their rating, we dropped. I'll explain why on the next slide.

And why not a neural network? For tabular data of this size, gradient boosting is the standard tool, and we can explain it.

### Slide 6 · Hypotheses we tested (critical thinking) (about 1 minute 30 seconds)

This slide is the one we are most proud of. These are six things we believed, how we tested each one, and what the data said.

We believed styles form clear groups. The silhouette score said no.

We believed "cheap leagues" hide bargains: at the same rating, players in the Japanese league cost half of what they cost in Spain. But those players are older, with lower potential. Once we control for that, the league changes the price by only 0.2 percent. That was confounding, not a bargain. Correlation is not causation.

We believed players who are cheap for their rating are good deals. But a simple model predicts EA's market value within 2 percent. EA's price is a formula, so there are no hidden bargains in it. That changed our strategy: the only way to beat a formula is to predict the future better than the formula does.

And our first design, which mixed style directly into the score, did worse than the simple rule in the backtest.

Two ideas survived: our model predicts growth better than EA's potential, and style works best as a filter. The final system is simply what survived these tests.

[Name C] will now show how we tested the whole system.


---

## Speaker C: Validation: backtest, stress tests, the Haaland warning

### Slide 7 · How we avoided fooling ourselves (about 1 minute 15 seconds)

To know whether the system works, we replayed history: 750 transfer decisions from 2019 to 2021, each checked two years later. We built five safeguards into this test.

One: no peeking at the future. The growth model only learns from seasons whose outcome was already known at the decision date. Otherwise we would have data leakage.

Two: every method chooses from exactly the same pool, so any difference comes from the method itself.

Three: players who disappear count as failures. Thirty-eight percent of players leave the game within two years. If we ignored them, every method would look better than it is.

Four: the bootstrap, to check it isn't luck. And five: we deliberately tried to break the result.

### Slide 8 · Backtest results (about 1 minute)

Here's the result. Success means that two years later the player is still in the game and at most three points below the player we sold.

The Replacement Finder succeeded 56 percent of the time. A scout who buys the best rating he can afford: 43. Buying the highest potential: 44. Random picks: 11, which shows how hard the task is with only 60 percent of the money.

The advantage over the simple rule is 13 percentage points, and the 95 percent bootstrap interval is plus 11 to plus 15, so it isn't luck.

It also wins for the right reasons: fewer of its picks drop out, their style matches the sold player much better, and they cost the same. The Finder doesn't win by spending more.

### Slide 9 · Stress tests (about 1 minute)

A result that only works in one setup is fragile, so we tried to break it.

It's best in every season, with stricter or looser success, and with a 20 percent smaller budget: 43 percent against 30 for the simple rule.

But we also want to show where it is weakest. In 2021 the lead is only about two points, because those outcomes were measured in January 2023, mid-season, so young players had less time to grow. And with the strictest definition, "as good as the sold player", it's tied with "highest potential", because becoming fully as good on 60 percent of the money mostly means betting on very young talent.

So the Finder wins every row, and we tell you where the margin is thin.

### Slide 10 · The Haaland case, and a warning (about 45 seconds)

One example. Summer 2019: Chicago Fire sells a striker, and the budget is 2.3 million. One of the Finder's five picks was a 17-year-old Norwegian rated 68, for 1.8 million: Erling Haaland, rated 87 two years later. The "best rating" scout bought five 28-year-olds rated 70, and none of them worked out.

But we want to be careful. We chose this case because it looks good, which is cherry-picking. One success proves nothing. The evidence is the 750 cases you saw before; this is only an illustration of the pattern: the Finder prefers players who are still improving.

[Name D] will now show the tool and our decisions.


---

## Speaker D: Demo, decisions, honesty, conclusion

### Slide 11 · Live demo (about 1 minute 45 seconds (live demo))

[Open the app before the talk starts; the first load takes about 10 seconds.]

Let me show you the tool. We choose summer 2022, Brighton, and the player we sell, Bissouma. The budget starts at 70 percent of his value.

For each candidate we see his rating now, his predicted rating in two years, his price, his style match and risk flags.

If we cut the budget by 20 percent [move the slider], Pablo Rosario moves to the top, so this recommendation doesn't depend on the exact budget.

And here is a case where the tool needs a human. Brentford selling Christian Eriksen [switch club and player]. With default settings the style matches are low: good midfielders, but not playmakers. Eriksen is a rare type. If we raise the minimum style match to 0.6, we get real creators like Kahveci and Denis Suárez.

That's the point: the tool doesn't decide. It shows its reasons, so the expert can correct it.

### Slide 12 · Summer 2022 decisions (about 1 minute 15 seconds)

The task asks us to end with a decision, so here is what we would have bought in summer 2022.

For Brighton: Pablo Rosario from Nice, the closest style match to Bissouma, for about half his value. He is also first on the list with a 20 percent smaller budget. But honestly, by January 2023 his rating had dropped two points, so our plan B is Djibril Sow. In reality, Brighton promoted their own young Caicedo, whom our growth model also rated highly.

For Real Madrid: Palhinha or Højbjerg, for half of Casemiro's value. On the radar, Palhinha's profile almost lies on top of Casemiro's. Real paid a reported 80 million or more for Tchouaméni, a more ball-playing midfielder. And Fulham really did buy Palhinha that summer, so he was available.

For Brentford: Denis Suárez or Kahveci, once we demand a closer style match.

### Slide 13 · Limits and how we checked the AI (about 1 minute)

Two kinds of honesty.

First, limits we can't remove: ratings are EA's opinion, not match events; FIFA values aren't real fees, Tchouaméni was 52 million in FIFA but cost around 80 in reality; we have no minutes played; and our data ends in January 2023.

Second, how we used AI. An LLM helped us write code and draft interpretations, as the task allows. But every number in this talk comes from code we ran. And three times, the AI's first interpretation was wrong and the data caught it: it claimed price depends on the league, which turned out to be confounding; it estimated the value model's error at 25 percent, when it was 2; and its first scoring design was rejected by the backtest. That is exactly why we test everything.

### Slide 14 · What this project shows (about 30 seconds)

To sum up, this project shows three things.

Machine learning used for a real decision: unsupervised similarity and supervised prediction, combined into a tool a sporting director could use.

Validation we can trust: a time-split backtest on 750 decisions, a bootstrap and stress tests, with a 13-point advantage over the simple rule.

And critical thinking: four of our six ideas were rejected by the data, and every rejection made the system better.

Next steps would be real transfer fees, minutes played and planning the whole squad. Thank you, we're happy to take your questions.
