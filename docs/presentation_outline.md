# Presentation plan (13–15 minutes)

The deck is `Replacement_Finder.pptx`; the full scripts are in `speaker_scripts.md`.
It follows the five required stages of the task, and puts the ML idea and our critical thinking in the centre.

| # | Slide | Speaker | Time | What the audience should remember |
|---|---|---|---|---|
| 1 | Cover | A | 20 seconds | Who we are and what the project does |
| 2 | The question we answer | A | 1 minute | One concrete decision: replace a sold player cheaply, same job, still good in 2 years |
| 3 | The main idea: two ML models, one decision | A | 1 minute 30 seconds | Unsupervised ML decides who fits, supervised ML decides who will be good, a backtest decides whether to trust it |
| 4 | Data and features | B | 1 minute 15 seconds | 10M rows → 394k; features we built; data problems we fixed |
| 5 | Methods and the alternatives we tested | B | 1 minute 30 seconds | Each method beat a simpler alternative; the value idea was dropped |
| 6 | Hypotheses we tested (critical thinking) | B | 1 minute 30 seconds | Four of our six ideas were rejected by the data, and each rejection changed the design |
| 7 | How we avoided fooling ourselves | C | 1 minute 15 seconds | No leakage, same pool, drop-outs as failures, bootstrap, stress tests |
| 8 | Backtest results | C | 1 minute | 56% vs 43% / 44% / 11%; +13 points, 95% interval +11 to +15 |
| 9 | Stress tests | C | 1 minute | Best in every row; we show where the margin is thin |
| 10 | The Haaland case, and a warning | C | 45 seconds | Haaland as a showcase, with an explicit cherry-picking warning |
| 11 | Live demo | D | 1 minute 45 seconds (live demo) | The tool works, and needs a human for rare player types |
| 12 | Summer 2022 decisions | D | 1 minute 15 seconds | Concrete 2022 recommendations with a reality check |
| 13 | Limits and how we checked the AI | D | 1 minute | Limits we can't remove; three AI claims the data proved wrong |
| 14 | What this project shows | D | 30 seconds | ML for a decision, validation we trust, critical thinking |

## Three sentences everyone must be able to say

1. "Unsupervised similarity finds players who do the same job; a gradient boosting model predicts who will still be good in two years."
2. "We replayed 750 transfer decisions without using future data. The Finder beats 'buy the best rating you can afford' by 13 points, and the interval is +11 to +15."
3. "Four of our six ideas were wrong. The data told us, and the final system is what survived."
