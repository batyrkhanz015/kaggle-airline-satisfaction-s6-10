# Predicting Airline Satisfaction

Kaggle Playground Series, Season 6 Episode 10: https://www.kaggle.com/competitions/playground-series-s6e10

## Problem statement

- **Goal:** Predict the probability that each of the 299,844 passengers in `test.csv` is satisfied with their flight.
- **Type:** Supervised binary classification on synthetic tabular data, batch predictions.
- **Metric:** ROC AUC on predicted probabilities, measured locally with stratified 5-fold CV. The public leaderboard (20% of the test set) is a secondary check.
- **Baselines:** a constant prediction scores 0.5 (this is what `sample_submission.csv` does); default CatBoost on the raw features scores ≈ 0.9575 (quick 80/20 holdout check).
- **Target:** ≥ 0.960 CV AUC.
- **Constraints:** deadline 31 Oct 2026, 23:59 UTC; 10 submissions per day and 2 final picks; trained on a MacBook M1 with 8 GB RAM (gradient-boosted trees, one heavy job at a time).
- **Assumptions to check:** `test.csv` follows the same distribution as `train.csv`; the original 129k-row dataset is close enough to help as extra training data or features.

## How to run

```bash
pip install -r requirements.txt
```

Get the data (join the competition on Kaggle first). The competition data can't be redistributed, so it isn't in this repo:

```bash
kaggle competitions download -c playground-series-s6e10 -p data
unzip data/playground-series-s6e10.zip -d data

# Optional: the original dataset the competition data was generated from (CC0, allowed by the rules)
kaggle datasets download arseniyshutko/binary-aviation-satisfaction-129k -p data --unzip
mv data/data.csv data/original.csv
```

On macOS, LightGBM and XGBoost need OpenMP: `brew install libomp`.

Then open `airline-satisfaction.ipynb` and run all cells.

## Project structure

```text
├── README.md                    this file
├── requirements.txt             pinned dependencies
├── airline-satisfaction.ipynb   the whole solution, one section per step
├── data/                        Kaggle files (gitignored)
├── preds/                       saved out-of-fold and test predictions (gitignored)
└── submissions/                 every CSV sent to Kaggle (gitignored)
```

## Progress

Following the 12-step ML Project Routine.

- [x] 1. Frame the problem (problem statement above)
- [x] 2. Set up the project (own git repo, `requirements.txt`, Jupyter kernel)
- [x] 3. Get the data (`data/`, verified: train 699,635 rows, test 299,844 rows, original 129,880 rows)
- [x] 4. Split and choose validation (adversarial AUC train vs test 0.50, so stratified 5-fold CV; notebook section 3)
- [x] 5. Explore the data (notebook sections 4–5)
- [x] 6. Baseline end to end (CV baseline: run 01; first submissions on 30 Oct: 01 CatBoost, 02 blend)
- [x] 7. Features and pipelines (notebook sections 6–8: tree features, chosen set = run 08; section 9: preprocessing for linear models; section 15: rating-pattern features, not kept)
- [x] 8. Iterate on models (notebook sections 10–11: 12 algorithms on the same folds, blend of the 4 boosting models; section 13: the original rows as training data, not kept)
- [ ] 9. Tune, finalize, submit (section 14: tuned LightGBM, XGBoost and CatBoost, new stack; section 16: seed averaging, not kept. Submitted 01: LB 0.96035, 02: LB 0.96051, 03 stack: LB 0.96069, 04 tuned LightGBM: LB 0.96063. To do: select 03 + 04 as the 2 finals on Kaggle before 31 Oct 23:59 UTC)
- [x] 10. Communicate results (Results section below)
- [ ] 11. Deploy (not needed for Kaggle)
- [ ] 12. Wrap up

## Experiment log

All rows use the same stratified 5-fold split (seed 42). OOF AUC is computed over all training rows at once. Runs 09–19 were developed by five parallel agents (linear, simple, bagging, boosting, CatBoost) on a 50k-row sample, then trained on the full data one process at a time with a 3.5 GB RAM watchdog. Every run stayed under 2.2 GB, and the whole queue took ~50 min. Runs 01–08 are the feature check in notebook section 8: LightGBM with fixed settings (learning rate 0.1, 63 leaves, early stopping on the validation fold). Each set adds one group to the previous one; run 08 is run 07 without the domain features. Runs 22–24 are notebook section 14: settings searched on fold 0 at full size (Optuna TPE with 30 trials, or a 12-fit grid; trial 0 = the previous setting), then the best one retrained on all 5 folds, with folds 1–4 as the honest test. Predictions are saved as `preds/NN-name-{oof,test}.npy`, and the searches as `preds/tuning-*.optuna.log`.

| # | Date | What changed | Model | OOF AUC | Fold AUC (mean ± std) | LB score | Keep? |
|---|------|--------------|-------|---------|-----------------------|----------|-------|
| 01 | 2026-10-29 | Raw features (21) | LightGBM | 0.95862 | 0.95863 ± 0.00067 | | baseline |
| 02 | 2026-10-29 | + domain features: rating summaries, delay sums, segments (35) | LightGBM | 0.95844 | 0.95845 ± 0.00061 | | no (−0.00019, 0/5 folds) |
| 03 | 2026-10-29 | + value counts of distance, age, delays and pairs, over train + test (42) | LightGBM | 0.95896 | 0.95897 ± 0.00062 | | yes (5/5 folds) |
| 04 | 2026-10-29 | + distance digits (45) | LightGBM | 0.95925 | 0.95926 ± 0.00059 | | yes (5/5 folds) |
| 05 | 2026-10-29 | + original-data lookups: satisfaction rate and count per key (59) | LightGBM | 0.95977 | 0.95978 ± 0.00045 | | yes (5/5 folds) |
| 06 | 2026-10-29 | + original-only LightGBM probability (60) | LightGBM | 0.96005 | 0.96006 ± 0.00055 | | yes (5/5 folds) |
| 07 | 2026-10-29 | + in-fold target encoding of distance, age and pairs (60 + 7) | LightGBM | 0.96037 | 0.96038 ± 0.00055 | | yes (5/5 folds) |
| 08 | 2026-10-29 | Run 07 without the domain features (46 + 7) | LightGBM | 0.96041 | 0.96042 ± 0.00062 | | **yes: chosen feature set** |
| 09 | 2026-10-30 | Section-9 linear data + in-fold TE (logit, scaled); C = 0.1 | Logistic regression | 0.95728 | 0.95728 ± 0.00057 | | blend candidate (different errors) |
| 10 | 2026-10-30 | Same data; hinge loss, alpha = 1e-4, averaged | Linear SVM (SGDClassifier) | 0.95588 | 0.95589 ± 0.00062 | | no |
| 11 | 2026-10-30 | Run-08 features + TE; min 500 rows per leaf | Decision tree | 0.95713 | 0.95712 ± 0.00044 | | no |
| 12 | 2026-10-30 | Linear data → SVD 64 dims; 200k-row sample per fold; k = 179 | KNN | 0.95167 | 0.95168 ± 0.00059 | | no |
| 13 | 2026-10-30 | Short feature list, 40 bins | Naive Bayes (categorical) | 0.94578 | 0.94578 ± 0.00035 | | no |
| 14 | 2026-10-30 | Run-08 features + TE; 300 trees, entropy, leaf 10, 50% bootstrap | Random forest | 0.95959 | 0.95959 ± 0.00041 | | blend candidate |
| 15 | 2026-10-30 | Run-08 features + TE; 300 trees, leaf 20, max_features 0.5 | Extra trees | 0.95980 | 0.95981 ± 0.00035 | | blend candidate |
| 16 | 2026-10-30 | Run-08 features + TE; lr 0.03, 63 leaves, min_child 50, early stop 300 | LightGBM (tuned) | 0.96075 | 0.96075 ± 0.00058 | | **yes** (+0.00033 vs 08, 5/5 folds) |
| 17 | 2026-10-30 | Run-08 features + TE; hist, lr 0.05, depth 7, native categoricals | XGBoost | 0.96075 | 0.96077 ± 0.00058 | | **yes** (+0.00035, 5/5 folds) |
| 18 | 2026-10-30 | Run-08 features + TE; lr 0.05, 63 leaves, native categoricals | HistGradientBoosting | 0.96068 | 0.96069 ± 0.00055 | | **yes** (+0.00026, 5/5 folds) |
| 19 | 2026-10-30 | Run-08 features + TE; depth 6, lr 0.08, one-hot cats ≤ 4 levels | CatBoost | 0.96076 | 0.96077 ± 0.00062 | 0.96035 | **yes: best single model** (+0.00035, 5/5 folds); submission 01 |
| 20 | 2026-10-30 | Simple average of runs 16–19 (stacking checked with nested CV: equal weights, no gain from more models) | Blend | 0.96095 | 0.96096 ± 0.00058 | 0.96051 | **yes: best overall** (+0.00019 vs CatBoost, 5/5 folds); submission 02 |
| 21 | 2026-10-30 | Run 16 + the 130k original rows in every training fold (`is_original` flag; their lookups and original-model probability computed out-of-fold within the original; TE fitted on synthetic rows only) | LightGBM (tuned) | 0.96081 | 0.96083 ± 0.00057 | | no: +0.00007 vs run 16, better in only 3/5 folds (rule: ≥ 4/5) |
| 22 | 2026-10-30 | Run 16 tuned with Optuna (30 trials on fold 0): 89 leaves, min_child 13, L2 3.4, subsample 0.98, **colsample 0.31** | LightGBM (Optuna) | 0.96104 | 0.96104 ± 0.00053 | 0.96063 | **yes** (+0.00029 vs 16, 5/5 folds; +0.00026 on the untouched folds 1–4); submission 04 |
| 23 | 2026-10-30 | Run 17 tuned with Optuna (30 trials on fold 0): depth 9, min_child_weight 0.63, L2 16, subsample 0.86, colsample 0.47 | XGBoost (Optuna) | 0.96082 | 0.96083 ± 0.00058 | | yes by the rule, but small (+0.00006 vs 17, 4/5 folds) |
| 24 | 2026-10-30 | Run 19, grid of depth 6–9 × L2 {1, 3, 10} (12 fits on fold 0): depth 7, L2 3 | CatBoost (grid) | 0.96077 | 0.96078 ± 0.00060 | | yes by the rule, a tie in practice (+0.00001 vs 19, 4/5 folds; the fold-2 "win" is +0.000002) |
| 25 | 2026-10-31 | Stack of runs 22, 23, 18, 24 (logistic regression on log-odds, nested CV; weights ≈ 1.4 / 0.8 / 0.3 / 0.5) | Blend | 0.96110 | 0.96112 ± 0.00054 | 0.96069 | **yes: best overall** (+0.00015 vs run 20, 5/5 folds; +0.00003 vs the simple average, 4/5); submission 03 |
| 26 | 2026-10-31 | Run 22 + the 130k original rows (run 21's setup); rule set before the run: ≥ 4/5 folds and mean gain ≥ +0.00005 | LightGBM (Optuna) | 0.96107 | 0.96108 ± 0.00052 | | no: +0.00003 vs run 22, 4/5 folds, below the +0.00005 minimum (with run 16's settings it was +0.00007, so tuning made the rows *less* useful) |
| 27 | 2026-10-31 | Run 22 + counts of rating patterns over train + test (keys: all 13 ratings, the 6 strongest, 13 + class + travel type) | LightGBM (Optuna) | 0.96095 | 0.96095 ± 0.00057 | | no: −0.00009 vs run 22, 0/5 folds |
| 28 | 2026-10-31 | Run 22 + the original's satisfaction rate and count per rating pattern (same 3 keys) | LightGBM (Optuna) | 0.96094 | 0.96095 ± 0.00052 | | no: −0.00010 vs run 22, 0/5 folds |
| 29 | 2026-10-31 | Run 22 + in-fold target encoding of the 3 rating patterns | LightGBM (Optuna) | 0.96094 | 0.96096 ± 0.00051 | | no: −0.00009 vs run 22, 0/5 folds; nothing kept, so the other models weren't retrained |
| 30 | 2026-10-31 | Run 23 + all rating-pattern features (counts, original lookups, target encoding): does a feature that hurts LightGBM hurt the others? | XGBoost (Optuna) | 0.96078 | 0.96079 ± 0.00054 | | no: −0.00005 vs run 23, 2/5 folds |
| 31 | 2026-10-31 | Run 18 + all rating-pattern features | HistGradientBoosting | 0.96065 | 0.96065 ± 0.00054 | | no: −0.00003 vs run 18, 2/5 folds |
| 32 | 2026-10-31 | Run 24 + all rating-pattern features | CatBoost (grid) | 0.96069 | 0.96070 ± 0.00055 | | no: −0.00008 vs run 24, 1/5 folds. For information: a stack of runs 22, 30, 31, 32 scores 0.96115 (+0.000047 vs run 25, 4/5 folds), just below the minimum, so not used |
| 33 | 2026-10-31 | Run 22 averaged over seeds 42, 43, 44 (same folds, features and settings) | LightGBM (Optuna, 3 seeds) | 0.96106 | 0.96107 ± 0.00054 | | no: +0.000025 vs run 22, 5/5 folds, below the minimum. Seeds 43 and 44 alone scored 0.96094, so seed 42 was a lucky seed |
| 34 | 2026-10-31 | Run 23 averaged over seeds 42, 43, 44 | XGBoost (Optuna, 3 seeds) | 0.96096 | 0.96097 ± 0.00057 | | yes by the rule (+0.00014 vs run 23, 5/5 folds) |
| 35 | 2026-10-31 | Run 24 averaged over seeds 42, 43, 44 | CatBoost (grid, 3 seeds) | 0.96087 | 0.96087 ± 0.00062 | | yes by the rule (+0.000095 vs run 24, 5/5 folds) |
| 36 | 2026-10-31 | Stack of runs 33, 34, 18, 35 (nested CV, as run 25) | Blend | 0.96109 | 0.96110 ± 0.00055 | | no: −0.000014 vs run 25, 1/5 folds; rank correlation 0.999 with run 25. The stack had already averaged out most of the seed noise |

Credits: value counts, distance digits, original-data features and pair target encoding were inspired by public notebooks, mainly [S6E10 LightGBM CV 5 folds 0.96111](https://www.kaggle.com/code/evgendvorkin/s6e10-lightgbm-cv-5-folds-0-96111).

## Results

### Final submissions

Chosen by cross-validation, not by the public leaderboard: the best blend, plus a different, robust single model.

| File | Model | CV AUC | Public LB | Private LB |
|---|---|---|---|---|
| `03-blend-tuned-stack.csv` | Stack (logistic regression on log-odds) of tuned LightGBM, XGBoost, CatBoost and HistGradientBoosting (run 25) | **0.96110** | **0.96069** | after the deadline |
| `04-lightgbm-optuna.csv` | LightGBM tuned with Optuna (run 22) | 0.96104 | 0.96063 | after the deadline |

On the public leaderboard on 30 Oct, 0.96069 ranked about 1,410 of 3,603 teams (top 39%). The top 10% started at 0.96212. The private leaderboard (the other 80% of the test set) is revealed after the deadline.

**The CV could be trusted.** All four submissions ranked on the public leaderboard exactly as CV predicted (03 > 04 > 02 > 01). The public scores are ~0.0004 below CV because this particular 20% of the test set is slightly harder for every team: on random 60k-row subsets of our out-of-fold predictions, the AUC moves by ±0.0008.

### How the score was built

| Step | Run | CV AUC | Gain |
|---|---|---|---|
| LightGBM on the raw features | 01 | 0.95862 | |
| + feature engineering: value counts, distance digits, original-data lookups, original-only model, in-fold target encoding | 08 | 0.96041 | +0.00179 |
| + hand-tuned boosting (LightGBM learning rate 0.1 → 0.03) | 16 | 0.96075 | +0.00034 |
| + average of 4 boosting models | 20 | 0.96095 | +0.00020 |
| + Optuna tuning, stack instead of average | 25 | **0.96110** | +0.00015 |

Feature engineering gave about 70% of the total gain.

### What worked

- **Features built from traces the synthetic-data generator left behind and from the original data:** counts of how often a value occurs (+0.00034), the digits of the flight distance (+0.00029), satisfaction rates looked up in the original 130k rows (+0.00052), a model trained on the original data alone (+0.00028), and in-fold target encoding (+0.00032).
- **Gradient boosting.** Of 12 algorithms on the same folds, all 4 boosting models beat every other kind of model. Bagging was 0.001 behind, and linear models, KNN and naive Bayes 0.003–0.015 behind.
- **Tuning LightGBM with Optuna** (+0.00029, 5/5 folds). The big lever was column sampling: each tree sees 31% of the columns instead of 80%, with a stronger L2 penalty.
- **Blending:** the stack beat the best single model by +0.00006 to +0.0002.

### What didn't work

Every idea was tested with a rule set before the run: keep it only if it improves at least 4 of 5 folds (from run 26 on, also by at least +0.00005 on average).

| Idea | Result |
|---|---|
| Domain features (rating averages, delay sums, segments) | −0.00019, 0/5 folds |
| The 130k original rows as extra training data | +0.00007 (3/5); +0.00003 after tuning |
| Rating-pattern features (two-thirds of passengers share an exact 13-rating pattern) | about −0.00009 for LightGBM, worse for every other model too |
| Averaging 3 seeds per model | each model +0.00003 to +0.00014, but the stack unchanged |

### Lessons

1. **Trust CV over the public leaderboard,** and judge every change fold by fold on the same folds.
2. **Set the decision rule before the run.** A tuned CatBoost passed "≥ 4/5 folds" with a fold "win" of +0.000002, which is why a minimum gain was added afterwards.
3. **Beware the winner's curse.** Settings tuned on fold 0 with seed 42 looked better partly by luck. With seeds 43 and 44, the same LightGBM scored 0.0001 lower. Untouched folds and other seeds give the honest picture.
4. **Unusual isn't the same as predictive.** The repeated rating patterns are a real trace of the generator, but they carry no information about satisfaction beyond the ratings themselves.
5. **A blend needs different errors.** Better copies of the same kind of model (seed averaging) didn't improve the stack. The gap to the top of the leaderboard comes from different model families (neural networks, tabular foundation models) and large stacks of them, which were out of scope for this laptop.
6. **8 GB is enough with discipline.** One model at a time on all cores, compact data types, and a RAM watchdog kept every run under 2.2 GB.

### What I'd try next

- A neural network (RealMLP) or a tabular foundation model (TabPFN, TabICL) on a Kaggle GPU, added to the stack.
- CatBoost's column sampling (`rsm`), the counterpart of LightGBM's biggest win, which was never searched.

## Data and credits

- Competition data: Kaggle, CC BY 4.0.
- Original dataset: [Binary Aviation Satisfaction 129k](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k), CC0.
- Ideas taken from public Kaggle notebooks are credited in the experiment log.
