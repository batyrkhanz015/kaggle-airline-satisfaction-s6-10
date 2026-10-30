# CLAUDE.md

Kaggle Playground Series S6E10, "Predicting Airline Satisfaction" (https://www.kaggle.com/competitions/playground-series-s6e10): binary classification that predicts the probability that a passenger is satisfied, scored by ROC AUC. The problem statement, how to run, progress and experiment log live in `README.md`; read it first.

## Competition facts
- **Deadline: 31 Oct 2026, 23:59 UTC (= 1 Nov, 04:59 local time, UTC+5).** Plan every step against it.
- Submission: `id,satisfaction`, one **probability** per test row (299,844 rows, ids 699635–999478, same order as `sample_submission.csv`). Never submit hard 0/1 labels, because AUC only uses the ranking.
- 10 submissions per day; the count resets at 00:00 UTC (05:00 local). Up to 2 final submissions count. The public LB uses 20% of test (~60k rows), the private LB uses the other 80%.
- Data: train 699,635 × 23, test 299,844 × 22. The target `satisfaction` is a bool (44.4% True), so cast it to int. Features: 13 service ratings 0–5 (0 = not rated; keep it as its own value), `Age`, `Flight Distance`, two delay columns, and 4 categoricals (`Gender`, `Customer Type`, `Type of Travel`, `Class`). Only `Arrival Delay in Minutes` has NaNs (~0.04%). Column names contain spaces and `/`.
- The data is synthetic, generated from the original Airline Satisfaction dataset (129,880 rows, the same 22 columns minus `id`, CC0 license); the README shows how to download it as `data/original.csv`. The rules allow external data.
- Score scale: default CatBoost on raw features gets ≈ 0.9575 (80/20 holdout); a strong single LightGBM with feature engineering gets ≈ 0.961 CV; the top public LB is ≈ 0.964 (large stacks). Real gains are 0.0001–0.001, so small differences matter.

## How we work: pair mode
- The user is learning to work as an ML developer. Claude writes the code for each step; the user reviews it and asks questions.
- Before writing code, state the plan in a few lines: which routine step, what we'll do, and why.
- After running it, explain every part in plain language: what the code does, why it is done this way, what the output shows, and what's next.
- Write code a learner can follow: small cells, clear names, comments that explain *why*, and a markdown cell above each code cell saying what it does.
- Follow the ML Project Routine doc: https://claude.ai/artifact/BW4T1z13A8gfU3VsGe6kjJ (read it with the Claude Docs tools). The deadline is close, so keep steps short: quick EDA, baseline, submit, then iterate. Track progress in the README's Progress checklist.

## Environment
- Python is the main install, `python3` (3.12 from python.org); there is no venv. Install packages with `python3 -m pip install`, then pin them in `requirements.txt`. The folder name contains spaces, so always quote paths.
- Notebooks use the Jupyter kernel **"Python 3.12 (airline)"** (`~/Library/Jupyter/kernels/airline`). It sets `DYLD_FALLBACK_LIBRARY_PATH` to scikit-learn's bundled OpenMP, which LightGBM and XGBoost need because this Mac has no Homebrew `libomp`. Python run from Bash needs the same variable: `DYLD_FALLBACK_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sklearn/.dylibs python3 ...`. CatBoost works without it.
- Git: this folder is its own repo (branch `main`), separate from the parent `MLCourse/` repo, which ignores it via its `.git/info/exclude`. It may be pushed to GitHub later (`gh` is logged in), so nothing private or from `data/` goes into commits.
- To open VS Code or another GUI app from Bash: `env -u CLAUDE_CODE_CHILD_SESSION -u CLAUDECODE open -a "Visual Studio Code" .` (a plain `open -a` turns off transcript saving in VS Code terminals).

## Hardware budget (MacBook Pro M1: 8 cores, 8 GB RAM, ~10 GB free disk)
- Measured: one CatBoost fit (1,000 trees, depth 6, 560k rows) takes ~2 min and ~0.9 GB RAM, so 5-fold CV takes ~10 min. LightGBM is faster: use it to iterate on features, then add XGBoost/CatBoost for diversity.
- Run one heavy job at a time on all cores (`n_jobs=-1` / `thread_count=-1`); never train folds in parallel.
- Store wide feature sets as `float32`/`category`, because encoded features × 1M rows fill 8 GB quickly.
- Time a single fold first, and give the user the estimate before starting anything over ~10 min.
- Out of scope on this Mac: tabular foundation models (TabPFN etc.), neural nets (RealMLP etc.) on 700k rows, and large Optuna sweeps. If one is worth it, run it in a Kaggle Notebook (free GPU) and bring its OOF/test predictions back here.

## Layout (kept minimal on purpose)
```text
README.md                    problem, how to run, progress, experiment log
requirements.txt             pinned packages
airline-satisfaction.ipynb   the whole solution, one section per routine step
data/                        train.csv, test.csv, sample_submission.csv, original.csv (gitignored)
preds/                       saved OOF + test predictions (gitignored)
submissions/                 NN-description.csv (gitignored)
```
Keep everything in the one notebook; ask the user before splitting it or adding `src/` modules.

## Conventions
- Never edit the files in `data/`.
- The notebook's first code cell holds the settings: paths (`DATA`, `PREDS`, `SUBMISSIONS`), `SEED = 42`, `TARGET = "satisfaction"`, `ID_COL = "id"`. Use them everywhere instead of hard-coding, and pass `random_state=SEED` to every model and split.
- Validation: one fixed `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)` split of the competition rows, defined once and reused by every model so their predictions can be compared and blended. Report OOF AUC plus fold mean ± std. Fold std is ~0.0006, so judge a change by its per-fold gain on the same folds. Trust CV over the public LB.
- Leakage: anything that uses the target (target encoding, target means from the original data, stacking) is fit on training folds only. Unsupervised counts/frequencies may use train + test. Original rows may join the training folds only (with an `is_original` flag); never put them in validation folds and never score on them.
- Shared code (the CV loop, feature functions) is written once as functions in the notebook and reused.
- Every CV run saves `preds/NN-name-oof.npy` and `preds/NN-name-test.npy` (rows in `train.csv` / `test.csv` order; NN = experiment number). A CV cell loads them instead of retraining when they exist, so a kernel restart doesn't cost a rerun.
- Before a commit, the notebook must run top to bottom without errors.
- Every experiment gets a row in the README experiment log; credit ideas taken from public notebooks there.
- Every submission is saved as `submissions/NN-description.csv` and checked before upload: 299,844 rows, ids match the sample, no NaN, values in [0, 1]. Submit with `kaggle competitions submit -c playground-series-s6e10 -f <file> -m "<message>"`; read scores with `kaggle competitions submissions -c playground-series-s6e10`.
- Final 2 submissions: choose by CV, not public LB: the best-CV blend plus one different, robust candidate (e.g. the best single model). Select them on Kaggle before the deadline; otherwise Kaggle picks the 2 best public scores.

## Ask the user first
- Before submitting to Kaggle (10 per day).
- Before committing or pushing.
- Before runs expected to take over ~10 min, and before installing large packages (the disk is nearly full).
