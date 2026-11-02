# Predicting Airline Satisfaction

Kaggle Playground Series, Season 6 Episode 10: https://www.kaggle.com/competitions/playground-series-s6e10

Predict the probability that a passenger is satisfied with their flight, scored by ROC AUC. **Final result: private LB 0.96138** (CV 0.96110), with a stack of four gradient-boosting models.

## Problem

- **Goal:** a satisfaction probability for each of the 299,844 passengers in `test.csv`.
- **Data:** 699,635 labelled training rows of synthetic tabular data, generated from the real 129,880-row Airline Satisfaction dataset. Features: 13 service ratings from 0 to 5 (0 = not rated), age, flight distance, departure and arrival delays, and 4 categoricals (gender, customer type, type of travel, class). The target is 44.4% satisfied.
- **Metric:** ROC AUC, measured locally with stratified 5-fold CV. The public leaderboard uses 20% of the test set, the private leaderboard the other 80%.
- **Hardware:** a MacBook M1 with 8 GB RAM, one model at a time on all cores.

## Solution

**Validation.** A model can't tell train rows from test rows (adversarial AUC 0.50), so stratified 5-fold CV mirrors the test set. The same 5 folds are used by every model, so models can be compared fold by fold and stacked. CV decided every choice; the public leaderboard was only a check.

**Features:** the 21 raw columns plus five groups, 46 + 7 columns in total. Each group improved CV AUC in all 5 folds; together they lift a LightGBM from 0.95862 to 0.96041.

| Group | Columns | Idea |
|---|---|---|
| Value counts | 7 | how often a distance, age, delay or pair occurs in train + test (a trace of the data generator) |
| Distance digits | 3 | the last three digits of the distance (another generator trace) |
| Original lookups | 14 | satisfaction rate and count per distance, age or segment in the original dataset |
| Original model | 1 | the probability from a LightGBM trained on the original dataset alone |
| Target encoding | 7 | satisfaction rate per distance, age and pairs, fitted inside each CV fold to avoid leakage |

**Models:** four gradient-boosting libraries on the same features and folds. They grow trees differently, so their errors differ.

| Model | Settings | CV AUC |
|---|---|---|
| LightGBM | Optuna (30 trials): 89 leaves, 31% of columns per tree, L2 3.4, learning rate 0.03 | 0.96104 |
| XGBoost | Optuna (30 trials): depth 9, 47% of columns per tree, L2 16, learning rate 0.05 | 0.96082 |
| HistGradientBoosting | 63 leaves, 60% of features per split, learning rate 0.05 | 0.96068 |
| CatBoost | grid of 12: depth 7, L2 3, learning rate 0.08 | 0.96077 |

**Stack:** a logistic regression on the four models' out-of-fold log-odds, validated with nested CV. Weights: LightGBM 1.41, XGBoost 0.77, CatBoost 0.54, HistGradientBoosting 0.29. **CV 0.96110**, better than LightGBM alone in all 5 folds.

Tested and not kept: rating averages and passenger segments, the original rows as extra training data, rating-pattern features, and seed averaging.

## Results

| File | Model | CV AUC | Public LB | Private LB |
|---|---|---|---|---|
| `03-blend-tuned-stack.csv` | Stack of the 4 models | **0.96110** | **0.96069** | **0.96138** |
| `04-lightgbm-optuna.csv` | LightGBM alone | 0.96104 | 0.96063 | 0.96132 |

The two final submissions were chosen by CV: the best blend, plus a simpler single model. Both leaderboards ranked them exactly as CV predicted. On the final public leaderboard, 0.96069 placed 1,480th of 3,748 teams.

## How to run

```bash
pip install -r requirements.txt
```

Get the data (join the competition on Kaggle first). The competition data can't be redistributed, so it isn't in this repo:

```bash
kaggle competitions download -c playground-series-s6e10 -p data
unzip data/playground-series-s6e10.zip -d data

# The original dataset the competition data was generated from (CC0, allowed by the rules)
kaggle datasets download arseniyshutko/binary-aviation-satisfaction-129k -p data --unzip
mv data/data.csv data/original.csv
```

On macOS, LightGBM and XGBoost need OpenMP: `brew install libomp`.

Then open `airline-satisfaction.ipynb` and run all cells. From scratch it trains the 4 models in about 10 minutes on 8 cores; after that, the saved predictions in `preds/` make a re-run take under a minute.

## Project structure

```text
├── README.md                    this file
├── requirements.txt             pinned dependencies
├── airline-satisfaction.ipynb   the final solution: data, features, models, stack, submissions
├── data/                        Kaggle files (gitignored)
├── preds/                       saved out-of-fold and test predictions (gitignored)
└── submissions/                 the CSVs sent to Kaggle (gitignored)
```

## Data and credits

- Competition data: Kaggle, CC BY 4.0.
- Original dataset: [Binary Aviation Satisfaction 129k](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k), CC0.
- The value counts, distance digits, original-data features and pair target encoding were inspired by public notebooks, mainly [S6E10 LightGBM CV 5 folds 0.96111](https://www.kaggle.com/code/evgendvorkin/s6e10-lightgbm-cv-5-folds-0-96111).
