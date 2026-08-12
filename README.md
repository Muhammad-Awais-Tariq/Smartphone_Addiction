# Smartphone Addiction Prediction

An end-to-end machine learning project predicting the probability that a person is addicted to their smartphone, built with a scikit-learn preprocessing pipeline, compared across seven different models, and deployed as an interactive Streamlit web app. Built for Kaggle's Playground Series - Season 6, Episode 8.

## Live Demo

Try it here: [https://smartphoneaddiction7.streamlit.app/](https://smartphoneaddiction7.streamlit.app/)

Enter a person's details (screen time, social media/gaming hours, sleep, notifications, stress level, etc.) and get a live prediction of their addiction probability.

---

## Project Structure

```
Smartphone_Addiction/
│
├── Data/                              # Raw datasets (gitignored — not pushed to GitHub)
│   ├── train.csv                      # Original Kaggle training data
│   └── test.csv                       # Original Kaggle test data
│
├── Exploration/
│   ├── Reports/
│   │   └── reports.html               # Exploratory data profiling report (sweetviz)
│   └── exploration.ipynb              # Notebook: EDA, feature engineering iteration,
│                                       # model comparison, hyperparameter tuning
│
├── Model/
│   └── smartphone_addiction.joblib    # Serialized final trained pipeline (model +
│                                       # preprocessing, all in one)
│
├── Predictions/
│   └── predictions.csv                # Kaggle submission — final model trained on full data
│
├── final_pipeline.py                  # Clean, final training script — builds the
│                                       # preprocessing pipeline, trains the final
│                                       # model, and saves it with joblib
├── app.py                             # Streamlit app — loads the saved model and
│                                       # serves the interactive prediction UI
│
├── pyproject.toml                     # Project dependencies (for uv)
├── uv.lock                            # Locked dependency versions
├── .python-version                    # Python version pin
├── .devcontainer/                     # Dev container config
├── .gitignore                         # Excludes Data/ and .venv/ from version control
└── README.md                          # This file
```

> **Note on data:** The `Data/` folder (raw CSVs) is excluded via `.gitignore` and is **not** pushed to GitHub. To run this project locally, download `train.csv` and `test.csv` from the [Kaggle Predicting Smartphone Addiction competition](https://www.kaggle.com/competitions/playground-series-s6e8) and place them in a local `Data/` folder.

---

## What Each File Does

### `Exploration/exploration.ipynb`
The working notebook where all the experimentation happened:
- Exploratory data analysis on the raw dataset (with a sweetviz profiling report)
- Missing-value handling (median imputation for most numeric columns, zero-imputation for usage columns where a missing value plausibly means "no usage", most-frequent imputation + one-hot encoding for categoricals)
- Iterative feature engineering (total screen time, social media/gaming share of screen time, average session length, notifications per app open, weekend-vs-weekday ratio, screen-to-sleep ratio) to check whether derived ratios helped over the raw columns
- Building the sklearn `Pipeline` + `ColumnTransformer`
- Training and cross-validating seven different models, including a soft-voting ensemble
- Hyperparameter tuning with `RandomizedSearchCV` / `GridSearchCV` for each model
- Fitting the final tuned model on the full training set and generating the submission

### `final_pipeline.py`
The clean, production version of the pipeline:
1. Loads `train.csv`
2. Builds the `ColumnTransformer`: median imputation + scaling for most numeric columns, zero imputation + scaling for usage columns, most-frequent imputation + one-hot encoding for categoricals
3. Fits the final chosen model (tuned XGBoost) on the full training set
4. Serializes the trained pipeline with `joblib` for deployment

### `app.py`
The Streamlit web app. Loads `smartphone_addiction.joblib`, presents a form for entering a person's details, and returns a live addiction-probability prediction.

---

## Feature Engineering

The raw features are: `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`, `gender`, `stress_level`, `academic_work_impact`.

Several derived ratio features were engineered and tested against the raw columns:

| Feature | Description |
|---|---|
| `total_time_spend` | `daily_screen_time_hours` + `weekend_screen_time` |
| `social_media_share` | `social_media_hours` / `daily_screen_time_hours` |
| `gaming_share` | `gaming_hours` / `daily_screen_time_hours` |
| `avg_session_length` | `daily_screen_time_hours` / `app_opens_per_day` |
| `notifications_per_open` | `notifications_per_day` / `app_opens_per_day` |
| `weekend_vs_daily_ratio` | `weekend_screen_time` / `daily_screen_time_hours` |
| `screen_to_sleep_ratio` | `daily_screen_time_hours` / `sleep_hours` |

Adding these engineered ratios only nudged the XGBoost cross-validation AUC from ~0.9544 to ~0.9546, so the final deployed pipeline keeps things simple and preprocesses the **raw columns only** — the marginal gain didn't justify the extra feature-engineering surface area in production.

**Preprocessing (`ColumnTransformer`):**
- **Numeric, median-imputed** (`age`, `daily_screen_time_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`, `work_study_hours`) → `SimpleImputer(strategy="median")` → `StandardScaler`
- **Numeric, zero-imputed** (`social_media_hours`, `gaming_hours`) → `SimpleImputer(strategy="constant", fill_value=0)` → `StandardScaler`
- **Categorical** (`gender`, `stress_level`, `academic_work_impact`) → `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`

---

## Evaluation Metric

Every model was scored with **ROC AUC** (`scoring="roc_auc"`) rather than accuracy, since the competition is judged on ranked probability estimates rather than hard class labels. Predictions are submitted as probabilities via `predict_proba` for the same reason.

---

## Models Compared

Seven approaches were trained through the same `ColumnTransformer` preprocessing, evaluated with 5-fold stratified cross-validation (`StratifiedKFold`, `scoring="roc_auc"`) on a stratified 15% sample of the training data, and tuned with `RandomizedSearchCV` / `GridSearchCV` where applicable. No separate held-out split was scored locally, so the "Held-out" column below is an estimate, taken as roughly 1.5–2 points lower than the CV score (mirroring the CV → held-out gap on similar projects):

| Model | CV ROC AUC | Est. Held-out ROC AUC |
|---|---|---|
| Logistic Regression | ~0.9025 | ~0.888 |
| Linear SVC | ~0.9028 | ~0.888 |
| Polynomial SVC (Calibrated, tuned) | ~0.9243 | ~0.907 |
| Decision Tree | ~0.9253 | ~0.909 |
| Random Forest (tuned) | ~0.9389 | ~0.921 |
| Voting Classifier (soft, weighted, tuned) | ~0.9463 | ~0.929 |
| **XGBoost (final, tuned)** | **~0.9521** | **~0.937** |

**Winner: XGBoost** — best cross-validation performance, and selected as the final deployed model. The soft-voting ensemble (Polynomial SVC + Random Forest + XGBoost, tuned weights `[1, 1, 2]`) came close at ~0.946 but added inference complexity for a smaller gain than tuning XGBoost directly, so it wasn't used for deployment.

**Final XGBoost hyperparameters** (found via `RandomizedSearchCV`):
```python
XGBClassifier(
    subsample=0.8,
    n_estimators=200,
    min_child_weight=15,
    max_depth=5,
    learning_rate=0.1,
    gamma=2,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc'
)
```

---

## Submission

The final model is trained on 100% of the available training data (`final_pipeline.py`), so there is no local held-out set left to score against — its true generalization performance is the score reported by Kaggle after submission.

| Training Data | CV ROC AUC | Kaggle Public Leaderboard |
|---|---|---|
| Full training set | ~0.952 | **~0.95** |

The leaderboard score closely tracks the cross-validation score, suggesting the model generalizes well and isn't meaningfully overfit to the sampled training subset used during model selection.

**Kaggle competition:** [Predicting Smartphone Addiction — Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8)

---

## How to Run Locally

### Prerequisites
- Python (see `.python-version`)
- `uv` package manager (or `pip`)

### Setup
```bash
git clone <repository-url>
cd Smartphone_Addiction

# Download train.csv and test.csv from Kaggle's Predicting Smartphone Addiction competition
# and place them inside a local Data/ folder (not included in this repo)

uv sync
```

### Train the model
```bash
uv run python final_pipeline.py
```
This regenerates `Model/smartphone_addiction.joblib`.

### Run the Streamlit app
```bash
uv run streamlit run app.py
```
Then open the URL shown in your terminal (usually `http://localhost:8501`).

---

## Technologies Used

- [scikit-learn](https://scikit-learn.org/) — pipelines, preprocessing, models, cross-validation, hyperparameter search
- [XGBoost](https://xgboost.readthedocs.io/) — gradient-boosted tree model
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data manipulation
- [Sweetviz](https://github.com/fbdesignpro/sweetviz) — exploratory data profiling
- [Streamlit](https://streamlit.io/) — interactive web app deployment
- [joblib](https://joblib.readthedocs.io/) — model serialization

---

## Key Learnings from This Project

- Building a `Pipeline`/`ColumnTransformer` that mixes different imputation strategies per column group (median vs. zero-fill vs. most-frequent)
- Testing engineered ratio features against raw columns and being willing to drop them when the AUC gain didn't justify the added complexity
- Comparing linear, tree-based, boosted, and ensemble (voting) models under an identical preprocessing pipeline
- Hyperparameter tuning with `RandomizedSearchCV` and `GridSearchCV` across multiple model families
- Choosing ROC AUC as the evaluation metric and submitting probability scores instead of hard labels
- Serializing a full pipeline (preprocessing + model) for deployment
- Deploying a trained pipeline behind a live Streamlit interface

---

## Author

Muhammad Awais Tariq

## References

- [Kaggle Predicting Smartphone Addiction Competition](https://www.kaggle.com/competitions/playground-series-s6e8)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

If you found this project useful, consider giving it a star.