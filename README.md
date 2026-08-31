 
# 🏠 House Price Prediction (Ames Housing)

Predicting house sale prices on the **Ames Housing dataset** using Gradient Boosting, with proper cross-validation and hyperparameter tuning via `GridSearchCV`. This isn't just "train one model and ship it" — the notebook walks through four experiments, each one fixing a weakness in the last, before landing on a tuned Gradient Boosting model and analyzing exactly where it still gets things wrong.

---

## What's in here

- `notebooks/House_price_prediction.ipynb` — the full end-to-end workflow: EDA → preprocessing → four modeling experiments → cross-validation → GridSearchCV → error analysis
- `data/AmesHousing.csv` — the raw dataset (2,930 houses, 82 columns)
- `models/house_price_model.joblib` — the final, tuned Gradient Boosting pipeline (preprocessing + model bundled together)
- `src/predict.py` — loads the saved model and runs a single prediction
- `requirements.txt`

```
├── data/
│   └── AmesHousing.csv
├── models/
│   └── house_price_model.joblib
├── notebooks/
│   └── House_price_prediction.ipynb
├── src/
│   └── predict.py
├── requirements.txt
└── .gitignore
```

---

## The dataset

[Ames Housing](http://jse.amstat.org/v19n3/decock.pdf) — 2,930 residential sales in Ames, Iowa, with 82 columns describing everything from lot size and neighborhood to roof material and kitchen quality.

- **Target:** `SalePrice`
- **Dropped:** `Order` and `PID` (just row identifiers, no predictive value)
- **Left with:** 79 real features — a mix of numerical (living area, lot size, year built...) and categorical (neighborhood, house style, exterior quality...) columns

---

## Preprocessing

Built as a single `ColumnTransformer` so the same pipeline handles both feature types consistently and can be reused at prediction time:

- **Numerical columns** → missing values filled with the **median**
- **Categorical columns** → missing values filled with the **most frequent** value, then **one-hot encoded**

This preprocessor gets bundled into every model pipeline below, so there's no separate "preprocessing script" to keep in sync — the saved model already knows how to handle raw input.

---

## The experiments

Rather than jumping straight to the "best" model, the notebook builds up through a few honest iterations:

### 1. Plain Linear Regression (baseline)
A first pass to get a number on the board.
- MAE: **$17,553** · RMSE: **$30,216** · R²: **0.886**

### 2. Linear Regression on log-transformed price
House prices are right-skewed — a handful of expensive homes can pull a linear model around. Tried `log1p(SalePrice)` to compress that scale and see if it helped a linear model specifically.
- MAE: **$16,449** · RMSE: **$31,441** · R²: **0.877**

Slightly better MAE, but RMSE actually got a touch worse — a sign linear regression just isn't flexible enough to capture the real relationships in this data, transform or not.

### 3. Random Forest
Swapped in an ensemble of 300 trees to capture non-linear relationships and feature interactions.
- MAE: **$15,662** · RMSE: **$26,577** · R²: **0.912**

Clear jump in performance — tree-based models are a much better fit for this kind of tabular, mixed-type data.

### 4. Gradient Boosting
Instead of averaging independent trees, boosting builds trees sequentially, each one correcting the errors of the last.
- MAE: **$14,640** · RMSE: **$25,111** · R²: **0.921**

Best of the four, and the one worth tuning properly.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $17,553.00 | $30,216.15 | 0.8861 |
| Log Linear Regression | $16,448.84 | $31,441.25 | 0.8767 |
| Random Forest | $15,661.56 | $26,577.27 | 0.9119 |
| **Gradient Boosting** | **$14,640.05** | **$25,110.76** | **0.9214** |

---

## Cross-validation

A single train/test split can be misleading — performance can shift just because of which houses happened to land in the test set. So before tuning anything, the Gradient Boosting model was checked with **5-fold cross-validation** (shuffled, `random_state=42`):

```
Fold RMSE: [25134.18, 21429.85, 21408.90, 22364.45, 20001.62]
Mean RMSE: 22,067.80
Std RMSE:   1,708.85
```

A relatively tight standard deviation across folds — the model's performance is fairly stable regardless of which chunk of houses it's tested on.

---

## Hyperparameter tuning — GridSearchCV

Searched over the parameters that matter most for Gradient Boosting, using the same 5-fold CV split and RMSE as the scoring metric:

```python
param_grid = {
    "model__n_estimators":  [100, 300, 500],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth":     [2, 3, 4]
}
```

**Best parameters found:**
```
learning_rate = 0.1
max_depth     = 3
n_estimators  = 500
```
**Best CV RMSE:** $21,407.83 — an improvement over the untuned model's $22,067.80 average.

---

## Final evaluation

| Metric | Value |
|---|---|
| MAE | $7,201.50 |
| MSE | $90,863,144.73 |
| RMSE | $9,532.22 |
| R² | 0.9887 |

⚠️ **Honest note on this number:** `GridSearchCV` was fit on the *full* dataset (`X`, `y`) rather than only `X_train`, and the final evaluation reused `X_test` — which by that point had already been seen during grid search. That's data leakage, and it's why this RMSE looks dramatically better than the cross-validated RMSE above ($9.5K vs. ~$21K). **The $21,407.83 CV RMSE is the trustworthy number** — it's what the model would realistically do on houses it's never seen. Fixing this is the top item in the "next steps" below: fit `GridSearchCV` on `X_train`/`y_train` only, and evaluate the final model on `X_test` alone.

---

## Error analysis

Sorted every prediction by absolute error to see where the model struggles most:

| Actual | Predicted | Error |
|---|---|---|
| $135,000 | $89,791 | -$45,209 |
| $276,000 | $316,519 | +$40,519 |
| $230,000 | $269,171 | +$39,171 |
| $344,133 | $306,264 | -$37,869 |
| $246,990 | $214,518 | -$32,472 |

The biggest misses tend to cluster around homes with unusual combinations of features relative to their price bracket — the kind of outliers no amount of tuning fully fixes. An actual-vs-predicted scatter plot in the notebook backs this up: predictions track the diagonal well for typical mid-range homes, and spread out more at the high end where sales are sparser and prices more volatile.

Training set: **2,344 houses** · Test set: **586 houses** (80/20 split, `random_state=42`).

---

## Setup

```bash
git clone https://github.com/bereketkefeni-creator/house_price_prediction.git
cd  https://github.com/bereketkefeni-creator/house_price_prediction.git
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
joblib
jupyter
```

### Run the notebook
```bash
jupyter notebook notebooks/House_price_prediction.ipynb
```

### Predict on a single house
`src/predict.py` loads the saved pipeline and predicts on the first row of the dataset as a sanity check:
```bash
python src/predict.py
```
```
Predicted house price: $XXX,XXX.XX
```

Since the model is a full pipeline (preprocessing + Gradient Boosting bundled together via `joblib`), it accepts raw rows straight from the CSV — no manual encoding needed.

---

## Next steps

- **Fix the train/test leakage** in the final evaluation (see the caveat above) — this is the honest priority
- Try `RandomizedSearchCV` or Bayesian optimization to search a wider hyperparameter space without the combinatorial blowup of `GridSearchCV`
- Add feature importance / SHAP values to explain *why* the model predicts what it predicts, not just how accurately
- Compare against XGBoost / LightGBM / CatBoost
- Investigate the high-price outliers specifically — possibly a segmented or log-target approach for luxury homes

---

## Author

**Bereket (Mr. Bit)** — SW student, ASTU
GitHub: [github.com/bereketkefeni-creator](https://github.com/bereketkefeni-creator)
