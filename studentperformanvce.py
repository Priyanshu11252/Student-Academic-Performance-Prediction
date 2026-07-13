"""
Student Academic Performance Prediction System
================================================
Predicts final academic performance (G3) using attendance, assignment/period
scores, internal assessment indicators, and other academic/behavioral factors.

Dataset: UCI "Student Performance" (Math course) - Cortez & Silva, 2008.
Source : https://archive.ics.uci.edu/dataset/320/student+performance

Pipeline: Data Cleaning -> Feature Engineering -> EDA -> Model Training
          -> Model Evaluation -> Artifact export (plots, metrics, model)
"""

import json
import warnings
import numpy as np
import pandas as pd
...
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 130

RANDOM_STATE = 42
DATA_PATH = "student-mat.csv"
OUT = "outputs"
import os
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DATA LOADING+
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=";")
print(f"Raw shape: {df.shape}")

report = {"raw_shape": df.shape}

# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------
cleaning_log = []

# 2a. Duplicates
n_dupes = df.duplicated().sum()
df = df.drop_duplicates()
cleaning_log.append(f"Removed {n_dupes} duplicate rows.")

# 2b. Missing values
missing_before = df.isnull().sum().sum()
# Numeric -> median, Categorical -> mode (dataset has none missing, but pipeline
# is written defensively so it generalizes to the user's own future data files)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for c in num_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode()[0])
cleaning_log.append(f"Missing values found: {missing_before} (imputed numeric->median, categorical->mode).")

# 2c. Outlier handling (IQR capping) on key numeric indicators
outlier_cols = ["absences", "G1", "G2"]
for c in outlier_cols:
    q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[c] < lower) | (df[c] > upper)).sum()
    df[c] = df[c].clip(lower, upper)
    cleaning_log.append(f"Capped {n_out} outliers in '{c}' to [{lower:.1f}, {upper:.1f}] (IQR method).")

# 2d. Type consistency
for c in cat_cols:
    df[c] = df[c].astype(str).str.strip()

cleaning_log.append(f"Final cleaned shape: {df.shape}")
report["cleaning_log"] = cleaning_log
report["cleaned_shape"] = df.shape
print("\n".join(cleaning_log))

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
# Attendance rate proxy: absences are counts over the year; convert to an
# attendance-rate-like feature (assuming ~180 school days).
SCHOOL_DAYS = 180
df["attendance_rate"] = 1 - (df["absences"] / SCHOOL_DAYS)
df["attendance_rate"] = df["attendance_rate"].clip(0, 1)

# Internal assessment / assignment proxies: G1 (period 1) and G2 (period 2)
# represent periodic internal assessments; we engineer aggregate + trend features.
df["avg_internal_score"] = df[["G1", "G2"]].mean(axis=1)
df["score_trend"] = df["G2"] - df["G1"]                   # improving vs declining
df["score_consistency"] = df[["G1", "G2"]].std(axis=1)    # volatility across terms

# Study effort composite
df["study_effort_index"] = (
    df["studytime"] * 0.5 + (1 - df["failures"].clip(0, 3) / 3) * 0.5
)

# Social/lifestyle risk composite (higher = more risk factors for performance)
df["lifestyle_risk_index"] = (
    df["Dalc"] + df["Walc"] + df["goout"] - df["health"]
)

# Parental education composite (proxy for home academic support)
df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

# Support composite: school support + family support + paid classes
support_map = {"yes": 1, "no": 0}
df["support_score"] = (
    df["schoolsup"].map(support_map)
    + df["famsup"].map(support_map)
    + df["paid"].map(support_map)
)

# Binary encode remaining yes/no columns
binary_cols = ["schoolsup", "famsup", "paid", "activities", "nursery",
                "higher", "internet", "romantic"]
for c in binary_cols:
    df[c + "_bin"] = df[c].map(support_map)

# Target variable: final grade G3 (scale 0-20) -> keep as regression target
target = "G3"

feature_log = [
    "attendance_rate = 1 - (absences / 180 school days), clipped to [0,1]",
    "avg_internal_score = mean(G1, G2)  [internal assessment aggregate]",
    "score_trend = G2 - G1  [improving(+) vs declining(-) trend]",
    "score_consistency = std(G1, G2)  [volatility across assessment periods]",
    "study_effort_index = 0.5*studytime + 0.5*(1 - failures/3)",
    "lifestyle_risk_index = Dalc + Walc + goout - health",
    "parent_edu_avg = mean(Medu, Fedu)",
    "support_score = schoolsup + famsup + paid (count of support systems)",
    "Binary-encoded yes/no indicators (schoolsup, famsup, paid, activities, "
    "nursery, higher, internet, romantic)",
]
report["feature_log"] = feature_log
print("\nEngineered features:\n" + "\n".join(feature_log))

df.to_csv(f"{OUT}/cleaned_engineered_data.csv", index=False)

# ---------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------------
eda_stats = {}

# 4a. Target distribution
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.histplot(df[target], bins=20, kde=True, color="#4C72B0", ax=ax)
ax.set_title("Distribution of Final Grade (G3)")
ax.set_xlabel("Final Grade (0-20)")
fig.tight_layout()
fig.savefig(f"{OUT}/eda_target_distribution.png")
plt.close(fig)

# 4b. Correlation heatmap of key engineered + raw numeric features
corr_cols = ["G3", "G1", "G2", "avg_internal_score", "attendance_rate",
             "absences", "studytime", "failures", "study_effort_index",
             "lifestyle_risk_index", "parent_edu_avg", "support_score",
             "goout", "health"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            annot_kws={"size": 7})
ax.set_title("Correlation Heatmap: Academic & Behavioral Indicators")
fig.tight_layout()
fig.savefig(f"{OUT}/eda_correlation_heatmap.png")
plt.close(fig)

eda_stats["target_correlation_with_G3"] = corr["G3"].sort_values(ascending=False).to_dict()

# 4c. Attendance vs Final Grade
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.scatterplot(data=df, x="attendance_rate", y="G3", hue="failures",
                 palette="viridis", ax=ax, alpha=0.75)
ax.set_title("Attendance Rate vs Final Grade")
ax.set_xlabel("Attendance Rate")
ax.set_ylabel("Final Grade (G3)")
fig.tight_layout()
fig.savefig(f"{OUT}/eda_attendance_vs_grade.png")
plt.close(fig)

# 4d. Internal assessment average vs Final Grade
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.regplot(data=df, x="avg_internal_score", y="G3", ax=ax,
            scatter_kws={"alpha": 0.5, "color": "#55A868"},
            line_kws={"color": "#C44E52"})
ax.set_title("Avg. Internal Assessment Score vs Final Grade")
fig.tight_layout()
fig.savefig(f"{OUT}/eda_internal_score_vs_grade.png")
plt.close(fig)

# 4e. Study time vs grade (boxplot)
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=df, x="studytime", y="G3", palette="Set2", ax=ax)
ax.set_title("Weekly Study Time Category vs Final Grade")
ax.set_xlabel("Study Time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)")
fig.tight_layout()
fig.savefig(f"{OUT}/eda_studytime_vs_grade.png")
plt.close(fig)

report["eda_stats"] = eda_stats
print("\nEDA plots saved.")

# ---------------------------------------------------------------------------
# 5. MODEL TRAINING (Regression)
# ---------------------------------------------------------------------------
# Feature set: engineered + selected raw features (exclude G3 itself, and
# exclude raw G1/G2 duplicates already captured via avg/trend to reduce leakage
# redundancy while keeping model realistic/useful)
feature_cols_numeric = [
    "attendance_rate", "avg_internal_score", "score_trend", "score_consistency",
    "study_effort_index", "lifestyle_risk_index", "parent_edu_avg",
    "support_score", "age", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]
feature_cols_categorical = ["sex", "address", "Mjob", "Fjob", "reason",
                             "guardian", "internet", "higher", "romantic"]

X = df[feature_cols_numeric + feature_cols_categorical]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), feature_cols_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_categorical),
    ]
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso Regression": Lasso(alpha=0.1, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8,
                                            random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300,
                                                     learning_rate=0.05,
                                                     max_depth=3,
                                                     random_state=RANDOM_STATE),
}

results = []
fitted_pipelines = {}
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

    results.append({
        "Model": name,
        "R2 (test)": round(r2, 4),
        "MAE (test)": round(mae, 4),
        "RMSE (test)": round(rmse, 4),
        "CV R2 mean": round(cv_scores.mean(), 4),
        "CV R2 std": round(cv_scores.std(), 4),
    })

results_df = pd.DataFrame(results).sort_values("R2 (test)", ascending=False)
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
print("\nModel comparison:\n", results_df.to_string(index=False))

report["model_results"] = results_df.to_dict(orient="records")

# ---------------------------------------------------------------------------
# 6. MODEL EVALUATION (best model deep-dive)
# ---------------------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
best_pipe = fitted_pipelines[best_name]
best_preds = best_pipe.predict(X_test)

report["best_model"] = best_name

# 6a. Predicted vs Actual
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(y_test, best_preds, alpha=0.6, color="#4C72B0", edgecolor="white")
lims = [min(y_test.min(), best_preds.min()) - 1, max(y_test.max(), best_preds.max()) + 1]
ax.plot(lims, lims, "--", color="#C44E52", label="Perfect Prediction")
ax.set_xlabel("Actual Final Grade")
ax.set_ylabel("Predicted Final Grade")
ax.set_title(f"Predicted vs Actual — {best_name}")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/eval_predicted_vs_actual.png")
plt.close(fig)

# 6b. Residual plot
residuals = y_test.values - best_preds
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.histplot(residuals, bins=20, kde=True, color="#8172B2", ax=ax)
ax.axvline(0, color="black", linestyle="--")
ax.set_title(f"Residual Distribution — {best_name}")
ax.set_xlabel("Residual (Actual - Predicted)")
fig.tight_layout()
fig.savefig(f"{OUT}/eval_residuals.png")
plt.close(fig)

# 6c. Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 4.5))
plot_df = results_df.sort_values("R2 (test)")
sns.barplot(data=plot_df, y="Model", x="R2 (test)", palette="crest", ax=ax)
ax.set_title("Model Comparison — Test R² Score")
ax.set_xlabel("R² Score")
fig.tight_layout()
fig.savefig(f"{OUT}/eval_model_comparison.png")
plt.close(fig)

# 6d. Feature importance (if tree-based best model, else use RF for reference)
importance_model_name = best_name if best_name in ["Random Forest", "Gradient Boosting"] else "Random Forest"
imp_pipe = fitted_pipelines[importance_model_name]
ohe_features = imp_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(feature_cols_categorical)
all_feature_names = feature_cols_numeric + list(ohe_features)
importances = imp_pipe.named_steps["model"].feature_importances_
imp_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=imp_series.values, y=imp_series.index, palette="flare", ax=ax)
ax.set_title(f"Top 15 Feature Importances ({importance_model_name})")
ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig(f"{OUT}/eval_feature_importance.png")
plt.close(fig)

report["top_features"] = imp_series.head(10).to_dict()

# ---------------------------------------------------------------------------
# 7. SAVE FULL REPORT (JSON) FOR DOCX GENERATION
# ---------------------------------------------------------------------------
with open(f"{OUT}/pipeline_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nBest model: {best_name}  (Test R² = {results_df.iloc[0]['R2 (test)']})")
print("\nAll outputs saved to:", OUT)