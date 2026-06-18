"""
CARA PAKAI:
    1. pip install xgboost lightgbm catboost optuna scikit-learn pandas numpy
    2. python loan_model_trainer.py
    3. Salin pipe_ensemble.pkl + model_meta.json ke folder dashboard

OUTPUT:
    pipe_ensemble.pkl   → model siap dipanggil di Streamlit
    model_meta.json     → metadata akurasi & fitur
    model_report.txt    → classification report lengkap
"""

import warnings
warnings.filterwarnings("ignore")

import os, pickle, json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import OrdinalEncoder, StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import StackingClassifier
from sklearn.metrics          import (accuracy_score, roc_auc_score, f1_score,
                                      classification_report, confusion_matrix)
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
CSV_PATH      = "loan_data_new.csv"
OUTPUT_PKL    = "pipe_ensemble.pkl"
OUTPUT_META   = "model_meta.json"
OUTPUT_REPORT = "model_report.txt"
OPTUNA_TRIALS = 40       
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5


COL_TARGET     = "Loan Status"
COL_AGE        = "Age"
COL_GENDER     = "Gender"
COL_EDUCATION  = "Education"
COL_INCOME     = "Person Income"
COL_EXP        = "Employee Experience"
COL_HOME       = "Home Onwership"       
COL_LOAN_AMT   = "Loan Amount"
COL_INTENT     = "Loan Intent"
COL_RATE       = "Loan interest Rate"
COL_DTI        = "Loan percentage"
COL_CRED_HIST  = "Credit History"
COL_CRED_SCORE = "Credit Score"
COL_PREV_LOAN  = "Previous Loan"

EDUCATION_ORDER = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]

# ══════════════════════════════════════════════════════
#  STEP 1 — LOAD & RENAME
# ══════════════════════════════════════════════════════
def load_dataset(path: str) -> pd.DataFrame:
    print(f"[1/6] Memuat dataset dari '{path}' ...")
    df = pd.read_csv(path)

    rename_map = {
        "person_age"                    : COL_AGE,
        "person_gender"                 : COL_GENDER,
        "person_education"              : COL_EDUCATION,
        "person_income"                 : COL_INCOME,
        "person_emp_exp"                : COL_EXP,
        "person_home_ownership"         : COL_HOME,
        "loan_amnt"                     : COL_LOAN_AMT,
        "loan_intent"                   : COL_INTENT,
        "loan_int_rate"                 : COL_RATE,
        "loan_percent_income"           : COL_DTI,
        "cb_person_cred_hist_length"    : COL_CRED_HIST,
        "credit_score"                  : COL_CRED_SCORE,
        "previous_loan_defaults_on_file": COL_PREV_LOAN,
        "loan_status"                   : COL_TARGET,
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    print(f"    Shape  : {df.shape[0]:,} baris × {df.shape[1]} kolom")
    print(f"    Target :\n{df[COL_TARGET].value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════
#  STEP 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════
def encode_previous_loan(series: pd.Series) -> pd.Series:
    """
    Konversi kolom Previous Loan ke int 0/1.
    Tangani semua kemungkinan dtype: object biasa, ArrowDtype (pandas 3.x+),
    category, bool, atau sudah int/float.
    """
    as_str = series.apply(lambda x: str(x).strip().lower() if x is not None else "nan")

    string_map = {
        "yes": 1, "no": 0,
        "y":   1, "n":  0,
        "1":   1, "0":  0,
        "1.0": 1, "0.0": 0,
        "true":1, "false":0,
    }

    mapped = as_str.map(string_map)

    if mapped.isna().any():
        numeric = pd.to_numeric(as_str, errors="coerce")
        mapped  = mapped.fillna(numeric)

    # Nilai tak dikenal → 0
    return mapped.fillna(0).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Feature engineering ...")
    df = df.copy()

    # ── 1. Encode Previous Loan  ──
    df[COL_PREV_LOAN] = encode_previous_loan(df[COL_PREV_LOAN])
    print(f"    Previous Loan encoded → {df[COL_PREV_LOAN].value_counts().to_dict()}")

    # ── 2. Rasio finansial baru ──
    df["loan_to_income_ratio"]   = df[COL_LOAN_AMT] / (df[COL_INCOME] + 1)
    df["annual_interest_burden"] = (df[COL_LOAN_AMT] * df[COL_RATE] / 100) / (df[COL_INCOME] + 1)
    df["income_per_exp_year"]    = df[COL_INCOME] / (df[COL_EXP] + 1)
    df["score_per_hist_year"]    = df[COL_CRED_SCORE] / (df[COL_CRED_HIST] + 1)
    df["affordability_index"]    = df[COL_DTI] * df[COL_RATE]
    df["credit_risk_score"]      = (df[COL_CRED_SCORE] / 850) - df[COL_DTI]
    df["financial_maturity"]     = df[COL_AGE] * 0.4 + df[COL_EXP] * 0.6

    # ── 3. Binning credit score ──
    df["credit_tier"] = pd.cut(
        df[COL_CRED_SCORE],
        bins=[0, 579, 669, 739, 850],
        labels=["Poor", "Fair", "Good", "Excellent"]
    ).astype(str)

    engineered = [
        "loan_to_income_ratio", "annual_interest_burden", "income_per_exp_year",
        "score_per_hist_year", "affordability_index", "credit_risk_score",
        "financial_maturity", "credit_tier",
    ]
    print(f"    {len(engineered)} fitur baru: {engineered}")
    return df


# ══════════════════════════════════════════════════════
#  STEP 3 — PREPROCESSOR
# ══════════════════════════════════════════════════════
def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    cat_ordinal = [COL_EDUCATION]
    cat_nominal = [COL_GENDER, COL_HOME, COL_INTENT, "credit_tier"]
    num_cols    = [
        COL_AGE, COL_INCOME, COL_EXP, COL_LOAN_AMT,
        COL_RATE, COL_DTI, COL_CRED_HIST, COL_CRED_SCORE, COL_PREV_LOAN,
        "loan_to_income_ratio", "annual_interest_burden", "income_per_exp_year",
        "score_per_hist_year", "affordability_index", "credit_risk_score",
        "financial_maturity",
    ]
    num_cols = [c for c in num_cols if c in X_train.columns]

    # Validasi: semua kolom numerik harus sudah bertipe numerik
    for col in num_cols:
        sample = X_train[col].dropna()
        if len(sample) > 0:
            try:
                pd.to_numeric(sample.head(5))
            except Exception:
                raise ValueError(
                    f"Kolom '{col}' mengandung nilai non-numerik: "
                    f"{sample.unique()[:5].tolist()}. "
                    f"Pastikan engineer_features() sudah dijalankan."
                )

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("edu", OrdinalEncoder(
            categories=[EDUCATION_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ), cat_ordinal),
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ), cat_nominal),
    ], remainder="drop")

    return preprocessor


# ══════════════════════════════════════════════════════
#  STEP 4 — OPTUNA TUNING
# ══════════════════════════════════════════════════════
def tune_xgboost(X, y, n_trials=OPTUNA_TRIALS):
    print(f"    Tuning XGBoost ({n_trials} trials) ...")
    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 200, 600),
            "max_depth"        : trial.suggest_int("max_depth", 3, 9),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
            "gamma"            : trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 0.5, 5.0),
            "eval_metric"      : "logloss",
            "random_state"     : RANDOM_STATE,
            "n_jobs"           : -1,
        }
        cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(XGBClassifier(**params), X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({"eval_metric": "logloss", "random_state": RANDOM_STATE, "n_jobs": -1})
    print(f"      Best XGB AUC: {study.best_value:.4f} | depth={best['max_depth']}, lr={best['learning_rate']:.4f}")
    return best


def tune_lgbm(X, y, n_trials=OPTUNA_TRIALS):
    print(f"    Tuning LightGBM ({n_trials} trials) ...")
    def objective(trial):
        params = {
            "n_estimators"      : trial.suggest_int("n_estimators", 200, 600),
            "max_depth"         : trial.suggest_int("max_depth", 3, 12),
            "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves"        : trial.suggest_int("num_leaves", 20, 120),
            "subsample"         : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples" : trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha"         : trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda"        : trial.suggest_float("reg_lambda", 0.5, 5.0),
            "random_state"      : RANDOM_STATE,
            "n_jobs"            : -1,
            "verbosity"         : -1,
        }
        cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(LGBMClassifier(**params), X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1})
    print(f"      Best LGBM AUC: {study.best_value:.4f} | leaves={best['num_leaves']}, lr={best['learning_rate']:.4f}")
    return best


def tune_catboost(X, y, n_trials=OPTUNA_TRIALS):
    print(f"    Tuning CatBoost ({n_trials} trials) ...")
    def objective(trial):
        params = {
            "iterations"          : trial.suggest_int("iterations", 200, 600),
            "depth"               : trial.suggest_int("depth", 3, 9),
            "learning_rate"       : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg"         : trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature" : trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength"     : trial.suggest_float("random_strength", 0.0, 1.0),
            "random_seed"         : RANDOM_STATE,
            "verbose"             : 0,
        }
        cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(CatBoostClassifier(**params), X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({"random_seed": RANDOM_STATE, "verbose": 0})
    print(f"      Best CatBoost AUC: {study.best_value:.4f} | depth={best['depth']}, lr={best['learning_rate']:.4f}")
    return best


# ══════════════════════════════════════════════════════
#  STEP 5 — STACKING ENSEMBLE
# ══════════════════════════════════════════════════════
def build_stacking_pipeline(preprocessor, xgb_p, lgbm_p, cb_p):
    stacking = StackingClassifier(
        estimators=[
            ("xgb",  XGBClassifier(**xgb_p)),
            ("lgbm", LGBMClassifier(**lgbm_p)),
            ("cb",   CatBoostClassifier(**cb_p)),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000,
                                            random_state=RANDOM_STATE, solver="lbfgs"),
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("stacking", stacking)])


# ══════════════════════════════════════════════════════
#  STEP 6 — EVALUATE & SAVE
# ══════════════════════════════════════════════════════
def evaluate_and_save(pipeline, X_test, y_test, feature_cols):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred, target_names=["Rejected", "Approved"])

    print("\n" + "═"*58)
    print("  EVALUASI MODEL AKHIR")
    print("═"*58)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{cr}")
    print("═"*58)

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n✅ Model    → '{OUTPUT_PKL}'")

    meta = {
        "model_type"      : "StackingClassifier (XGB + LGBM + CatBoost → LogReg)",
        "accuracy"        : round(acc, 4),
        "auc_roc"         : round(auc, 4),
        "f1_score"        : round(f1, 4),
        "feature_columns" : feature_cols,
        "n_features"      : len(feature_cols),
        "education_order" : EDUCATION_ORDER,
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Metadata → '{OUTPUT_META}'")

    report = f"""LOANSCOPE ENSEMBLE MODEL REPORT
================================
Model    : Stacking (XGBoost + LightGBM + CatBoost → Logistic Regression)
Accuracy : {acc:.4f}  ({acc*100:.2f}%)
AUC-ROC  : {auc:.4f}
F1-Score : {f1:.4f}

Confusion Matrix:
{cm}

Classification Report:
{cr}

Fitur yang digunakan ({len(feature_cols)}):
{chr(10).join('  - ' + c for c in feature_cols)}
"""
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ Laporan  → '{OUTPUT_REPORT}'")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    print("ENSEMBLE MODEL TRAINER")

    # 1. Load
    df = load_dataset(CSV_PATH)

    # 2. Feature engineering (termasuk encode Previous Loan)
    df = engineer_features(df)

    # 3. Split
    feature_cols = [
        COL_AGE, COL_GENDER, COL_EDUCATION, COL_INCOME, COL_EXP,
        COL_HOME, COL_LOAN_AMT, COL_INTENT, COL_RATE, COL_DTI,
        COL_CRED_HIST, COL_CRED_SCORE, COL_PREV_LOAN,
        "loan_to_income_ratio", "annual_interest_burden",
        "income_per_exp_year", "score_per_hist_year",
        "affordability_index", "credit_risk_score",
        "financial_maturity", "credit_tier",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[COL_TARGET].astype(int)

    print(f"[3/6] Train-test split (80/20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"    Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # 4. Preprocessor
    print("[4/6] Membangun preprocessor pipeline")
    preprocessor  = build_preprocessor(X_train)
    X_train_proc  = preprocessor.fit_transform(X_train)
    X_test_proc   = preprocessor.transform(X_test)
    print(f"    Shape setelah preprocessing: {X_train_proc.shape}")

    # 5. Optuna tuning
    print("\n[5/6] Optuna hyperparameter search")
    xgb_params  = tune_xgboost(X_train_proc, y_train.values)
    lgbm_params = tune_lgbm(X_train_proc, y_train.values)
    cb_params   = tune_catboost(X_train_proc, y_train.values)

    # 6. Train stacking + save
    print("\n[6/6] Melatih Stacking Ensemble")
    pipeline = build_stacking_pipeline(preprocessor, xgb_params, lgbm_params, cb_params)
    pipeline.fit(X_train, y_train)
    print("    Training selesai ✓")

    evaluate_and_save(pipeline, X_test, y_test, feature_cols)

    print("\n" + "━"*58)
    print("  SELESAI,Salin ke folder dashboard:")
    print(f"    ✅ {OUTPUT_PKL}")
    print(f"    ✅ {OUTPUT_META}")
    print("━"*58 + "\n")


if __name__ == "__main__":
    main()