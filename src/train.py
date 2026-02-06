# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error

# -----------------------------
# CONFIG
# -----------------------------
FILENAME = "dataset_clean_sinoe_france_vs_hdf.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, FILENAME)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Fichier introuvable: {DATA_PATH}\n"
        f"-> Exécute d'abord le script de nettoyage."
    )

RANDOM_SEED = 42
N_BOOT = 2000  # bootstrap résiduel pour intervalles (simple et rapide)

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_csv(DATA_PATH)

needed = ["ANNEE", "L_TYP_REG_DECHET", "TONNAGE_FR"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes: {missing}. Colonnes dispo: {list(df.columns)}")

df["ANNEE"] = df["ANNEE"].astype(int)
df["L_TYP_REG_DECHET"] = df["L_TYP_REG_DECHET"].astype(str).str.strip()
df["TONNAGE_FR"] = pd.to_numeric(df["TONNAGE_FR"], errors="coerce")
df = df.dropna(subset=["TONNAGE_FR"]).copy()

# -----------------------------
# POP FR (INSEE) -> feature POP_FR + interpolation
# -----------------------------
POP_FR_BY_YEAR = {
    2009: 64304500,
    2011: 64933400,
    2013: 65564756,
    2015: 66422469,
    2017: 66774482,
    2019: 67257982,
    2021: 67635124,
}

df["POP_FR"] = df["ANNEE"].map(POP_FR_BY_YEAR)

if df["POP_FR"].isna().any():
    tmp = (
        pd.DataFrame(
            {
                "ANNEE": sorted(POP_FR_BY_YEAR.keys()),
                "POP_FR": [POP_FR_BY_YEAR[y] for y in sorted(POP_FR_BY_YEAR.keys())],
            }
        )
        .set_index("ANNEE")
        .sort_index()
    )
    all_years = pd.Index(sorted(df["ANNEE"].unique()), name="ANNEE")
    tmp2 = tmp.reindex(all_years).interpolate(method="linear").ffill().bfill()
    fill_map = tmp2["POP_FR"].to_dict()
    df["POP_FR"] = df["ANNEE"].map(fill_map)

# -----------------------------
# Helpers
# -----------------------------
def safe_mape(y_true, y_pred, eps=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def bootstrap_residual_intervals(y_hat, resid, n_boot=2000, q_low=0.05, q_high=0.95, rng=None):
    y_hat = np.asarray(y_hat, dtype=float)
    resid = np.asarray(resid, dtype=float)
    if resid.size == 0:
        return y_hat.copy(), y_hat.copy()
    if rng is None:
        rng = np.random.default_rng(42)
    idx = rng.integers(0, resid.size, size=(n_boot, y_hat.size))
    samples = y_hat[None, :] + resid[idx]
    lo = np.quantile(samples, q_low, axis=0)
    hi = np.quantile(samples, q_high, axis=0)
    return lo, hi

# -----------------------------
# Sort + setup
# -----------------------------
df = df.sort_values(["L_TYP_REG_DECHET", "ANNEE"]).reset_index(drop=True)

global_last_year = int(df["ANNEE"].max())
categories = sorted(df["L_TYP_REG_DECHET"].unique().tolist())

pipelines = {
    "LinearRegression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
    "Lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=20000))]),
}

param_grids = {
    "LinearRegression": {},
    "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
}

# -----------------------------
# POP scénarios (sensibilité simple)
# -----------------------------
MAX_POP_YEAR = max(POP_FR_BY_YEAR.keys())
last_pop_known = float(POP_FR_BY_YEAR[MAX_POP_YEAR])

def pop_hold_last(y: int) -> float:
    return float(POP_FR_BY_YEAR.get(int(y), last_pop_known))

def pop_cagr(y: int, annual_rate: float) -> float:
    y = int(y)
    if y <= MAX_POP_YEAR:
        return float(POP_FR_BY_YEAR.get(y, last_pop_known))
    dt = y - MAX_POP_YEAR
    return float(last_pop_known * ((1.0 + annual_rate) ** dt))

POP_SCENARIOS = {
    "HOLD": lambda y: pop_hold_last(y),
    "LOW_02PCT": lambda y: pop_cagr(y, 0.002),
    "HIGH_04PCT": lambda y: pop_cagr(y, 0.004),
}

# -----------------------------
# Train/Eval per category
# -----------------------------
best_choice_by_cat = {}
best_model_name_by_cat = {}
rows = []

print("\n===========================================")
print("BASELINE FORECAST — 1 modèle PAR CATÉGORIE")
print("Features: ANNEE + POP_FR | Validation: TimeSeriesSplit + dernier point en test")
print("Dernière année (globale):", global_last_year)
print("===========================================")

for cat in categories:
    df_cat = df[df["L_TYP_REG_DECHET"] == cat].sort_values("ANNEE").copy()
    years_cat = np.array(sorted(df_cat["ANNEE"].unique()), dtype=int)
    n_years = int(years_cat.size)

    n_splits = min(4, n_years - 2)
    if n_splits < 2:
        print(f"[SKIP] {cat}: pas assez d'années ({n_years})")
        continue

    last_year_cat = int(df_cat["ANNEE"].max())
    train_df = df_cat[df_cat["ANNEE"] < last_year_cat]
    test_df = df_cat[df_cat["ANNEE"] == last_year_cat]

    X_train = train_df[["ANNEE", "POP_FR"]].to_numpy(dtype=float)
    y_train = train_df["TONNAGE_FR"].to_numpy(dtype=float)

    X_test = test_df[["ANNEE", "POP_FR"]].to_numpy(dtype=float)
    y_test = test_df["TONNAGE_FR"].to_numpy(dtype=float)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_row = None
    best_test_mae = None

    for name, pipe in pipelines.items():
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grids[name],
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)

        best_mae_cv = -gs.best_score_
        best_params = gs.best_params_

        best_est = gs.best_estimator_
        y_pred = best_est.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)

        mae_test = mean_absolute_error(y_test, y_pred)
        mape_test = safe_mape(y_test, y_pred, eps=1.0)

        row = {
            "categorie": cat,
            "model": name,
            "test_year": last_year_cat,
            "n_years": n_years,
            "cv_mae": float(best_mae_cv),
            "test_mae": float(mae_test),
            "test_mape": float(mape_test),
            "best_params": best_params,
        }
        rows.append(row)

        if (best_test_mae is None) or (mae_test < best_test_mae):
            best_test_mae = mae_test
            best_row = row

    best_choice_by_cat[cat] = best_row
    best_model_name_by_cat[cat] = best_row["model"]
    print(
        f"[OK] {cat} -> {best_row['model']} | test_MAE={best_row['test_mae']:.2f} "
        f"| test_MAPE={100*best_row['test_mape']:.2f}% | n_years={n_years}"
    )

results_df = pd.DataFrame(rows)
if results_df.empty:
    raise RuntimeError("Aucun modèle entraîné (données insuffisantes).")

print("\n===========================================")
print("MEILLEUR MODÈLE PAR CATÉGORIE (trié par test_mae)")
print("===========================================")
best_df = (
    pd.DataFrame(list(best_choice_by_cat.values()))
    .sort_values("test_mae", ascending=True)
    .reset_index(drop=True)
)
print(best_df[["categorie", "model", "cv_mae", "test_mae", "test_mape", "test_year", "n_years", "best_params"]])

# -----------------------------
# Backtest walk-forward (stabilité / dispersion d'erreur) — option léger
# -----------------------------
bt_rows = []
for cat, best_info in best_choice_by_cat.items():
    df_cat = df[df["L_TYP_REG_DECHET"] == cat].sort_values("ANNEE").copy()
    years = np.array(sorted(df_cat["ANNEE"].unique()), dtype=int)
    n = int(years.size)

    if n < 6:
        continue

    model_name = best_info["model"]
    best_params = best_info["best_params"]

    pipe = pipelines[model_name]
    if isinstance(best_params, dict) and len(best_params) > 0:
        pipe = pipelines[model_name].set_params(**best_params)

    test_years = years[-3:]
    for ty in test_years:
        train_df = df_cat[df_cat["ANNEE"] < ty]
        test_df = df_cat[df_cat["ANNEE"] == ty]
        if train_df.empty or test_df.empty:
            continue

        X_tr = train_df[["ANNEE", "POP_FR"]].to_numpy(dtype=float)
        y_tr = train_df["TONNAGE_FR"].to_numpy(dtype=float)

        X_te = test_df[["ANNEE", "POP_FR"]].to_numpy(dtype=float)
        y_te = test_df["TONNAGE_FR"].to_numpy(dtype=float)

        pipe.fit(X_tr, y_tr)
        y_hat = np.clip(pipe.predict(X_te), 0, None)

        bt_rows.append(
            {
                "categorie": cat,
                "model": model_name,
                "test_year": int(ty),
                "mae": float(mean_absolute_error(y_te, y_hat)),
                "mape": float(safe_mape(y_te, y_hat, eps=1.0)),
                "n_train_years": int(train_df["ANNEE"].nunique()),
            }
        )

backtest_df = pd.DataFrame(bt_rows)
if not backtest_df.empty:
    bt_summary = (
        backtest_df.groupby(["categorie", "model"], as_index=False)
        .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"), mape_mean=("mape", "mean"), mape_std=("mape", "std"))
        .sort_values("mae_mean", ascending=True)
        .reset_index(drop=True)
    )
else:
    bt_summary = pd.DataFrame(columns=["categorie", "model", "mae_mean", "mae_std", "mape_mean", "mape_std"])

# -----------------------------
# Refit + forecast 2022..2050 + intervalles (bootstrap résiduel)
# -----------------------------
future_years = list(range(global_last_year + 1, 2051))
rng = np.random.default_rng(RANDOM_SEED)

pred_rows = []
resid_rows = []

for cat, best_info in best_choice_by_cat.items():
    df_cat = df[df["L_TYP_REG_DECHET"] == cat].sort_values("ANNEE").copy()

    X_all = df_cat[["ANNEE", "POP_FR"]].to_numpy(dtype=float)
    y_all = df_cat["TONNAGE_FR"].to_numpy(dtype=float)

    model_name = best_info["model"]
    best_params = best_info["best_params"]

    final_pipe = pipelines[model_name]
    if isinstance(best_params, dict) and len(best_params) > 0:
        final_pipe.set_params(**best_params)

    final_pipe.fit(X_all, y_all)

    y_hat_in = np.clip(final_pipe.predict(X_all), 0, None)
    resid = (y_all - y_hat_in).astype(float)
    resid = resid[np.isfinite(resid)]
    resid_rows.append({"categorie": cat, "resid_mean": float(np.mean(resid)), "resid_std": float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0})

    for scen_name, pop_fn in POP_SCENARIOS.items():
        X_f = np.array([[y, pop_fn(y)] for y in future_years], dtype=float)
        y_f = np.clip(final_pipe.predict(X_f), 0, None)

        lo, hi = bootstrap_residual_intervals(y_f, resid, n_boot=N_BOOT, q_low=0.05, q_high=0.95, rng=rng)
        lo = np.clip(lo, 0, None)
        hi = np.clip(hi, 0, None)

        for i, y in enumerate(future_years):
            pred_rows.append(
                {
                    "ANNEE": int(y),
                    "POP_FR": float(pop_fn(y)),
                    "POP_SCENARIO": scen_name,
                    "L_TYP_REG_DECHET": cat,
                    "TONNAGE_PRED_FR": float(y_f[i]),
                    "PI05_FR": float(lo[i]),
                    "PI95_FR": float(hi[i]),
                }
            )

future_pred_df = pd.DataFrame(pred_rows)
resid_df = pd.DataFrame(resid_rows)

# -----------------------------
# EXPORTS (compat + enrichis)
# -----------------------------
# Fichiers historiques du projet (pour compat avec politiques.py)
EXPORT_BAU_BYCAT_FILENAME = f"predictions_bau_ml_bycat_{future_years[0]}_{future_years[-1]}.csv"
EXPORT_BAU_BYCAT_PATH = os.path.join(SCRIPT_DIR, EXPORT_BAU_BYCAT_FILENAME)

EXPORT_TOTAL_FILENAME = f"predictions_bau_ml_total_{future_years[0]}_{future_years[-1]}.csv"
EXPORT_TOTAL_PATH = os.path.join(SCRIPT_DIR, EXPORT_TOTAL_FILENAME)

# On garde le scénario HOLD comme "BAU" principal
bau_hold_bycat = future_pred_df[future_pred_df["POP_SCENARIO"] == "HOLD"].copy()
bau_hold_bycat[["ANNEE", "POP_FR", "L_TYP_REG_DECHET", "TONNAGE_PRED_FR"]].to_csv(
    EXPORT_BAU_BYCAT_PATH, index=False, encoding="utf-8"
)

total_pred = bau_hold_bycat.groupby("ANNEE", as_index=False)["TONNAGE_PRED_FR"].sum()
total_pred.to_csv(EXPORT_TOTAL_PATH, index=False, encoding="utf-8")

# Exports enrichis (scénarios + intervalles)
EXPORT_ENRICHED_BYCAT_FILENAME = f"predictions_baseline_bycat_scenarios_pi_{future_years[0]}_{future_years[-1]}.csv"
EXPORT_ENRICHED_BYCAT_PATH = os.path.join(SCRIPT_DIR, EXPORT_ENRICHED_BYCAT_FILENAME)
future_pred_df.to_csv(EXPORT_ENRICHED_BYCAT_PATH, index=False, encoding="utf-8")

EXPORT_BACKTEST_FILENAME = "backtest_walkforward_summary_bycat.csv"
EXPORT_BACKTEST_PATH = os.path.join(SCRIPT_DIR, EXPORT_BACKTEST_FILENAME)
bt_summary.to_csv(EXPORT_BACKTEST_PATH, index=False, encoding="utf-8")

EXPORT_BACKTEST_RAW_FILENAME = "backtest_walkforward_raw_bycat.csv"
EXPORT_BACKTEST_RAW_PATH = os.path.join(SCRIPT_DIR, EXPORT_BACKTEST_RAW_FILENAME)
backtest_df.to_csv(EXPORT_BACKTEST_RAW_PATH, index=False, encoding="utf-8")

EXPORT_RESID_FILENAME = "residuals_in_sample_stats_bycat.csv"
EXPORT_RESID_PATH = os.path.join(SCRIPT_DIR, EXPORT_RESID_FILENAME)
resid_df.to_csv(EXPORT_RESID_PATH, index=False, encoding="utf-8")

print("\n===========================================")
print("EXPORTS — OK")
print("===========================================")
print("BAU (compat) BYCAT :", EXPORT_BAU_BYCAT_PATH)
print("BAU (compat) TOTAL :", EXPORT_TOTAL_PATH)
print("ENRICHI (scénarios+PI) :", EXPORT_ENRICHED_BYCAT_PATH)
print("BACKTEST summary :", EXPORT_BACKTEST_PATH)
print("BACKTEST raw :", EXPORT_BACKTEST_RAW_PATH)
print("Résidus (in-sample stats) :", EXPORT_RESID_PATH)

print("\nAperçu BAU HOLD BYCAT:")
print(bau_hold_bycat.head(10))

print("\nAperçu ENRICHI (scénarios+PI):")
print(future_pred_df.head(10))

if not bt_summary.empty:
    print("\nRésumé backtest (walk-forward) — stabilité:")
    print(bt_summary.head(10))
