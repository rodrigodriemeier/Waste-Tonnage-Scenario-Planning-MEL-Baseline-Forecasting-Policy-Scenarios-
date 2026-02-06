# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLEAN_FILENAME = "dataset_clean_sinoe_france_vs_hdf.csv"
BAU_BYCAT_FILENAME = "predictions_bau_ml_bycat_2022_2050.csv"

# Si ton train.py enrichi a été exécuté, tu auras aussi ce fichier:
ENRICHED_BYCAT_FILENAME = "predictions_baseline_bycat_scenarios_pi_2022_2050.csv"

TARGET_REDUCTION = 0.15
BASE_YEAR = 2010
TARGET_YEAR = 2030
END_YEAR = 2050

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CLEAN_PATH = os.path.join(SCRIPT_DIR, CLEAN_FILENAME)
BAU_BYCAT_PATH = os.path.join(SCRIPT_DIR, BAU_BYCAT_FILENAME)
ENRICHED_BYCAT_PATH = os.path.join(SCRIPT_DIR, ENRICHED_BYCAT_FILENAME)

for p in [CLEAN_PATH, BAU_BYCAT_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Introuvable: {p}")

has_enriched = os.path.exists(ENRICHED_BYCAT_PATH)

df = pd.read_csv(CLEAN_PATH)
bau_cat = pd.read_csv(BAU_BYCAT_PATH)

df["ANNEE"] = df["ANNEE"].astype(int)
bau_cat["ANNEE"] = bau_cat["ANNEE"].astype(int)

POP_MEL = 1_194_040
POP_FR_BY_YEAR = {
    2009: 64304500, 2010: 64612939, 2011: 64933400, 2012: 65241241,
    2013: 65564756, 2014: 66130873, 2015: 66422469, 2016: 66602645,
    2017: 66774482, 2018: 66992159, 2019: 67257982,
    2020: 67454122, 2021: 67626396,
}
MAX_POP_YEAR = max(POP_FR_BY_YEAR.keys())

def pop_fr(y: int) -> int:
    return int(POP_FR_BY_YEAR.get(int(y), POP_FR_BY_YEAR[MAX_POP_YEAR]))

def norm_mel(y: int, v):
    return np.asarray(v, dtype=float) * (POP_MEL / pop_fr(int(y)))

BAU_COL = [c for c in bau_cat.columns if c.upper().startswith("TONNAGE_PRED")][0]

hist_total = df.groupby("ANNEE")["TONNAGE_FR"].sum().sort_index()

categories = sorted(bau_cat["L_TYP_REG_DECHET"].unique().tolist())

bau_pivot = (
    bau_cat.pivot_table(index="ANNEE", columns="L_TYP_REG_DECHET", values=BAU_COL, aggfunc="sum")
    .sort_index()
    .reindex(columns=categories)
    .clip(lower=0)
)

hist_pivot = (
    df.pivot_table(index="ANNEE", columns="L_TYP_REG_DECHET", values="TONNAGE_FR", aggfunc="sum")
    .sort_index()
    .reindex(columns=categories)
)

last_hist_year = int(hist_total.index.max())

future_years = bau_pivot.index[bau_pivot.index > last_hist_year]
if len(future_years) == 0:
    raise ValueError("Aucune année de prédiction > historique dans BAU bycat.")
first_pred_year = int(future_years.min())

end_year = int(min(END_YEAR, int(bau_pivot.index.max())))

if TARGET_YEAR not in bau_pivot.index:
    raise ValueError(f"BAU bycat ne contient pas {TARGET_YEAR}. Dispo: {bau_pivot.index.min()}→{bau_pivot.index.max()}")

bau_2030_total = float(bau_pivot.loc[TARGET_YEAR].sum())

if BASE_YEAR in hist_total.index:
    baseline_year = int(BASE_YEAR)
else:
    baseline_year = int(hist_total.index[np.argmin(np.abs(hist_total.index.to_numpy() - BASE_YEAR))])

baseline_total = float(hist_total.loc[baseline_year])
target_2030_total = baseline_total * (1.0 - TARGET_REDUCTION)

factor = (target_2030_total / bau_2030_total) if bau_2030_total > 0 else 0.0

target_2030_bycat = (bau_pivot.loc[TARGET_YEAR] * factor).clip(lower=0)

meta_years = np.arange(first_pred_year, TARGET_YEAR + 1)
meta_bycat = pd.DataFrame(index=meta_years, columns=categories, dtype=float)
for cat in categories:
    v0 = float(bau_pivot.loc[first_pred_year, cat])
    v1 = float(target_2030_bycat.loc[cat])
    meta_bycat[cat] = np.linspace(v0, v1, len(meta_years))
meta_bycat = meta_bycat.clip(lower=0)

post_years = np.arange(TARGET_YEAR, end_year + 1)

A_bycat = (bau_pivot.loc[post_years] * factor).clip(lower=0)

bau_2030_bycat = bau_pivot.loc[TARGET_YEAR].astype(float)
B_bycat = bau_pivot.loc[post_years].astype(float).sub(bau_2030_bycat, axis=1).add(target_2030_bycat, axis=1)
B_bycat = B_bycat.clip(lower=0)

def df_to_mel(df_):
    out = df_.copy().astype(float)
    for y in out.index:
        out.loc[int(y)] = norm_mel(int(y), out.loc[int(y)].to_numpy())
    return out

hist_mel = df_to_mel(hist_pivot)
bau_mel = df_to_mel(bau_pivot.loc[first_pred_year:end_year])
meta_mel = df_to_mel(meta_bycat)
A_mel = df_to_mel(A_bycat)
B_mel = df_to_mel(B_bycat)

# -----------------------------
# Option: si fichier enrichi existe, on trace aussi des bandes d'incertitude (PI05/PI95)
# -----------------------------
pi_available = False
if has_enriched:
    enriched = pd.read_csv(ENRICHED_BYCAT_PATH)
    enriched["ANNEE"] = enriched["ANNEE"].astype(int)
    req_cols = {"ANNEE", "L_TYP_REG_DECHET", "POP_SCENARIO", "TONNAGE_PRED_FR", "PI05_FR", "PI95_FR"}
    pi_available = req_cols.issubset(set(enriched.columns))
    if pi_available:
        enriched = enriched[(enriched["POP_SCENARIO"] == "HOLD")].copy()
        pi_pivot_lo = (
            enriched.pivot_table(index="ANNEE", columns="L_TYP_REG_DECHET", values="PI05_FR", aggfunc="sum")
            .sort_index()
            .reindex(columns=categories)
            .clip(lower=0)
        )
        pi_pivot_hi = (
            enriched.pivot_table(index="ANNEE", columns="L_TYP_REG_DECHET", values="PI95_FR", aggfunc="sum")
            .sort_index()
            .reindex(columns=categories)
            .clip(lower=0)
        )
        pi_total_lo = pi_pivot_lo.sum(axis=1)
        pi_total_hi = pi_pivot_hi.sum(axis=1)

# -----------------------------
# PLOTS — par catégorie (normalisé MEL)
# -----------------------------
for cat in categories:
    plt.figure(figsize=(12, 5))

    h = hist_mel[cat].dropna()
    if len(h) > 0:
        plt.plot(h.index, h.values, marker="o", linewidth=2, label="Historique — MEL", zorder=6)

    b = bau_mel[cat].dropna()
    if len(b) > 0:
        plt.plot(b.index, b.values, "--o", linewidth=2, label="BAU (baseline) — MEL", zorder=3)

    m = meta_mel[cat].dropna()
    if len(m) > 0:
        plt.plot(m.index, m.values, "--o", linewidth=3, label="Trajectoire cible — MEL", zorder=10)

    a = A_mel[cat].dropna()
    if len(a) > 0:
        plt.plot(a.index, a.values, "--o", linewidth=2.2, label="Scénario A — MEL", zorder=7)

    bb = B_mel[cat].dropna()
    if len(bb) > 0:
        plt.plot(bb.index, bb.values, "--o", linewidth=2.2, label="Scénario B — MEL", zorder=7)

    if pi_available:
        lo = df_to_mel(pi_pivot_lo.loc[first_pred_year:end_year])[cat].dropna()
        hi = df_to_mel(pi_pivot_hi.loc[first_pred_year:end_year])[cat].dropna()
        if len(lo) > 0 and len(hi) > 0:
            plt.fill_between(lo.index, lo.values, hi.values, alpha=0.15, label="Incertitude (PI 5–95%)", zorder=1)

    plt.axvline(last_hist_year, color="gray", linestyle=":")
    plt.axvline(TARGET_YEAR, color="black", linestyle=":")

    plt.title(f"Tonnage — {cat} (normalisé MEL)")
    plt.xlabel("Année")
    plt.ylabel("Tonnes")
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.show()

print("\n--- DEBUG ---")
print("baseline_year:", baseline_year)
print("first_pred_year:", first_pred_year)
print("end_year:", end_year)
print("factor:", factor)
print("check A(2030) == cible (par cat) -> max abs diff:",
      float((A_bycat.loc[TARGET_YEAR] - target_2030_bycat).abs().max()))
print("check B(2030) == cible (par cat) -> max abs diff:",
      float((B_bycat.loc[TARGET_YEAR] - target_2030_bycat).abs().max()))

# ==============================
# TOTAL — HIST + CIBLE + BAU + A + B (normalisé MEL)
# ==============================
hist_total_mel = pd.Series(
    {int(y): float(norm_mel(int(y), v)) for y, v in hist_total.items()}
).sort_index()

bau_total_mel = pd.Series(
    {int(y): float(norm_mel(int(y), bau_pivot.loc[y].sum()))
     for y in bau_pivot.index if first_pred_year <= int(y) <= end_year}
).sort_index()

meta_total_mel = meta_mel.sum(axis=1).sort_index()
A_total_mel = A_mel.sum(axis=1).sort_index()
B_total_mel = B_mel.sum(axis=1).sort_index()

plt.figure(figsize=(16, 7))

plt.plot(hist_total_mel.index, hist_total_mel.values, marker="o", linewidth=2, label="Historique (dataset) — MEL", zorder=6)
plt.plot(bau_total_mel.index, bau_total_mel.values, marker="o", linestyle="--", linewidth=2, label="BAU (baseline) — MEL", zorder=3)
plt.plot(meta_total_mel.index, meta_total_mel.values, marker="o", linestyle="--", linewidth=3.5, label=f"Trajectoire cible (→{TARGET_YEAR}) — MEL", zorder=10)
plt.plot(A_total_mel.index, A_total_mel.values, marker="o", linestyle="--", linewidth=2.5, label="Scénario A — MEL", zorder=7)
plt.plot(B_total_mel.index, B_total_mel.values, marker="o", linestyle="--", linewidth=2.5, label="Scénario B — MEL", zorder=7)

if pi_available:
    lo = pd.Series({int(y): float(norm_mel(int(y), pi_total_lo.loc[y])) for y in pi_total_lo.index if first_pred_year <= int(y) <= end_year}).sort_index()
    hi = pd.Series({int(y): float(norm_mel(int(y), pi_total_hi.loc[y])) for y in pi_total_hi.index if first_pred_year <= int(y) <= end_year}).sort_index()
    if len(lo) > 0 and len(hi) > 0:
        plt.fill_between(lo.index, lo.values, hi.values, alpha=0.15, label="Incertitude (PI 5–95%)", zorder=1)

plt.axvline(last_hist_year, color="gray", linestyle=":", label="Frontière historique / prévision")
plt.axvline(TARGET_YEAR, color="black", linestyle=":", label=f"{TARGET_YEAR} (cible)")

plt.title("TOTAL tonnages — Historique + BAU + trajectoire cible + scénarios A/B (normalisé MEL)")
plt.xlabel("Année")
plt.ylabel("Tonnage total estimé MEL (tonnes)")
plt.grid(True)
plt.ylim(bottom=0)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
