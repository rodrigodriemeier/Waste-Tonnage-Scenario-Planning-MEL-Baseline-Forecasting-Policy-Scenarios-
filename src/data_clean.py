# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILENAME = "tonnage-decheterie-par-type-dechet-par-dept.csv"
REGION_NAME = "Hauts-de-France"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, FILENAME)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Fichier introuvable: {DATA_PATH}\n"
        f"-> Mets ce script .py dans le même dossier que {FILENAME}, ou change FILENAME."
    )

def sniff_sep(path, n=5000):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(n)
    seps = [",", ";", "\t", "|"]
    counts = {s: sample.count(s) for s in seps}
    return max(counts, key=counts.get)

def parse_french_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace("\u202f", "").replace("\xa0", "").replace(" ", "")
    if "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    s = re.sub(r"[^0-9\.\-+]", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan

def tvd(p, q):
    return 0.5 * np.sum(np.abs(p - q))

sep = sniff_sep(DATA_PATH)
df = pd.read_csv(DATA_PATH, sep=sep, dtype=str, encoding="utf-8", low_memory=False)
df.columns = [c.strip() for c in df.columns]

needed = ["ANNEE", "L_REGION", "L_TYP_REG_DECHET", "TONNAGE_T"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes {missing}\nColonnes disponibles: {list(df.columns)}")

df["ANNEE"] = df["ANNEE"].astype(str).str.strip()
df = df[df["ANNEE"].str.match(r"^\d{4}$", na=False)].copy()
df["ANNEE"] = df["ANNEE"].astype(int)

df["L_REGION"] = df["L_REGION"].astype(str).str.strip()
df["L_TYP_REG_DECHET"] = df["L_TYP_REG_DECHET"].astype(str).str.strip()

df["TONNAGE_T"] = df["TONNAGE_T"].apply(parse_french_number)
df = df.dropna(subset=["TONNAGE_T"]).copy()

fr_t = (
    df.groupby(["ANNEE", "L_TYP_REG_DECHET"], as_index=False)["TONNAGE_T"]
      .sum()
      .rename(columns={"TONNAGE_T": "TONNAGE_FR"})
)

hdf_t = (
    df[df["L_REGION"] == REGION_NAME]
      .groupby(["ANNEE", "L_TYP_REG_DECHET"], as_index=False)["TONNAGE_T"]
      .sum()
      .rename(columns={"TONNAGE_T": "TONNAGE_HDF"})
)

m = pd.merge(fr_t, hdf_t, on=["ANNEE", "L_TYP_REG_DECHET"], how="outer").fillna(0.0)
tot = (
    m.groupby("ANNEE", as_index=False)[["TONNAGE_FR", "TONNAGE_HDF"]]
     .sum()
     .rename(columns={"TONNAGE_FR": "TOTAL_FR", "TONNAGE_HDF": "TOTAL_HDF"})
)
m = m.merge(tot, on="ANNEE", how="left")

m["SHARE_FR"] = np.where(m["TOTAL_FR"] > 0, m["TONNAGE_FR"] / m["TOTAL_FR"], 0.0)
m["SHARE_HDF"] = np.where(m["TOTAL_HDF"] > 0, m["TONNAGE_HDF"] / m["TOTAL_HDF"], 0.0)

years = sorted(m["ANNEE"].unique())
cats = sorted(m["L_TYP_REG_DECHET"].unique())

mix_fr = (
    m.pivot_table(index="ANNEE", columns="L_TYP_REG_DECHET", values="SHARE_FR", aggfunc="sum")
     .fillna(0.0)
     .reindex(index=years, columns=cats)
)

mix_hdf = (
    m.pivot_table(index="ANNEE", columns="L_TYP_REG_DECHET", values="SHARE_HDF", aggfunc="sum")
     .fillna(0.0)
     .reindex(index=years, columns=cats)
)

dist_rows = []
for y in years:
    p = mix_fr.loc[y].to_numpy()
    q = mix_hdf.loc[y].to_numpy()
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    dist_rows.append({"ANNEE": y, "TVD": tvd(p, q)})

dist = pd.DataFrame(dist_rows)

pd.set_option("display.width", 2000)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

print("\n===============================")
print("DISTANCES TVD PAR ANNÉE (France vs Hauts-de-France)")
print("===============================")
print(dist.to_string(index=False))

plt.figure(figsize=(10, 4))
plt.plot(dist["ANNEE"], dist["TVD"], marker="o")
plt.title(f"Distance de mix (France vs {REGION_NAME}) — TVD")
plt.xlabel("Année")
plt.ylabel("TVD (0 = identique)")
plt.grid(True)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(16, 10))

ax1 = plt.subplot(2, 1, 1)
for col in mix_fr.columns:
    ax1.plot(mix_fr.index, mix_fr[col], marker="o", linewidth=1, label=col)
ax1.set_title("Mix (parts) par catégorie — France entière")
ax1.set_xlabel("Année")
ax1.set_ylabel("Part du tonnage")
ax1.grid(True)
ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

ax2 = plt.subplot(2, 1, 2)
for col in mix_hdf.columns:
    ax2.plot(mix_hdf.index, mix_hdf[col], marker="o", linewidth=1, label=col)
ax2.set_title(f"Mix (parts) par catégorie — {REGION_NAME}")
ax2.set_xlabel("Année")
ax2.set_ylabel("Part du tonnage")
ax2.grid(True)
ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()

ml_df = m[[
    "ANNEE",
    "L_TYP_REG_DECHET",
    "TONNAGE_FR",
    "TONNAGE_HDF",
    "SHARE_FR",
    "SHARE_HDF"
]].copy()

ml_df = ml_df.sort_values(["L_TYP_REG_DECHET", "ANNEE"]).reset_index(drop=True)

EXPORT_FILENAME = "dataset_clean_sinoe_france_vs_hdf.csv"
EXPORT_PATH = os.path.join(SCRIPT_DIR, EXPORT_FILENAME)
ml_df.to_csv(EXPORT_PATH, index=False, encoding="utf-8")

print("\n===============================")
print("EXPORT DATASET CLEAN POUR ML")
print("===============================")
print(f"Fichier exporté : {EXPORT_PATH}")
print(f"Shape : {ml_df.shape}")
print("Colonnes :", list(ml_df.columns))
