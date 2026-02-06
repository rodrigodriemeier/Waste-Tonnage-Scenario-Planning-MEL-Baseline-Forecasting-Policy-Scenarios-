# Waste-Tonnage-Scenario-Planning-MEL-Baseline-Forecasting-Policy-Scenarios-
Category-level baseline forecasting of waste tonnages used as a scenario planning tool (BAU, 2030 target, post-2030 scenarios) for the Métropole Européenne de Lille (MEL). The project emphasizes the translation of projections into operational decisions, with a specific focus on hazardous waste (DDS).

# Waste Tonnage Scenario Planning — MEL

This repository presents a **waste tonnage scenario planning tool** developed as part of a group academic challenge in collaboration with the Métropole Européenne de Lille (MEL).  
My contribution covers the entire **data and modeling pipeline (baseline forecasting)** as well as the construction of **policy scenarios and operational strategies** derived directly from the projections.

## Project Objective

The goal is not to produce a causal or high-fidelity forecast, but to provide a **planning-oriented framework**:
- establish a **Business As Usual (BAU)** baseline by waste category,
- explicitly integrate a public policy target (**–15% by 2030**),
- frame post-2030 uncertainty through contrasted scenarios,
- translate projected trajectories into **operational and strategic implications**, with a particular focus on hazardous waste (DDS).

## Methodology Overview

- **Data source**: National waste statistics (ADEME – SINOE), used as a proxy due to limited local historical data.
- **Scale transfer**: National trends are converted to MEL-level orders of magnitude via demographic normalization.
- **Modeling approach**: Simple, interpretable regression models (Linear / Ridge / Lasso), trained **per waste category**.
- **Validation**: Strict temporal validation (TimeSeriesSplit), walk-forward backtesting, MAE and MAPE metrics.
- **Uncertainty handling**: Population sensitivity scenarios and residual bootstrap prediction intervals.
- **Scenario design**:
  - BAU (baseline),
  - 2030 target trajectory (–15%),
  - Scenario A: durable policy effect,
  - Scenario B: partially transitory policy effect.

## Key Insights

- Category-level analysis is critical: operational constraints are driven by **composition**, not only by total tonnage.
- **Hazardous waste (DDS)** emerges as a strategic signal despite smaller volumes, due to strong regulatory, safety, and cost constraints.
- The framework is intended for **order-of-magnitude reasoning and decision support**, not for precise local prediction.

## Repository Structure

├── src/
│ ├── data_clean.py # Data cleaning and preprocessing
│ ├── train.py # Baseline forecasting, validation, and projections
│ └── politiques.py # Policy targets and scenario construction
│
├── Data/
│ ├── dataset_clean_sinoe_france_vs_hdf.csv
│ ├── predictions_bau_ml_2022_2050.csv
│ ├── predictions_bau_ml_bycat_2022_2050.csv
│ ├── predictions_bau_ml_total_2022_2050.csv
│ ├── predictions_france_2022_2050.csv
│ ├── predictions_france_2026_2050.csv
│ ├── scenarios_mel_totaux_2022_2050.csv
│ └── tonnage-decheterie-par-type-dechet-*.csv
│
├── Projet_déchets/
│ └── Projet_déchets.pdf # Final report (methodology, results, and strategies)
│
└── README.md

## Notes and Limitations

- The use of national data as a proxy implies structural uncertainty at the local scale.
- Demographic normalization provides **orders of magnitude**, not exact estimates.
- Results should be interpreted as **scenarios**, not deterministic forecasts.

## References

- ADEME – SINOE: National waste statistics database.  
- INSEE: French population statistics (annual legal populations).

AUTHOR Rodrigo Driemeier dos Santos EESC – University of São Paulo (USP), São Carlos, Brazil — Mechatronics Engineering École Centrale de Lille, France — Generalist Engineering
📧 rodrigo.driemeier@centrale.centralelille.fr
📧 rodrigodriemeier@usp.br
