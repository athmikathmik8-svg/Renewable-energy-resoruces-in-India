# Renewable Energy Sources in India

**Repository:** `Renewable_Energy_Sources_in_India`  
**Project type:** Data science / supervised machine learning  
**Notebook:** Google Colab-ready Jupyter notebook

## Project overview

This project explores state/region-level renewable energy capacity from 2006–2024 and develops a Gradient Boosting model to estimate one-year-ahead changes in total installed renewable capacity.

- Source data: 635 rows, 10 columns, 34 states/regions, 2006–2024
- Technology fields: bio-mass, small hydro, solar, wind and waste-to-energy
- Prediction target: sum of the five technology capacity fields
- Training period: eligible records through 2021 (499 rows)
- Test period: 2022–2024 (102 rows)
- Model: Gradient Boosting Regressor predicting year-over-year capacity change

**Unit caution:** the supplied CSV does not document the capacity unit or data provenance. The project therefore retains the original numeric unit and does not label it MW/GW.

## Start in Google Colab

1. Download this repository ZIP or upload the notebook `Renewable_Energy_Sources_in_India_Colab.ipynb` to Google Colab.
2. Run the package-install cell.
3. Upload `renewable_energy_dataset.csv` when prompted.
4. Run all cells from top to bottom.
5. The notebook will create the prepared data, model outputs and charts.

## Run locally

```bash
python -m pip install -r requirements.txt
jupyter notebook Renewable_Energy_Sources_in_India_Colab.ipynb
```

Or run the standalone script:

```bash
python renewable_energy_model.py
```

## Data preparation

1. Validate required columns, data types, missing values, duplicate rows, duplicate state-year pairs and negative numeric values.
2. Sum the five installed-capacity columns to create `Total_Renewable_Capacity`.
3. Sort by state and year.
4. Create previous-year capacity values within each state/region.
5. Remove the first observed row per state/region from modelling because no previous-year input exists (635 to 601 usable rows).
6. One-hot encode `State` and median-impute numeric inputs inside a scikit-learn pipeline. There are no missing values in the original dataset, so imputation is a pipeline safeguard.
7. Retain high-capacity states; do not automatically delete plausible capacity outliers.
8. Avoid leakage by excluding current-year technology capacities from predictors.

## Model and evaluation

The Gradient Boosting model predicts the year-over-year change. The predicted change is added to the preceding year's observed total. It is evaluated chronologically, not with a random split.

| Model | MAE | RMSE | R² | WAPE |
|---|---:|---:|---:|---:|
| Persistence baseline | 584.36 | 2,142.44 | 0.921 | 13.71% |
| Gradient Boosting | 621.81 | 1,911.59 | 0.937 | 14.59% |

Interpretation: Gradient Boosting has lower RMSE and higher R², while the simple previous-year baseline has lower MAE and WAPE. Neither model wins on every metric.

The 2022–2024 results are a **rolling one-year-ahead backtest**. For each test year the previous year's observed capacity is used. This is not a three-year forecast from 2021 alone.

## Git and GitHub

Create an empty GitHub repository with this exact name: `Renewable_Energy_Sources_in_India`. From this project folder:

```bash
git init
git config user.name "YOUR NAME"
git config user.email "YOUR EMAIL"
git add .
git commit -m "Initial renewable energy data science project"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/Renewable_Energy_Sources_in_India.git
git push -u origin main
```

Replace the placeholders. Do not commit passwords, tokens or private data. Check that the dataset may be publicly shared before pushing it to GitHub.

## Main files

- `Renewable_Energy_Sources_in_India_Colab.ipynb` — notebook with explanations and Colab upload workflow
- `renewable_energy_model.py` — standalone pipeline script
- `renewable_energy_dataset.csv` — original input
- `renewable_energy_preprocessed.csv` — prepared modelling table
- `renewable_energy_backtest_predictions.csv` — test predictions
- `renewable_energy_model_metrics.csv` — evaluation metrics
- `renewable_energy_feature_importance.csv` — permutation feature importance
- `renewable_energy_model.joblib` — final fitted pipeline
- `plots/` — EDA and model diagnostics
- `Renewable_Energy_Sources_in_India_Project_Report.pdf` — 12-page project report

## Practical relevance and limitations

State-level projections can support preliminary discussion of grid expansion, substations, transmission capacity and storage planning. They are not engineering recommendations. Actual planning requires verified utility, demand, generation, land, network, cost and grid-connection data. A multi-year future forecast also requires future population/GSDP inputs and a recursive forecast design.