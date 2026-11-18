# Renewable Energy Sources in India

**GitHub repository:** https://github.com/athmikathmik8-svg/Renewable-energy-resoruces-in-India  
**Project name:** `Renewable_Energy_Sources_in_India`  
**Project type:** Data science and supervised machine learning  
**Notebook:** Google Colab-ready

[Open the notebook in Google Colab](https://colab.research.google.com/github/athmikathmik8-svg/Renewable-energy-resoruces-in-India/blob/main/Renewable_Energy_Sources_in_India_Colab.ipynb)

## Project overview

This project explores state/region-level renewable energy capacity from 2006–2024 and builds a Gradient Boosting regression model to estimate one-year-ahead changes in total installed renewable capacity.

- Source dataset: 635 records, 10 columns, 34 states/regions, 2006–2024.
- Five technology fields: bio-mass, small hydro, solar, wind and waste-to-energy.
- Prediction target: sum of the five installed-capacity fields.
- Modelling rows: 601 after creating previous-year features.
- Training period: records through 2021 (499 rows).
- Test period: 2022–2024 (102 rows).
- Model: Gradient Boosting Regressor predicting year-over-year capacity change.

**Unit caution:** the supplied CSV does not document the capacity unit or provenance. Values are kept in the source unit and are not labelled MW or GW.

## Project chapters

1. **Introduction:** background, problem statement, objectives and practical relevance.
2. **Preprocessing and EDA:** data validation, target creation, lag features, cleaning and visual exploration.
3. **Git and GitHub:** repository structure, version-control commands and collaboration workflow.
4. **Model development:** chronological train/test split, preprocessing pipeline, Gradient Boosting and feature importance.
5. **Results and conclusion:** MAE, RMSE, R², WAPE, baseline comparison, limitations and future work.

## Run in Google Colab

1. Open the Colab notebook using the link above.
2. Run the package-install cell.
3. Upload `renewable_energy_dataset.csv` when prompted.
4. Run all cells from top to bottom.
5. The notebook generates the preprocessed dataset, model, metrics, predictions and EDA plots.

## Run locally

```bash
python -m pip install -r requirements.txt
python renewable_energy_model.py
```

## Preprocessing

1. Check required columns, data types, missing values, duplicates, duplicate State–Year keys and negative numeric values.
2. Calculate `Total_Renewable_Capacity` as the sum of the five technology columns.
3. Sort by state and year and create previous-year capacity features.
4. Remove each state's first observed row from modelling because no prior-year capacity is available.
5. One-hot encode `State` and median-impute numeric inputs within a scikit-learn pipeline.
6. Retain plausible high-capacity states rather than automatically deleting them as outliers.
7. Exclude current-year technology capacities from predictors to avoid target leakage.

## Model and evaluation

The Gradient Boosting model learns the year-over-year capacity change. Its predicted change is added to the previous year's actual total to estimate the current year's total. Data is split chronologically, not randomly.

On the held-out 2022–2024 records, the reported Gradient Boosting metrics are:

- MAE: 621.81
- RMSE: 1,911.59
- R²: 0.937
- WAPE: 14.59%

The persistence baseline (previous year's capacity carried forward) has lower MAE and WAPE, while Gradient Boosting improves RMSE and R². The model therefore does not outperform the baseline on every metric.

## Main files

- `Renewable_Energy_Sources_in_India_Colab.ipynb` — runnable notebook with explanations.
- `renewable_energy_model.py` — standalone Python pipeline.
- `renewable_energy_dataset.csv` — original dataset.
- `renewable_energy_preprocessed.csv` — modelling table with lag features.
- `renewable_energy_backtest_predictions.csv` — held-out predictions.
- `renewable_energy_model_metrics.csv` — model and baseline metrics.
- `renewable_energy_feature_importance.csv` — permutation feature importance.
- `Renewable_Energy_Sources_in_India_Project_Report.pdf` — illustrated 12-page report.
- `plots/` — EDA and model diagnostic charts.
- `requirements.txt` — Python dependencies.

## Git workflow

Clone the repository and commit future changes:

```bash
git clone https://github.com/athmikathmik8-svg/Renewable-energy-resoruces-in-India.git
cd Renewable-energy-resoruces-in-India
git status
git add .
git commit -m "Update renewable energy analysis"
git push
```

## Practical relevance and limitations

State-level growth estimates can support preliminary discussion of grid expansion, substations, transmission capacity and storage planning. They are not engineering recommendations. Actual planning needs verified demand, generation, utility, network, land, cost and grid-connection information. Multi-year forecasts also need future population/GSDP inputs and a validated recursive forecast design.
