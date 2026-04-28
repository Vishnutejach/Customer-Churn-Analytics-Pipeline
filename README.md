# Customer Churn & Retention Analytics

End-to-end machine learning repository for customer churn prediction, customer risk segmentation, and retention strategy analysis using the `Customer-Churn-Records.csv` dataset.

## Repository Structure

```text
.
|-- data/
|   |-- raw/
|   |   `-- Customer-Churn-Records.csv
|   `-- processed/
|-- docs/
|-- models/
|-- outputs/
|   |-- dashboards/
|   |   |-- dashboard1_eda.png
|   |   |-- dashboard2_model_perf.png
|   |   |-- dashboard3_features_risk.png
|   |   `-- dashboard4_retention_strategy.png
|   |-- reports/
|   `-- summary.json
|-- src/
|   `-- churn_pipeline.py
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Included Files

- `src/churn_pipeline.py`: main analytics and modeling pipeline
- `data/raw/Customer-Churn-Records.csv`: source dataset
- `outputs/summary.json`: model and business summary snapshot
- `outputs/dashboards/`: exported dashboard images for EDA, model performance, feature importance, and retention strategy

## What The Pipeline Does

- loads and cleans the churn dataset
- engineers predictive features and risk indicators
- trains multiple classifiers:
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Deep Neural Network (`MLPClassifier`)
- compares models with ROC-AUC, average precision, and accuracy
- segments customers into churn risk tiers
- generates dashboard-ready PNG outputs
- writes a summary JSON file

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the pipeline:

```bash
python src/churn_pipeline.py
```

## Outputs

Running the pipeline writes artifacts to:

- `outputs/dashboards/dashboard1_eda.png`
- `outputs/dashboards/dashboard2_model_perf.png`
- `outputs/dashboards/dashboard3_features_risk.png`
- `outputs/dashboards/dashboard4_retention_strategy.png`
- `outputs/summary.json`

## Current Snapshot

Based on the included `summary.json`:

- Best model: `Gradient Boosting`
- Dataset rows: `10000`
- Churn rate: `20.38%`
- High-risk customers in test segmentation: `406`
- Estimated total net ROI: `$1,750,690`

## Notes

- The checked-in dashboard images are the provided project outputs.
- The pipeline has been adapted to use repository-relative paths so it can run locally from this repo.
"# Customer-Churn-Analytics-Pipeline" 
"# Customer-Churn-Analytics-Pipeline" 
