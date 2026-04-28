# Tableau Dashboard Specification

Use `outputs/segment_summary.csv`, `outputs/purpose_summary.csv`, and the borrower-level scored dataset as Tableau inputs.

## Dashboard Title

`Credit Risk Analysis and Default Prediction`

## Recommended KPI Tiles

- Total Loans
- Total Exposure
- Average PD
- Actual Default Rate
- Total Expected Loss

## Recommended Visuals

### 1. Portfolio Risk by Purpose

- Chart: horizontal bar chart
- Dimension: `purpose`
- Measure: `SUM(expected_loss)`
- Tooltip: loan count, average PD, actual default rate

### 2. Housing vs Duration Heatmap

- Rows: `housing`
- Columns: `duration_band`
- Color: `AVG(probability_of_default)`
- Label: `SUM(expected_loss)`

### 3. Borrower Segment Risk

- Chart: stacked or grouped bar chart
- Dimensions: `sex`, `job`
- Measures: `AVG(probability_of_default)` or `COUNT(loans)`

### 4. High-Risk Borrower Table

- Columns: age, sex, housing, purpose, credit amount, duration, PD, expected loss
- Sort: expected loss descending
- Filter: PD threshold parameter

### 5. Exposure vs Default Scatter Plot

- X-axis: `credit_amount`
- Y-axis: `probability_of_default`
- Size: `expected_loss`
- Color: `predicted_risk_label`

## Filters

- Sex
- Housing
- Purpose
- Duration band
- Predicted risk label

## Dashboard Narrative

This dashboard should help answer:

- Which borrower segments drive the highest expected loss?
- Which loan purposes have the highest default concentration?
- Do higher loan amounts or longer durations correspond to higher modeled risk?
- Which individual loans should be prioritized for manual review?

## Tableau Caption for Portfolio Use

Interactive dashboard for monitoring borrower-level default risk, segment exposure, and expected loss across demographic and loan-purpose profiles.

