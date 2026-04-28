# Excel Review Layout

Use `data/processed/test_scored.csv` or the combined scored file loaded into your SQL layer.

## Recommended Sheets

### 1. `Raw_Data`

- Import the scored CSV as a table.
- Preserve field names exactly for pivot compatibility.

### 2. `KPI_Summary`

Create card-style metrics for:

- Total Loans
- Total Exposure
- Actual Default Rate
- Average Probability of Default
- Total Expected Loss
- Average Credit Amount

### 3. `Segment_Pivots`

Suggested pivots:

- Rows: `Purpose` | Values: count of loans, avg PD, sum of expected loss
- Rows: `Housing`, `Duration band` | Values: avg PD, actual default rate, sum of expected loss
- Rows: `Sex`, `Job` | Values: count of loans, avg credit amount, avg PD

### 4. `High_Risk_Cases`

Apply filters for:

- `probability_of_default >= 0.60`
- sort by `expected_loss` descending

Include:

- borrower segment fields
- credit amount
- duration
- predicted risk label
- expected loss

## Excel Charts

- Clustered column chart: expected loss by purpose
- Heatmap-style conditional formatting: housing x duration band
- Bar chart: average PD by job category
- Slicer-enabled pivot dashboard for `Sex`, `Housing`, `Purpose`, and `Predicted risk label`

## Portfolio Storyline

Use Excel to validate:

- which borrower segments carry the highest expected loss
- whether longer-duration loans show elevated default risk
- whether specific purposes or housing profiles concentrate bad loans

