-- Recommended staging table for loading scored borrower-level outputs.

CREATE TABLE scored_loans (
    age INTEGER,
    sex VARCHAR(20),
    job INTEGER,
    housing VARCHAR(30),
    saving_accounts VARCHAR(30),
    checking_account VARCHAR(30),
    credit_amount NUMERIC(12,2),
    duration INTEGER,
    purpose VARCHAR(50),
    risk VARCHAR(10),
    risk_flag INTEGER,
    age_band VARCHAR(20),
    duration_band VARCHAR(20),
    monthly_payment_proxy NUMERIC(12,2),
    monthly_income_normalized NUMERIC(12,2),
    loan_to_income_ratio NUMERIC(12,4),
    probability_of_default NUMERIC(12,6),
    loss_given_default NUMERIC(12,4),
    exposure_at_default NUMERIC(12,2),
    expected_loss NUMERIC(12,2),
    predicted_risk_flag INTEGER,
    predicted_risk_label VARCHAR(10)
);

