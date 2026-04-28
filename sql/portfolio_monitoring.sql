-- Portfolio-level KPIs
SELECT
    COUNT(*) AS total_loans,
    SUM(exposure_at_default) AS total_exposure,
    AVG(probability_of_default) AS avg_probability_of_default,
    AVG(risk_flag) AS actual_default_rate,
    SUM(expected_loss) AS total_expected_loss
FROM scored_loans;


-- Risk by borrower purpose
SELECT
    purpose,
    COUNT(*) AS loans,
    SUM(exposure_at_default) AS total_exposure,
    AVG(probability_of_default) AS avg_probability_of_default,
    AVG(risk_flag) AS actual_default_rate,
    SUM(expected_loss) AS total_expected_loss
FROM scored_loans
GROUP BY purpose
ORDER BY total_expected_loss DESC, avg_probability_of_default DESC;


-- Risk by housing and duration profile
SELECT
    housing,
    duration_band,
    COUNT(*) AS loans,
    AVG(probability_of_default) AS avg_probability_of_default,
    AVG(risk_flag) AS actual_default_rate,
    SUM(expected_loss) AS total_expected_loss
FROM scored_loans
GROUP BY housing, duration_band
ORDER BY total_expected_loss DESC, avg_probability_of_default DESC;


-- Borrower segment drill-down
SELECT
    sex,
    job,
    purpose,
    COUNT(*) AS loans,
    AVG(credit_amount) AS avg_credit_amount,
    AVG(probability_of_default) AS avg_probability_of_default,
    SUM(expected_loss) AS total_expected_loss
FROM scored_loans
GROUP BY sex, job, purpose
ORDER BY total_expected_loss DESC;


-- High-risk accounts for case review
SELECT
    age,
    sex,
    housing,
    purpose,
    credit_amount,
    duration,
    probability_of_default,
    exposure_at_default,
    expected_loss,
    loan_to_income_ratio
FROM scored_loans
WHERE probability_of_default >= 0.60
ORDER BY expected_loss DESC, probability_of_default DESC;


-- Affordability view, populated only if income exists in the dataset
SELECT
    purpose,
    COUNT(*) AS loans_with_income,
    AVG(loan_to_income_ratio) AS avg_loan_to_income_ratio,
    AVG(probability_of_default) AS avg_probability_of_default,
    SUM(expected_loss) AS total_expected_loss
FROM scored_loans
WHERE loan_to_income_ratio IS NOT NULL
GROUP BY purpose
ORDER BY avg_loan_to_income_ratio DESC;

