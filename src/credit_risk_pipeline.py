from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "german_credit_data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

TARGET_COLUMN = "risk_flag"
LGD_ASSUMPTION = 0.45
RANDOM_STATE = 42


def ensure_directories() -> None:
    for path in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Download the Kaggle CSV and save it as data/raw/german_credit_data.csv."
        )
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        col.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        for col in df.columns
    ]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)

    removable_columns = {"unnamed:_0", "unnamed:_0.1", "unnamed:0"}
    drop_candidates = [col for col in df.columns if col in removable_columns]
    if drop_candidates:
        df = df.drop(columns=drop_candidates)

    expected_map = {
        "saving_accounts": "saving_accounts",
        "checking_account": "checking_account",
        "credit_amount": "credit_amount",
        "duration": "duration",
        "purpose": "purpose",
        "sex": "sex",
        "housing": "housing",
        "job": "job",
        "age": "age",
        "risk": "risk",
    }

    for current_name, target_name in expected_map.items():
        if current_name in df.columns and current_name != target_name:
            df = df.rename(columns={current_name: target_name})

    required_columns = {
        "age",
        "sex",
        "job",
        "housing",
        "saving_accounts",
        "checking_account",
        "credit_amount",
        "duration",
        "purpose",
        "risk",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing expected columns: " + ", ".join(missing_columns)
        )

    df["risk"] = df["risk"].astype(str).str.strip().str.lower()
    risk_map = {"good": 0, "bad": 1}
    unexpected_risk_values = sorted(set(df["risk"]) - set(risk_map))
    if unexpected_risk_values:
        raise ValueError(
            "Unexpected risk labels found: " + ", ".join(unexpected_risk_values)
        )
    df[TARGET_COLUMN] = df["risk"].map(risk_map)

    for column in ["saving_accounts", "checking_account", "purpose", "housing", "sex"]:
        df[column] = df[column].astype("string").fillna("unknown")

    df["job"] = pd.to_numeric(df["job"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["credit_amount"] = pd.to_numeric(df["credit_amount"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    df["duration_band"] = pd.cut(
        df["duration"],
        bins=[0, 12, 24, 36, 48, np.inf],
        labels=["0-12", "13-24", "25-36", "37-48", "49+"],
        include_lowest=True,
    ).astype("string")
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, np.inf],
        labels=["<=25", "26-35", "36-45", "46-55", "56+"],
        include_lowest=True,
    ).astype("string")
    df["monthly_payment_proxy"] = df["credit_amount"] / df["duration"].replace(0, np.nan)

    income_column = None
    for candidate in ("monthly_income", "annual_income", "income"):
        if candidate in df.columns:
            income_column = candidate
            break

    if income_column == "annual_income":
        df["monthly_income_normalized"] = pd.to_numeric(df[income_column], errors="coerce") / 12
    elif income_column:
        df["monthly_income_normalized"] = pd.to_numeric(df[income_column], errors="coerce")
    else:
        df["monthly_income_normalized"] = np.nan

    df["loan_to_income_ratio"] = df["credit_amount"] / (
        df["monthly_income_normalized"] * 12
    )

    return df


def build_model_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def add_risk_metrics(df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    scored = df.copy()
    scored["probability_of_default"] = probabilities
    scored["loss_given_default"] = LGD_ASSUMPTION
    scored["exposure_at_default"] = scored["credit_amount"]
    scored["expected_loss"] = (
        scored["probability_of_default"]
        * scored["loss_given_default"]
        * scored["exposure_at_default"]
    )
    scored["predicted_risk_flag"] = (scored["probability_of_default"] >= 0.5).astype(int)
    scored["predicted_risk_label"] = scored["predicted_risk_flag"].map({0: "good", 1: "bad"})
    return scored


def export_summaries(df: pd.DataFrame) -> None:
    segment_summary = (
        df.groupby(["housing", "purpose", "duration_band"], dropna=False)
        .agg(
            loans=("risk_flag", "size"),
            actual_default_rate=("risk_flag", "mean"),
            average_pd=("probability_of_default", "mean"),
            total_exposure=("exposure_at_default", "sum"),
            expected_loss=("expected_loss", "sum"),
            avg_credit_amount=("credit_amount", "mean"),
        )
        .reset_index()
        .sort_values(["expected_loss", "average_pd"], ascending=[False, False])
    )

    purpose_summary = (
        df.groupby("purpose", dropna=False)
        .agg(
            loans=("risk_flag", "size"),
            bad_loans=("risk_flag", "sum"),
            default_rate=("risk_flag", "mean"),
            average_pd=("probability_of_default", "mean"),
            expected_loss=("expected_loss", "sum"),
            total_exposure=("exposure_at_default", "sum"),
        )
        .reset_index()
        .sort_values("expected_loss", ascending=False)
    )

    segment_summary.to_csv(OUTPUTS_DIR / "segment_summary.csv", index=False)
    purpose_summary.to_csv(OUTPUTS_DIR / "purpose_summary.csv", index=False)


def export_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> None:
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    report = classification_report(y_true, y_pred, digits=4)

    model_summary = "\n".join(
        [
            "Credit Risk Model Performance",
            "============================",
            f"Accuracy: {accuracy:.4f}",
            f"ROC AUC: {roc_auc:.4f}",
            "",
            "Classification Report",
            "---------------------",
            report,
        ]
    )

    (OUTPUTS_DIR / "model_metrics.txt").write_text(model_summary, encoding="utf-8")


def main() -> None:
    ensure_directories()

    raw_df = load_data(RAW_DATA_PATH)
    clean_df = clean_data(raw_df)

    clean_df.to_csv(PROCESSED_DIR / "clean_credit_data.csv", index=False)

    feature_columns = [
        "age",
        "job",
        "credit_amount",
        "duration",
        "monthly_payment_proxy",
        "sex",
        "housing",
        "saving_accounts",
        "checking_account",
        "purpose",
        "age_band",
        "duration_band",
    ]

    if clean_df["monthly_income_normalized"].notna().any():
        feature_columns.extend(["monthly_income_normalized", "loan_to_income_ratio"])

    X = clean_df[feature_columns]
    y = clean_df[TARGET_COLUMN]

    numeric_features = [
        column
        for column in X.columns
        if pd.api.types.is_numeric_dtype(X[column])
    ]
    categorical_features = [column for column in X.columns if column not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_model_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    train_probabilities = pipeline.predict_proba(X_train)[:, 1]
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)

    train_scored = add_risk_metrics(clean_df.loc[X_train.index], train_probabilities)
    test_scored = add_risk_metrics(clean_df.loc[X_test.index], test_probabilities)
    combined_scored = pd.concat([train_scored, test_scored], axis=0).sort_index()

    train_scored.to_csv(PROCESSED_DIR / "train_scored.csv", index=False)
    test_scored.to_csv(PROCESSED_DIR / "test_scored.csv", index=False)

    export_summaries(combined_scored)
    export_metrics(y_test, test_predictions, test_probabilities)

    joblib.dump(pipeline, MODELS_DIR / "logistic_regression_credit_risk.joblib")

    print("Pipeline completed successfully.")
    print(f"Processed data saved to: {PROCESSED_DIR}")
    print(f"Model saved to: {MODELS_DIR}")
    print(f"Outputs saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()

