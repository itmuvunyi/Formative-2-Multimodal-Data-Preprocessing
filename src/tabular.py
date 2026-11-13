"""Task 1: clean the two tables, merge them, and engineer leakage-safe features.

Grain of the output: one row per transaction.
Target: product_category.
Rule for features: only what is known *before* the purchase happens. The amount
and rating of the current transaction are recorded after the purchase, so they
are kept for reference but never fed to the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PROCESSED, TABULAR

TARGET = "product_category"
PLATFORMS = ["Facebook", "Instagram", "LinkedIn", "TikTok", "Twitter"]
SENTIMENT = {"Negative": -1, "Neutral": 0, "Positive": 1}

NUMERIC_FEATURES = [
    "engagement_mean", "engagement_max", "interest_mean", "interest_max",
    "sentiment_mean", "social_records", "platforms_used",
    *[f"share_{p.lower()}" for p in PLATFORMS],
    "purchase_month", "purchase_dayofweek",
    "prior_txn_count", "prior_avg_spend", "prior_avg_rating", "days_since_prior",
]
CATEGORICAL_FEATURES = ["primary_platform", "prior_category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(TABULAR / "customer_transactions.csv"),
        pd.read_csv(TABULAR / "customer_social_profiles.csv"),
    )


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    log = {"rows_in": len(out), "exact_duplicates": int(out.duplicated().sum()),
           "null_ratings": int(out["customer_rating"].isna().sum())}
    out = out.drop_duplicates()
    out["purchase_date"] = pd.to_datetime(out["purchase_date"], errors="raise")
    out["customer_id"] = out["customer_id_legacy"].astype(int)
    # Missing rating = customer did not rate. Keep the flag; the median is only
    # used when the rating history is averaged in prior_avg_rating.
    out["rating_missing"] = out["customer_rating"].isna().astype(int)
    out["customer_rating_filled"] = out["customer_rating"].fillna(out["customer_rating"].median())
    log["rows_out"] = len(out)
    return out.sort_values(["purchase_date", "transaction_id"]).reset_index(drop=True), log


def clean_social(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    log = {"rows_in": len(out), "exact_duplicates": int(out.duplicated().sum()),
           "nulls": int(out.isna().sum().sum())}
    out = out.drop_duplicates()
    # 'A178' in this table and 178 in the transactions are the same customer.
    out["customer_id"] = out["customer_id_new"].str.extract(r"(\d+)", expand=False).astype(int)
    out["sentiment_score"] = out["review_sentiment"].map(SENTIMENT).astype(int)
    log["rows_out"] = len(out)
    log["customers"] = int(out["customer_id"].nunique())
    log["customers_with_multiple_rows"] = int((out.groupby("customer_id").size() > 1).sum())
    return out.reset_index(drop=True), log


def aggregate_social(social: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per customer so the join cannot duplicate transactions."""
    grouped = social.groupby("customer_id")
    agg = grouped.agg(
        engagement_mean=("engagement_score", "mean"),
        engagement_max=("engagement_score", "max"),
        interest_mean=("purchase_interest_score", "mean"),
        interest_max=("purchase_interest_score", "max"),
        sentiment_mean=("sentiment_score", "mean"),
        social_records=("sentiment_score", "size"),
        platforms_used=("social_media_platform", "nunique"),
    )
    shares = pd.crosstab(social["customer_id"], social["social_media_platform"], normalize="index")
    shares = shares.reindex(columns=PLATFORMS, fill_value=0.0)
    shares.columns = [f"share_{c.lower()}" for c in shares.columns]
    # Ties break alphabetically so the result is reproducible.
    primary = (
        social.groupby(["customer_id", "social_media_platform"]).size().rename("n").reset_index()
        .sort_values(["customer_id", "n", "social_media_platform"], ascending=[True, False, True])
        .drop_duplicates("customer_id").set_index("customer_id")["social_media_platform"]
        .rename("primary_platform")
    )
    return agg.join(shares).join(primary).reset_index()


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Customer history built only from earlier transactions (shift before aggregating)."""
    out = df.sort_values(["customer_id", "purchase_date", "transaction_id"]).copy()
    g = out.groupby("customer_id", sort=False)
    out["prior_txn_count"] = g.cumcount()
    out["prior_avg_spend"] = g["purchase_amount"].transform(lambda s: s.shift().expanding().mean())
    out["prior_avg_rating"] = g["customer_rating_filled"].transform(lambda s: s.shift().expanding().mean())
    out["days_since_prior"] = g["purchase_date"].diff().dt.days
    out["prior_category"] = g[TARGET].shift().astype(object).where(lambda s: s.notna(), "NoHistory")
    out["purchase_month"] = out["purchase_date"].dt.month
    out["purchase_dayofweek"] = out["purchase_date"].dt.dayofweek
    return out.sort_values(["purchase_date", "transaction_id"]).reset_index(drop=True)


def merge_tables(transactions: pd.DataFrame, customer_social: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Inner join, validated many-to-one: every merged row must keep its own transaction."""
    merged = transactions.merge(customer_social, on="customer_id", how="inner", validate="many_to_one")
    checks = {
        "transactions_before": len(transactions),
        "customers_in_transactions": int(transactions["customer_id"].nunique()),
        "customers_in_social": int(customer_social["customer_id"].nunique()),
        "customers_matched": int(merged["customer_id"].nunique()),
        "rows_after_merge": len(merged),
        "transactions_dropped_no_social": len(transactions) - len(merged),
        "duplicate_transaction_ids": int(merged["transaction_id"].duplicated().sum()),
        "null_join_keys": int(merged["customer_id"].isna().sum()),
    }
    if checks["duplicate_transaction_ids"] or checks["rows_after_merge"] > checks["transactions_before"]:
        raise ValueError(f"Merge fanned out: {checks}")
    return merged, checks


def build_dataset(save: bool = True) -> tuple[pd.DataFrame, dict]:
    tx_raw, social_raw = load_raw()
    tx, tx_log = clean_transactions(tx_raw)
    social, social_log = clean_social(social_raw)
    tx = add_history_features(tx)
    merged, merge_log = merge_tables(tx, aggregate_social(social))
    merged = merged.drop(columns=["customer_id_legacy"]).sort_values("transaction_id").reset_index(drop=True)
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        merged.to_csv(PROCESSED / "merged_dataset.csv", index=False)
    return merged, {"transactions": tx_log, "social": social_log, "merge": merge_log}


def latest_customer_features(merged: pd.DataFrame, customer_id: int, when: pd.Timestamp | None = None) -> pd.DataFrame:
    """Feature row describing a customer's *next* visit, using all history so far."""
    rows = merged[merged["customer_id"] == customer_id].sort_values("purchase_date")
    if rows.empty:
        raise KeyError(f"Customer {customer_id} is not in the merged dataset")
    last = rows.iloc[-1]
    when = when or merged["purchase_date"].max() + pd.Timedelta(days=1)
    row = last.to_dict()
    row.update(
        prior_txn_count=len(rows),
        prior_avg_spend=float(rows["purchase_amount"].mean()),
        prior_avg_rating=float(rows["customer_rating_filled"].mean()),
        days_since_prior=float((when - last["purchase_date"]).days),
        prior_category=last[TARGET],
        purchase_month=when.month,
        purchase_dayofweek=when.dayofweek,
    )
    return pd.DataFrame([row])[FEATURES]
