from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_audit_list(
    anomaly_summary: pd.DataFrame,
    building_meta: pd.DataFrame,
    min_hours: int = 100,
    top_n: int | None = None,
) -> pd.DataFrame:
    filtered = anomaly_summary[anomaly_summary["total_hours"] >= min_hours].copy()

    audit = filtered.merge(
        building_meta[["building_id", "site_id", "primary_use", "square_feet"]],
        on="building_id",
        how="left",
    )

    audit["excess_per_sqft"] = np.where(
        audit["square_feet"] > 0,
        audit["total_excess"] / audit["square_feet"],
        0.0,
    ).astype(np.float32)

    score_weights = {
        "anomaly_rate": 0.4,
        "max_anomaly_score": 0.3,
        "excess_per_sqft": 0.3,
    }

    for col in score_weights:
        col_max = audit[col].max()
        if col_max > 0:
            audit[f"{col}_norm"] = audit[col] / col_max
        else:
            audit[f"{col}_norm"] = 0.0

    audit["priority_score"] = sum(
        audit[f"{col}_norm"] * w for col, w in score_weights.items()
    )

    audit = audit.sort_values("priority_score", ascending=False).reset_index(drop=True)
    audit["rank"] = range(1, len(audit) + 1)

    if top_n:
        audit = audit.head(top_n)

    output_cols = [
        "rank", "building_id", "meter", "site_id", "primary_use",
        "square_feet", "anomaly_rate", "anomaly_hours", "total_hours",
        "mean_anomaly_score", "max_anomaly_score",
        "total_excess", "excess_per_sqft", "priority_score",
    ]
    return audit[[c for c in output_cols if c in audit.columns]]


def export_audit_list(audit: pd.DataFrame, output_path: str | Path, fmt: str = "csv") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        audit.to_json(output_path, orient="records", indent=2)
    else:
        audit.to_csv(output_path, index=False)

    logger.info("Audit list exported to %s (%d entries)", output_path, len(audit))
    return output_path
