
"""MVP reservoir screening from the sampled seismic attribute CSV."""

import numpy as np
import pandas as pd

REQUIRED = (
    "Inline", "Crossline", "TWT",
    "TotalSandProb", "PHIT_pred",
    "mu_rho", "VpVs")


def screen_reservoirs(
    df,
    sand_threshold=0.65,
    phit_threshold=0.10,
    top_n=100):
    """Return ranked reservoir candidates and summary."""

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    cols = list(REQUIRED) + [
        c for c in ("P_Impedance",
            "S_Impedance",
            "lambda_rho")
        if c in df.columns]

    samples = df[cols].copy()

    for col in cols:
        samples[col] = pd.to_numeric(
            samples[col], errors="coerce")

    valid = np.isfinite(samples[list(REQUIRED)].to_numpy(dtype=float)).all(axis=1)

    valid &= samples["TotalSandProb"].between(0, 1)
    valid &= samples["PHIT_pred"].between(0, 1)
    valid &= samples["mu_rho"] > 0
    valid &= samples["VpVs"] > 0

    if {"P_Impedance", "S_Impedance"}.issubset(samples.columns):
        valid &= ((samples["P_Impedance"] > 0) &
            (samples["S_Impedance"] > 0))

    mask = (valid &
        (samples["TotalSandProb"] >= sand_threshold) &
        (samples["PHIT_pred"] >= phit_threshold))

    candidates = samples.loc[mask].copy()

    if not candidates.empty:

        # Lower Mu-Rho = higher score
        candidates["mu_rho_score"] = (
            1.0 -
            candidates["mu_rho"].rank(
                pct=True,
                method="average"
            ))

        # Lower Vp/Vs = higher score
        candidates["vpvs_score"] = (
            1.0 -
            candidates["VpVs"].rank(pct=True,method="average"))

        # Combined reservoir screening score
        candidates["screening_score"] = (
            0.40 * candidates["TotalSandProb"] +
            0.30 * candidates["PHIT_pred"] +
            0.15 * candidates["mu_rho_score"] +
            0.15 * candidates["vpvs_score"])

    candidates = (
        candidates.sort_values(
            "screening_score",
            ascending=False)
        .head(top_n)
        .reset_index(drop=True))

    candidates.insert(
        0, "rank",
        np.arange(1, len(candidates) + 1))

    summary = {
        "input_rows": len(df),
        "valid_rows": int(valid.sum()),
        "qualifying_rows": int(mask.sum()),
        "returned_rows": len(candidates)}

    return candidates, summary
