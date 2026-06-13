"""Fit GARCH(1,1) per HMM regime → models/garch_models.pkl

Runs off features_daily_d1.parquet alone; needs nothing from the H1/GDELT
backfill. One GARCH per regime, keyed by LABEL (not state index), with a
persistence check and a 1-day 99% VaR per regime.

LOOK-AHEAD NOTE (read before trusting this in a backtest)
---------------------------------------------------------
This script *characterizes* each regime's vol process in-sample. The regime
label per day comes from a full-sample HMM decode (smoothed/Viterbi), which is
fine for estimating per-regime parameters — it is NOT used to emit a tradeable
point-in-time signal here. Two things must stay true downstream or you leak:
  1. LIVE / walk-forward: assign the *current* regime from the FILTERED
     posterior P(s_t | obs_1..t) (forward-only), then look up the matching
     pre-fit GARCH model. Never assign the live regime with Viterbi/smoothed.
  2. Inside each walk-forward train window, refit on that window's data only —
     never on the full sample. This global fit is the characterization, not the
     backtest estimator.

KNOWN APPROXIMATION
-------------------
A regime's days are non-contiguous (16 switches over ~6.4y). We pool each
regime's returns and fit one GARCH; the variance recursion therefore crosses
segment boundaries. Standard for "per-regime GARCH"; a fully correct treatment
is Markov-switching GARCH (out of scope). Noted so it isn't a silent assumption.

Usage:
    py train_garch.py \
        --features data/features_daily_d1.parquet \
        --hmm models/hmm_model.pkl \
        --out models/garch_models.pkl
"""

from __future__ import annotations

import argparse
import math
import pickle
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning, DataScaleWarning
from scipy.stats import t as student_t

# single source of truth — adjust the import path to wherever this lives in your repo
from regime_labels import STATE_TO_LABEL, LABEL_TO_MULTIPLIER

# Returns are modeled in PERCENT (×100). arch's optimizer misbehaves on raw ~0.01
# returns (DataScaleWarning); ×100 puts gold daily returns near unit scale.
SCALE = 100.0

# Candidate column names — override on the CLI if yours differ.
PRICE_COL_CANDIDATES = ["close", "Close", "adj_close", "Adj Close", "price", "px_close"]
RETURN_COL_CANDIDATES = ["log_return", "logret", "log_ret", "ret", "return", "returns"]
REGIME_COL_CANDIDATES = ["state", "regime", "hmm_state", "regime_state", "regime_idx"]

# Only used if no persisted regime column exists and we must decode from the HMM.
# MUST match the columns AND order (and any scaling) the HMM was trained on.
HMM_FEATURE_COLS: list[str] | None = None  # e.g. ["log_return", "DFII10", "DTWEXBGS", "VIXCLS"]


def load_returns(df: pd.DataFrame, price_col: str | None) -> pd.Series:
    """Return daily log returns in PERCENT, indexed like df, NaNs dropped.

    Prefers computing from a price column (unit-controlled). Falls back to an
    existing return column with a unit guess if no price is available.
    """
    if price_col:
        if price_col not in df.columns:
            raise KeyError(f"--price-col '{price_col}' not in parquet columns: {list(df.columns)}")
        px = df[price_col].astype(float)
        ret = np.log(px / px.shift(1)) * SCALE
        return ret.dropna()

    for c in PRICE_COL_CANDIDATES:
        if c in df.columns:
            px = df[c].astype(float)
            return (np.log(px / px.shift(1)) * SCALE).dropna()

    for c in RETURN_COL_CANDIDATES:
        if c in df.columns:
            r = df[c].astype(float).dropna()
            # guess units: if it looks like a fraction (~0.01), scale to percent
            if r.abs().median() < 0.5:
                r = r * SCALE
                print(f"[returns] using existing column '{c}', scaled ×{SCALE:g} to percent")
            else:
                print(f"[returns] using existing column '{c}', assumed already in percent")
            return r

    raise KeyError(
        "No price or return column found. Pass --price-col, or rename a column to one of "
        f"{PRICE_COL_CANDIDATES} / {RETURN_COL_CANDIDATES}."
    )


def get_regime_states(df: pd.DataFrame, hmm_path: str, regime_col: str | None) -> pd.Series:
    """Integer HMM state per row, aligned to df.index.

    Priority: (1) a persisted regime column (assumed written by the HMM training
    run) — trusted as-is; (2) decode from the saved HMM over HMM_FEATURE_COLS.
    """
    # explicit override
    if regime_col:
        if regime_col not in df.columns:
            raise KeyError(f"--regime-col '{regime_col}' not in parquet columns: {list(df.columns)}")
        return df[regime_col].astype(int)

    # auto-detect a persisted column
    for c in REGIME_COL_CANDIDATES:
        if c in df.columns:
            print(f"[regime] using persisted column '{c}' from parquet")
            return df[c].astype(int)

    # fall back to decoding from the HMM
    if HMM_FEATURE_COLS is None:
        raise RuntimeError(
            "No regime column in the parquet and HMM_FEATURE_COLS is unset.\n"
            "Either: (a) have the HMM training step write decoded states back to the parquet "
            "(simplest, no drift), or (b) set HMM_FEATURE_COLS to the EXACT columns+order the "
            "HMM was trained on. NOTE: decoding here must reproduce training preprocessing "
            "(same scaler), or the state numbering will not line up with STATE_TO_LABEL."
        )

    print(f"[regime] decoding from {hmm_path} over {HMM_FEATURE_COLS}")
    print("[regime] WARNING: ensure these columns + scaling match HMM training exactly.")
    with open(hmm_path, "rb") as fh:
        model = pickle.load(fh)
    missing = [c for c in HMM_FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"HMM feature columns missing from parquet: {missing}")
    X = df[HMM_FEATURE_COLS].astype(float).to_numpy()
    states = model.predict(X)  # full-sample (Viterbi) decode — see LOOK-AHEAD NOTE
    return pd.Series(states, index=df.index, dtype=int)


def std_q01(dist: str, nu: float | None) -> float:
    """1% quantile of the UNIT-VARIANCE standardized innovation (negative number)."""
    if dist == "normal" or nu is None:
        return -2.3263478740408408  # Phi^{-1}(0.01)
    if nu <= 2.0:  # variance only finite for nu>2; guard degenerate fits
        nu = 2.05
    # standardize Student-t to unit variance: multiply raw quantile by sqrt((nu-2)/nu)
    return float(student_t.ppf(0.01, nu) * math.sqrt((nu - 2.0) / nu))


def fit_one_regime(ret_pct: pd.Series, dist: str, max_persistence: float) -> dict:
    """Fit GARCH(1,1) on one regime's pooled returns (percent). Returns a compact,
    pickle-portable dict (plain floats only — no arch objects inside)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", DataScaleWarning)
        am = arch_model(ret_pct, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
        res = am.fit(disp="off")

    p = res.params
    mu = float(p["mu"])
    omega = float(p["omega"])
    alpha = float(p["alpha[1]"])
    beta = float(p["beta[1]"])
    nu = float(p["nu"]) if "nu" in p.index else None
    persistence = alpha + beta

    # seeds for the live one-step recursion (percent² units): next-step variance
    # is omega + alpha*last_resid^2 + beta*last_var — store both so live code can
    # roll it forward without re-reading history.
    last_var = float(res.conditional_volatility.iloc[-1] ** 2)
    last_resid2 = float(res.resid.iloc[-1] ** 2)
    fc_var = omega + alpha * last_resid2 + beta * last_var
    sigma = math.sqrt(fc_var)

    q01 = std_q01(dist, nu)
    var99_ret = mu + sigma * q01  # signed 1-day 1% return, percent (negative)

    uncond_vol = math.sqrt(omega / (1.0 - persistence)) if persistence < 1.0 else None

    flags = []
    if res.convergence_flag != 0:
        flags.append("not_converged")
    if persistence >= max_persistence:
        flags.append(f"high_persistence(>={max_persistence})")  # near-IGARCH, unstable uncond var

    return {
        "fitted": True,
        "dist": dist,
        "scale": SCALE,  # params/vol are in percent units
        "params": {"mu": mu, "omega": omega, "alpha[1]": alpha, "beta[1]": beta, "nu": nu},
        "persistence": persistence,
        "uncond_vol_pct": uncond_vol,
        "next_sigma_pct": sigma,             # one-step-ahead conditional vol, percent
        "last_var_pct2": last_var,           # seed for live recursion
        "last_resid2_pct2": last_resid2,     # seed for live recursion
        "var99_1d_return_pct": var99_ret,    # signed (negative)
        "var99_1d_loss_pct": -var99_ret,     # positive magnitude
        "loglik": float(res.loglikelihood),
        "aic": float(res.aic),
        "convergence_flag": int(res.convergence_flag),
        "flags": flags,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit GARCH(1,1) per HMM regime.")
    ap.add_argument("--features", default="data/features_daily_d1.parquet")
    ap.add_argument("--hmm", default="models/hmm_model.pkl")
    ap.add_argument("--out", default="models/garch_models.pkl")
    ap.add_argument("--price-col", default=None, help="close-price column (auto-detected if omitted)")
    ap.add_argument("--regime-col", default=None, help="persisted HMM state column (auto-detected if omitted)")
    ap.add_argument("--dist", default="t", choices=["t", "normal"], help="innovation distribution")
    ap.add_argument("--min-obs", type=int, default=250, help="skip a regime below this many days")
    ap.add_argument("--max-persistence", type=float, default=0.999, help="flag alpha+beta at/above this")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    if not isinstance(df.index, pd.DatetimeIndex):
        # try to find a date-ish column to index by; else keep positional order
        for c in ("date", "Date", "timestamp", "ts"):
            if c in df.columns:
                df = df.set_index(pd.to_datetime(df[c])).sort_index()
                break
    print(f"[load] {args.features}: {df.shape[0]} rows, {df.shape[1]} cols")

    ret = load_returns(df, args.price_col)
    states = get_regime_states(df, args.hmm, args.regime_col)

    data = pd.DataFrame({"ret": ret, "state": states}).dropna()
    print(f"[align] {len(data)} aligned (return, state) rows")
    counts = data["state"].value_counts().sort_index()
    print("[regime counts] " + ", ".join(
        f"{idx}:{STATE_TO_LABEL.get(idx, '?')}={n}" for idx, n in counts.items()))

    models: dict = {
        "_meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "features_path": args.features,
            "hmm_path": args.hmm,
            "dist": args.dist,
            "scale": SCALE,
            "units": "params/vol in PERCENT; variance in PERCENT^2. Divide vol by 100 for fractional.",
            "state_to_label": dict(STATE_TO_LABEL),
            "label_to_multiplier": dict(LABEL_TO_MULTIPLIER),
            "keyed_by": "label",
            "decode": "full-sample (smoothed/Viterbi) — live regime must use FILTERED posterior",
        }
    }

    for state_idx, label in STATE_TO_LABEL.items():
        seg = data.loc[data["state"] == state_idx, "ret"]
        n = len(seg)
        if n < args.min_obs:
            # DO NOT fall back to the full series — that silently collapses 4 regimes
            # into 1 (a past bug). Flag and skip instead.
            print(f"[{label}] SKIP — only {n} obs (< --min-obs {args.min_obs})")
            models[label] = {
                "fitted": False,
                "reason": "insufficient_obs",
                "n_obs": n,
                "state_index": state_idx,
                "multiplier": LABEL_TO_MULTIPLIER[label],
            }
            continue

        info = fit_one_regime(seg, args.dist, args.max_persistence)
        info.update(label=label, state_index=state_idx,
                    multiplier=LABEL_TO_MULTIPLIER[label], n_obs=n)
        models[label] = info
        flag_str = (" [" + ",".join(info["flags"]) + "]") if info["flags"] else ""
        print(f"[{label}] n={n}  ω={info['params']['omega']:.4f}  "
              f"α={info['params']['alpha[1]']:.3f}  β={info['params']['beta[1]']:.3f}  "
              f"persist={info['persistence']:.3f}  VaR99(1d)={info['var99_1d_loss_pct']:.2f}%{flag_str}")

    with open(args.out, "wb") as fh:
        pickle.dump(models, fh)
    print(f"[save] {args.out}  ({sum(1 for k,v in models.items() if k!='_meta' and v.get('fitted'))}/4 regimes fitted)")


if __name__ == "__main__":
    main()
