# PRE-REGISTRATION — XAU/USD London-Open Breakout (Edge-PoC)

**Status:** FROZEN before any code. No criterion below may change after the first walk-forward run.
**Repo:** github.com/Chutipoon/Xau-usd
**Frozen:** 2026-06-18
**Lineage:** Successor probe to v1 (`v1.0-closed-negative-result`). v1 killed single-instrument *directional prediction* on D1/H1 (h≥1). This probe tests an **unexplored timeframe/structure**: intraday session-breakout at London open. It does NOT re-litigate v1's finding; it tests a class v1 never ran.

---

## 1. Hypothesis (H1)

> Price breaking the Asian-session range at London open carries directional information that produces positive expectancy **net of IUX cost** and **in excess of a same-bracket random-direction control** (i.e. the edge is the breakout *signal* — alpha — not merely being in a volatile session with favorable bracket geometry — beta).

Null (H0): the range break is noise; a random-direction entry with identical brackets performs as well.

---

## 2. Scope of THIS run (Pass 1)

- **Instrument:** XAU/USD only.
- **Timeframe:** H1 (`features_ohlcv.parquet`, look-ahead audit already passing). M5 is **out of scope** unless Pass 1 wins.
- **Sizing:** continuous fractional risk = **1.0% of equity per trade** (NOT 0.01-lot). The $100 / lot-granularity constraint is an *execution-PoC* concern, deliberately excluded here so edge is not conflated with granularity.
- **No COT / FRED / GDELT / HMM / LSTM / GARCH.** Pure OHLCV breakout. (If a later pass adds any joined feature, it MUST follow `SKILL_Lookahead-guard.md`.)

---

## 3. Strategy definition

- **Session timezone:** define windows in **Europe/London local time, DST-aware**, converted to UTC per-date (backtest spans 2015–2026 → many DST transitions; a fixed-clock window would drift vs. real open).
- **Asian range:** high / low of bars **strictly before** London open (declared window in §4). The breakout bar itself is **never** included in the range.
- **Entry:** the first H1 bar whose close is beyond the range → enter at that close in the break direction (long if above high, short if below low). One entry per side per day, max one position at a time.
- **Stop / target:** declared in §4, derived from the day's range width (so brackets are vol-scaled by construction).
- **Time stop:** flat at session end (declared in §4) if neither stop nor target hit.

---

## 4. Pre-declared parameter grid (report FULL grid, no cherry-pick)

| Param | Values to test |
|-------|----------------|
| Asian-range window (London local) | 00:00–07:00, 02:00–07:00 |
| London-open entry window | 07:00–10:00 |
| Breakout buffer | 0.0×, 0.10× range width |
| Stop | range-low/high (opposite side), 1.0× range width |
| Target | 1.0R, 2.0R, session-end |
| Session end (flat-by) | 16:00 London |

**Robustness rule:** WIN must hold for the **median configuration** of the grid, not only the best. Report a table of every config's OOS expectancy. A single passing config = overfit, treated as FAIL.

---

## 5. Cost model (IUX raw) + sensitivity sweep

Commission $7 / 1.0 lot round-turn = $0.07/oz. Spread 0.2 pip — **pip definition is a 10× ambiguity** ($0.10 vs $1.00 per pip); confirm from IUX spec/a live quote and parameterize.

| Scenario | Round-turn cost / oz | Use |
|----------|---------------------|-----|
| Low | $0.09 (pip=$0.10, no slip) | report only |
| **Mid** | **$0.15 (+ ~1 tick/side slip)** | **WIN gate evaluated here** |
| High | $0.27 (pip=$1.00 + slip) | report only |

Apply cost per trade as `cost_per_oz × oz_traded`. If WIN holds only at Low → flag as fragile (not a pass).

---

## 6. Baselines (mandatory — every number is excess-over-baseline)

1. **Buy-and-hold** XAU/USD over the same OOS span.
2. **Random-direction control (falsification gate):** for each *real* trade, re-use the **exact same entry day/time and the exact same stop & target distances**, flip direction to a 50/50 coin (Monte Carlo, **K=1000**). Build the null distribution of mean expectancy. `p = fraction of random runs with mean expectancy ≥ real`. This isolates whether *direction* is informative given identical, vol-matched bracket geometry.

*(Optional secondary, report-only: random entry-time within session — tests whether timing/selection adds info. Not a hard gate.)*

---

## 7. Look-ahead discipline

- Asian range uses **closed prior bars only**; breakout bar excluded.
- Any feature that could peek at its own period gets `.shift(1)`.
- Session/DST conversion computed from date, never from a fixed clock offset.
- Walk-forward: no information from a test window touches its own or earlier in-sample fit.

---

## 8. Walk-forward structure

- **12 windows** (consistent with v1). Params are **pre-declared (§4), not optimized in-sample** — grid is fixed; walk-forward measures OOS *stability*, not param search.
- Report per-window expectancy, trade count, win rate, and the full-grid table.

---

## 9. WIN condition (ALL four must hold; evaluate at Mid cost)

1. OOS mean expectancy / trade **> 0** net of cost.
2. OOS mean expectancy **> random-direction control, p < 0.05** (K=1000).
3. **≥ 9 / 12** OOS windows with positive expectancy.
4. **Beats buy-and-hold** on a risk-adjusted basis over the full OOS span.

*(Sharpe>0.97 from v1 deliberately NOT imported — different strategy/timeframe.)*

---

## 10. KILL condition (terminal — principle #6)

If any WIN gate fails: record to `NEGATIVE_RESULT_STUDY.md` / dead-ends with the raw numbers and **STOP**. Do **not** tune parameters to force a pass (that is the in-sample-artifact trap that sank the HMM). Proceed to Pass 2 (M5) **only** if Pass 1 wins.

---

## 11. Out of scope (scope fence)

No M5/tick build, no broker connection, no live/paper trading, no 0.01-lot simulation, no alternative-data features, no new ML models, no refactor of unrelated repo modules. Reuse the existing walk-forward harness, look-ahead guard, and falsification utilities — do not rebuild them.
