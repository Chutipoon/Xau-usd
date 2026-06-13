"""Canonical regime mapping — the ONLY source of truth.

The trained HMM state INDEX is not the blueprint order. Anything that turns a
regime into a number (GARCH lookup, position sizing, RegimeSignalBridge) must
go state -> label -> value, keyed by the LABEL string. Import these constants;
never copy the literals into another file. If the HMM is retrained and the index
order changes, only this file changes and every downstream lookup still resolves.
"""

# state index (as the trained hmm_model.pkl numbers them) -> semantic label
STATE_TO_LABEL = {
    0: "high_vol_choppy",
    1: "trending_bull",
    2: "low_vol_range",
    3: "trending_bear",
}

# label -> position-size multiplier (gold realized vol; R0 is gold-rally vol, not equity fear)
LABEL_TO_MULTIPLIER = {
    "high_vol_choppy": 0.3,
    "trending_bull": 1.0,
    "low_vol_range": 0.8,
    "trending_bear": 0.6,
}

LABEL_TO_STATE = {label: idx for idx, label in STATE_TO_LABEL.items()}

# fail fast if the mapping is ever edited inconsistently
assert LABEL_TO_MULTIPLIER[STATE_TO_LABEL[0]] == 0.3
assert LABEL_TO_MULTIPLIER[STATE_TO_LABEL[1]] == 1.0
assert LABEL_TO_MULTIPLIER[STATE_TO_LABEL[2]] == 0.8
assert LABEL_TO_MULTIPLIER[STATE_TO_LABEL[3]] == 0.6
