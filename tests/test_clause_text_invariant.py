"""Regression guard for the precomputed-search-column invariant.

The search fast path relies on _CT_LC / _CT_LCJ always mirroring Clause_Text.
set_clause_text() is the single write path that must keep them in lock-step;
_add_search_cols() must build them with the same normalization. These tests fail
if either drifts (e.g. someone changes the normalization in one place only, or a
future edit writes Clause_Text without refreshing the derived columns).

Dependency-free: run with `python tests/test_clause_text_invariant.py`
(or via pytest, which will collect the test_* functions).
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import iris_brain as brain


def _fresh_df():
    return pd.DataFrame({
        "Clause_ID": ["C1", "C2"],
        "Source_Doc": ["Doc", "Doc"],
        "Clause_Text": ["Free-Look Period", "Grievance Redressal"],
    })


def test_derive_matches_add_search_cols():
    """Scalar helper and the bulk column builder must agree, cell for cell."""
    df = brain._add_search_cols(_fresh_df())
    for i, txt in enumerate(df["Clause_Text"]):
        lc, lcj = brain._derive_search_columns(txt)
        assert df[brain._CT_LC].iloc[i] == lc
        assert df[brain._CT_LCJ].iloc[i] == lcj


def test_set_clause_text_keeps_columns_in_sync():
    """Editing Clause_Text via the helper updates the derived columns too."""
    saved = brain.KB_CACHE_DF
    try:
        brain.KB_CACHE_DF = brain._add_search_cols(_fresh_df())
        mask = brain.KB_CACHE_DF["Clause_ID"] == "C1"
        brain.set_clause_text(mask, "New-Text Value")

        row = brain.KB_CACHE_DF[mask].iloc[0]
        assert row["Clause_Text"] == "New-Text Value"
        assert row[brain._CT_LC] == "new-text value"        # lower-cased
        assert row[brain._CT_LCJ] == "newtext value"        # hyphen removed
        # The untouched row must be unchanged.
        other = brain.KB_CACHE_DF[brain.KB_CACHE_DF["Clause_ID"] == "C2"].iloc[0]
        assert other[brain._CT_LC] == "grievance redressal"
    finally:
        brain.KB_CACHE_DF = saved


if __name__ == "__main__":
    test_derive_matches_add_search_cols()
    test_set_clause_text_keeps_columns_in_sync()
    print("OK: clause-text invariant holds (derived columns stay in sync).")
