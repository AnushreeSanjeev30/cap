"""
fix_duplicate_rows.py
======================
One-off cleanup for a merge bug discovered in the Phase A ingestion stage
(not present in this repo -- see MODEL_TRUST_GUIDE.md / KALMAN_FILTER_RATIONALE.md
for context): two countries, NLD and VNM, ended up with multiple rows per
(iso3, year) in output/master_soft_power_panel.csv -- 4 rows/year for NLD,
2 rows/year for VNM -- with genuinely different indicator values, not exact
duplicates. Every other country has exactly one row per year.

This corrupts anything that consumes the panel as one-row-per-country-year:
most visibly, the Kalman filter (archive/kalman_softpower_complete.py) treats
each duplicate row as a separate time step, inflating apparent sample size
and producing a spurious trend slope for these two countries specifically.

Fix: for each (iso3, year) group with more than one row, keep the row with
the highest `outlier_feature_coverage` -- the fraction of source indicators
that were actually observed rather than imputed for that row. This is the
same "trust the row built from more real data" logic MODEL_TRUST_GUIDE.md
already asks readers to apply by hand; applying it here is a resolution of
an unrecoverable data conflict (the row values genuinely differ and there's
no way to know which merge input was "correct" without the original,
missing Phase A notebook), not a guess.

Run once from working/:
    .venv/Scripts/python.exe archive/fix_duplicate_rows.py
"""

import pandas as pd

PANEL_PATH = 'output/master_soft_power_panel.csv'
PHASE_B_PATH = 'output/master_phase_b.parquet'


def dedupe(df: pd.DataFrame, coverage_col: str, label: str) -> pd.DataFrame:
    sizes = df.groupby(['iso3', 'year']).size()
    dupes = sizes[sizes > 1]
    if len(dupes) == 0:
        print(f"  [{label}] No duplicate (iso3, year) rows found -- nothing to do.")
        return df

    affected = sorted(dupes.index.get_level_values('iso3').unique())
    print(f"  [{label}] {len(dupes)} duplicated (iso3, year) groups across "
          f"{len(affected)} countries: {affected}")

    if coverage_col in df.columns:
        df = (df.sort_values(coverage_col, ascending=False)
                .drop_duplicates(subset=['iso3', 'year'], keep='first')
                .sort_values(['iso3', 'year'])
                .reset_index(drop=True))
    else:
        # Phase B's column set doesn't carry outlier_feature_coverage --
        # just keep the first row per group for consistency with the panel.
        df = (df.drop_duplicates(subset=['iso3', 'year'], keep='first')
                .sort_values(['iso3', 'year'])
                .reset_index(drop=True))

    new_sizes = df.groupby(['iso3', 'year']).size()
    print(f"  [{label}] Rows removed: {sizes.sum() - new_sizes.sum()}. "
          f"Max rows per (iso3, year) is now {new_sizes.max()}.")
    return df


def main():
    print("Fixing duplicate (iso3, year) rows in the panel files...\n")

    panel = pd.read_csv(PANEL_PATH)
    panel = dedupe(panel, 'outlier_feature_coverage', 'master_soft_power_panel.csv')
    panel.to_csv(PANEL_PATH, index=False)
    print(f"  Wrote {PANEL_PATH} ({len(panel)} rows)\n")

    phase_b = pd.read_parquet(PHASE_B_PATH)
    phase_b = dedupe(phase_b, 'outlier_feature_coverage', 'master_phase_b.parquet')
    phase_b.to_parquet(PHASE_B_PATH, index=False)
    print(f"  Wrote {PHASE_B_PATH} ({len(phase_b)} rows)\n")

    print("Done. Files downstream of these two (Kalman outputs, trained models,")
    print("feature-importance/SHAP tables, dashboard artifacts) were built from")
    print("the corrupted versions and should be regenerated to pick up the fix.")


if __name__ == '__main__':
    main()
