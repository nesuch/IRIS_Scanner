# IRIS Expansion Design — Multi-Source Regulatory Data Platform

Status: **Draft / direction** (not yet implemented). Author seed: design discussion, 2026-06.

Goal: grow IRIS from "reconciled Handbook viewer" into the canonical, validated,
**cross-source** metric store for IRDAI data — Handbook **+ public disclosures +
regulatory returns** — with a standard ingestion format, automated data-quality
checks, and cross-source reconciliation.

This document is the plan to get there without breaking what already works. It is
deliberately incremental: every layer ships value on its own.

---

## 0. Where we are today (the foundation already exists)

The hard part — a **canonical metric layer** — is already built and is the right
primitive for everything below. Do **not** rebuild it; extend it.

| Concern | Today | File |
|---|---|---|
| Canonical metric id + unit | `metric_id`, `unit_code`, `unit_scale`, `value_base` on `financial_metrics`; `metric_dim` | `tools/canonicalize_financials.py` |
| Canonical entity id | `insurer_id`, `insurer_dim` | same |
| Dimension typing | `lob_id/lob_type`, `cob_id/cob_type`, `lob_dim`, `cob_dim` | `tools/categorize_dimensions.py` |
| Format-adaptive ingest | `aggregate_submissions` (additive, key-deduped, curated-wins) | `iris_brain.py` |
| DQ checks (within-source) | sum-vs-total, within-series magnitude outliers | `tools/audit_financials.py`, `tools/flag_data_quality.py`, `data_quality_flags` |
| Prod data versioning | content-signature swap of handbook tables | `migrate_financial.py` |

**Key invariant to preserve:** anything analytical SUMs `value_base` (= `value ×
unit_scale`), never raw `value`. Every new source must populate `value_base`.

---

## 1. Metric Registry (promote `metric_dim` into a real dictionary)

`metric_dim` today is a thin id↔name map. Promote it to a curated registry that
**declares** each metric's contract, so ingestion and validation can be driven by
data instead of bespoke code.

```sql
CREATE TABLE metric_registry (
  metric_id        TEXT PRIMARY KEY,   -- stable slug, e.g. "gross_direct_premium"
  canonical_name   TEXT NOT NULL,      -- display label
  definition       TEXT,               -- one-line meaning / regulatory basis
  value_type       TEXT NOT NULL,      -- currency | count | percentage | ratio | days | text
  unit_code        TEXT,               -- INR | COUNT | PERCENT | USD | NUMBER ...
  unit_scale       REAL DEFAULT 1,     -- raw -> base multiplier (Lakh=1e5, Crore=1e7)
  additive         INTEGER DEFAULT 1,  -- can rows be summed? (percentages/ratios = 0)
  min_value        REAL,               -- inclusive sane floor (NULL = unbounded)
  max_value        REAL,               -- inclusive sane ceiling
  allow_negative   INTEGER DEFAULT 0,
  reported_in      TEXT,               -- JSON: ["handbook","bap_disclosure","form_L1",...]
  derived_formula  TEXT,               -- NULL if primary; else expression over metric_ids
  notes            TEXT
);
```

- Seed it from the existing `metric_dim` + observed `unit_code`/`value_base`.
- `value_type`/`min`/`max`/`allow_negative` power the DQ engine (§3).
- `reported_in` powers cross-source reconciliation (§4) — which sources *should*
  agree on this metric.
- `additive` already exists implicitly (`ADDITIVE` set in `iris_brain.py`); make it
  a column so the TOTAL-row logic and any roll-ups read it from the registry.

A parallel light registry for **entities** (`insurer_id` ↔ canonical name + aliases)
formalizes the alias map we hand-curated in `reingest_pivot_statements.py` so name
fragmentation can't recur across new sources.

---

## 2. Standard Ingestion Format + generic ingester

Today each Handbook table needs a bespoke `reingest_*` parser. New sources
(disclosures, returns) must **not** require new parsers. Define **one tidy upload
contract**; anything conforming ingests automatically.

### 2a. Canonical tidy schema (the "standard return/disclosure format")
One row per fact:

| column | required | notes |
|---|---|---|
| `source` | ✓ | e.g. `bap_disclosure_2024q4`, `form_L1_2024-25` |
| `insurer_id` | ✓ | resolved via entity registry (accept name + alias-map fallback) |
| `metric_id` | ✓ | must exist in `metric_registry` |
| `period` | ✓ | `2024-25` (FY) or `2024-Q3` or `2024-03-31` (as-on) |
| `value` | ✓ | raw number as reported |
| `unit` | ✓ | drives `unit_scale` lookup → `value_base` |
| `dimension` | | default `Insurer`; or `State`, `Financials`, … |
| `line_of_business`, `class_of_business` | | optional breakdowns |

### 2b. Two-stage ingest (staging → validated → live)
```
upload file ──► STAGING table (raw, untouched)
                    │  map insurer→insurer_id, metric→metric_id, unit→scale
                    │  compute value_base
                    ▼
              VALIDATION (§3)  ──► flags table (errors/warnings)
                    │  (block on hard errors, allow on warnings)
                    ▼
              MERGE into financial_metrics  (additive, key-dedup; curated wins)
```
- Reuse the additive/key-dedup logic already in the new `aggregate_submissions`.
- Keep a `source`/batch id so an ingest can be **reverted** cleanly.
- Unmapped `metric_id`/`insurer_id` → row goes to a **review queue**, not silently
  dropped (the lesson from the pivot-table name fragmentation).

---

## 3. Data-Quality (DQ) Validation Engine

Rules are **declarative**, read from `metric_registry`. Each produces a row in a
`dq_flags` table: `(source, insurer_id, metric_id, period, severity, rule, detail)`.

| Rule | Severity | Source of truth |
|---|---|---|
| missing/empty value where metric is required | error | registry |
| wrong type (text in a numeric metric) | error | `value_type` |
| negative where `allow_negative=0` | error | registry |
| percentage outside [0,100] (or ratio < 0) | error | `value_type`+bounds |
| value outside `[min_value, max_value]` | warning | registry |
| within-series magnitude outlier (z-score / IQR vs the metric's own history) | warning | extend `audit_financials.py` |
| sum(parts) ≠ declared total beyond tolerance | warning | extend `flag_data_quality.py` |
| duplicate key in upload | error | ingester |

Hard errors block the row from going live; warnings ingest but surface in the
admin **review queue** (reuse the existing flag/announcement UI patterns).

---

## 4. Cross-Source Reconciliation (the differentiator)

Because a metric carries one `metric_id` across Handbook, disclosures, and returns,
the same `(insurer_id, metric_id, period)` is directly comparable.

```
for each (insurer_id, metric_id, period) reported by ≥2 sources in registry.reported_in:
    compare value_base across sources
    if |max - min| / max > tolerance(metric):
        emit reconciliation flag (which sources, the values, the gap)
```

- Tolerance per metric (exact for counts; small % band for rounded ₹ figures).
- Output: a **reconciliation dashboard** — "for insurer X, metric Y, the disclosure
  says A, the return says B, the Handbook says C; they disagree by Z%."
- This is the regulator-grade feature an LLM app cannot do reliably, and it falls
  out almost for free once §1–§2 exist.

---

## 5. Scale: serving re-architecture (do this BEFORE 10×-ing the rows)

**Hard limit today:** `load_master_data_engine()` loads the entire table into an
in-memory pandas `UNIFIED_DF` at startup, and `get_filter_options()` builds *every*
distinct cascade combo. Memory & payload grow linearly with rows.

| Rows | In-memory pandas (current) |
|---|---|
| ~253k (today) | ~100 MB — fine |
| ~1–2M (5–10×) | OK on a bigger instance (4–8 GiB) |
| 12–25M (50–100×) | **5–13 GB RAM + minute-long cold starts + oversized filter payloads — won't work** |

SQLite on disk is *not* the bottleneck (it handles tens of millions of rows fine).
The "load everything into RAM" pattern is.

**Fix (low-friction, keeps single-binary simplicity): DuckDB query-on-demand.**
- Replace whole-table pandas load with DuckDB queries over the SQLite/Parquet data.
- `filter_data` / `get_filter_options` → SQL `GROUP BY`/`DISTINCT` with indexes;
  filter options lazy-loaded/paginated per cascade step instead of precomputed.
- `value_base` SUMs, TOTAL rows, statement builds → SQL aggregates.
- Keep the canonical model (`metric_id`, `value_base`, dim tables) **unchanged**.
- Next step only if multi-instance / concurrent writes needed: Postgres (Cloud SQL).

Also revisit at scale: Litestream restore time and image size (already managed via
`.dockerignore`/`.gcloudignore`); consider Parquet snapshots for the large
read-only fact data and keep SQLite for operational tables.

---

## 6. Suggested sequencing (each step independently shippable)

1. **Metric Registry** (§1) — promote `metric_dim`; seed from current data.
2. **Entity registry** — formalize the insurer alias map.
3. **Standard format + staging ingester** (§2) — retire bespoke parsers for *new* data.
4. **DQ engine** (§3) — declarative rules + review queue.
5. **Serving re-architecture to DuckDB** (§5) — before pushing row counts up.
6. **Cross-source reconciliation** (§4) — the headline feature.
7. Onboard disclosures, then returns, one source at a time.

Do **not** load 10M+ rows into the current pandas model first — re-architect (step 5)
before, or alongside, the first big source onboarding.
