"""
Build per-CUI topic lookup pickle from MRREL.

Reads:  BigQuery MRCONSO, MRSTY, MRREL
Writes: cui_topic_lookup.pkl

For each CUI, the pickle stores:
  - label, semantic_type
  - topic_cuis: deduped list of all CUIs this CUI relates to via MRREL.
    Each entry includes the relation metadata (REL, RELA) for reference,
    but the adaptor only needs the cui field.

All MRREL rows contribute regardless of REL or RELA value.

Output pickle structure:
  {
    cui: {
      "label":         str,
      "semantic_type": str,
      "topic_cuis": [
        {
          "cui":           str,
          "label":         str,
          "semantic_type": str,
          "rel":           str,         # MRREL REL value (PAR/CHD/RB/RN/RO/SY/SIB/...)
          "rela":          str | None,  # MRREL RELA value (isa, has_finding_site, ...) if present
        },
        ...
      ]
    }
  }

Adaptor usage (ignores rel/rela):
  >>> import pickle
  >>> with open("cui_topic_lookup.pkl", "rb") as f:
  ...     lookup = pickle.load(f)
  >>> entry = lookup.get("C0231807")          # knee pain
  >>> [t["cui"] for t in entry["topic_cuis"]] # all topic CUIs
"""

import os
import pickle
import sys
import time
from collections import defaultdict
from google.cloud import bigquery

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


PROJECT = "your-gcp-project-id"
DATASET = "your-bigquery-dataset"
OUTPUT  = "cui_topic_lookup.pkl"


def fetch_topic_data(client):
    """Pull source-target rows joined with semantic types and labels.
    REL and RELA are kept as metadata per row.
    """
    sql = f"""
        WITH semantic_map AS (
            SELECT DISTINCT
                CUI,
                LOWER(STY) AS semantic_type
            FROM `{PROJECT}.{DATASET}.MRSTY`
        ),
        concept_label AS (
            SELECT
                CUI,
                ANY_VALUE(STR) AS label
            FROM `{PROJECT}.{DATASET}.MRCONSO`
            WHERE SAB='SNOMEDCT_US'
              AND LAT='ENG'
              AND ISPREF='Y'
              AND SUPPRESS='N'
            GROUP BY CUI
        )
        SELECT
            r.CUI1                       AS source_cui,
            l1.label                     AS source_label,
            COALESCE(s1.semantic_type,'unknown') AS source_semantic_type,
            r.CUI2                       AS topic_cui,
            l2.label                     AS topic_label,
            COALESCE(s2.semantic_type,'unknown') AS topic_semantic_type,
            r.REL                        AS rel,
            r.RELA                       AS rela
        FROM `{PROJECT}.{DATASET}.MRREL` r
        JOIN concept_label l1 ON r.CUI1 = l1.CUI
        JOIN concept_label l2 ON r.CUI2 = l2.CUI
        LEFT JOIN semantic_map s1 ON r.CUI1 = s1.CUI
        LEFT JOIN semantic_map s2 ON r.CUI2 = s2.CUI
        WHERE r.SAB='SNOMEDCT_US'
          AND r.SUPPRESS='N'
    """
    print("  Querying topic data...")
    t = time.time()
    df = client.query(sql).to_dataframe()
    print(f"  -> {len(df):,} rows ({time.time()-t:.1f}s)")
    return df


def build_lookup(df):
    """Group rows by source_cui. Each topic entry records the topic CUI
    plus the REL and RELA from MRREL.

    Dedup key is (topic_cui, rel, rela): if the same source -> target
    pair appears via two different relations, both are kept as separate
    entries. If the same exact (rel, rela) row repeats, only one is kept.
    """
    print("  Building lookup...")
    t = time.time()
    lookup = {}

    rows = df.to_dict("records")
    n_rows = len(rows)
    seen = defaultdict(set)  # source_cui -> set of (topic_cui, rel, rela)

    import math

    for i, row in enumerate(rows):
        if i % 500000 == 0 and i > 0:
            print(f"    processed {i:,}/{n_rows:,}", flush=True)

        source_cui = row["source_cui"]
        topic_cui  = row["topic_cui"]
        rel        = row["rel"]
        rela       = row["rela"]

        # pandas converts NULL -> NaN; normalize to None for clean pickle
        if rela is None or (isinstance(rela, float) and math.isnan(rela)):
            rela = None

        if source_cui not in lookup:
            lookup[source_cui] = {
                "label":         row["source_label"],
                "semantic_type": row["source_semantic_type"],
                "topic_cuis":    [],
            }

        dedup_key = (topic_cui, rel, rela)
        if dedup_key in seen[source_cui]:
            continue
        seen[source_cui].add(dedup_key)

        lookup[source_cui]["topic_cuis"].append({
            "cui":           topic_cui,
            "label":         row["topic_label"],
            "semantic_type": row["topic_semantic_type"],
            "rel":           rel,
            "rela":          rela,
        })

    print(f"  -> {len(lookup):,} CUIs indexed ({time.time()-t:.1f}s)")

    if lookup:
        per_cui_counts = sorted(len(e["topic_cuis"]) for e in lookup.values())
        n = len(per_cui_counts)
        n_total_topics = sum(per_cui_counts)
        print(f"    Total (source, topic, rel, rela) entries: {n_total_topics:,}")
        print(f"    Per-CUI topic count: "
              f"min={per_cui_counts[0]}  median={per_cui_counts[n//2]}  "
              f"mean={sum(per_cui_counts)/n:.1f}  "
              f"p95={per_cui_counts[int(n*0.95)]}  max={per_cui_counts[-1]}")

        # Distinct topic CUIs per source (ignoring rel/rela duplicates)
        per_cui_distinct = sorted(
            len({t["cui"] for t in e["topic_cuis"]}) for e in lookup.values()
        )
        print(f"    Per-CUI DISTINCT topic CUI count: "
              f"min={per_cui_distinct[0]}  median={per_cui_distinct[n//2]}  "
              f"mean={sum(per_cui_distinct)/n:.1f}  "
              f"p95={per_cui_distinct[int(n*0.95)]}  max={per_cui_distinct[-1]}")

    return lookup


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = round(os.path.getsize(path) / 1e6, 1)
    print(f"  Saved {len(obj):,} entries -> {path}  ({size_mb} MB)")


if __name__ == "__main__":
    start = time.time()

    print(f"\n{'='*60}")
    print(f"  Topic lookup builder")
    print(f"  Project : {PROJECT}")
    print(f"  Dataset : {DATASET}")
    print(f"  Output  : {OUTPUT}")
    print(f"{'='*60}\n")

    client = bigquery.Client(project=PROJECT)

    print("[1/3] Querying BigQuery...")
    df = fetch_topic_data(client)
    print()

    print("[2/3] Building lookup...")
    lookup = build_lookup(df)
    print()

    print("[3/3] Saving pickle...")
    save_pickle(lookup, OUTPUT)
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")
