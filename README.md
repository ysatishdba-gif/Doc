"""
Build per-CUI topic lookup pickle from MRREL.

Reads:  BigQuery MRCONSO, MRSTY, MRREL
Writes: cui_topic_lookup.pkl

For each CUI, the pickle stores:
  - label, semantic_type
  - topics_by_relation: dict mapping effective_relation -> list of topic CUIs

"Effective relation" is RELA if present (more specific clinical relation),
otherwise REL (generic relation type). All rows in MRREL contribute.

Output pickle structure:
  {
    cui: {
      "label":         str,
      "semantic_type": str,
      "topics_by_relation": {
        "<relation>": [
          {"topic_cui": str, "topic_label": str, "topic_semantic_type": str},
          ...
        ],
        ...
      }
    }
  }

Adaptor usage:
  >>> import pickle
  >>> with open("cui_topic_lookup.pkl", "rb") as f:
  ...     lookup = pickle.load(f)
  >>> entry = lookup.get("C0231807")     # knee pain
  >>> entry["topics_by_relation"]["isa"] # CUIs knee pain isa
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
    """Pull source-target-relation rows joined with semantic types and
    preferred labels. One row per MRREL relationship.
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
            COALESCE(r.RELA, r.REL)      AS relation
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
    """Group rows by source_cui, then by relation, into the output dict."""
    print("  Building lookup...")
    t = time.time()
    lookup = {}

    rows = df.to_dict("records")
    n_rows = len(rows)
    seen_per_source = defaultdict(set)  # (source, relation, topic_cui) dedupe

    for i, row in enumerate(rows):
        if i % 500000 == 0 and i > 0:
            print(f"    processed {i:,}/{n_rows:,}", flush=True)

        source_cui    = row["source_cui"]
        source_label  = row["source_label"]
        source_sem    = row["source_semantic_type"]
        topic_cui     = row["topic_cui"]
        topic_label   = row["topic_label"]
        topic_sem     = row["topic_semantic_type"]
        relation      = row["relation"]

        if source_cui not in lookup:
            lookup[source_cui] = {
                "label":         source_label,
                "semantic_type": source_sem,
                "topics_by_relation": defaultdict(list),
            }

        # Dedupe (source, relation, topic_cui)
        dedupe_key = (relation, topic_cui)
        if dedupe_key in seen_per_source[source_cui]:
            continue
        seen_per_source[source_cui].add(dedupe_key)

        lookup[source_cui]["topics_by_relation"][relation].append({
            "topic_cui":           topic_cui,
            "topic_label":         topic_label,
            "topic_semantic_type": topic_sem,
        })

    # Convert defaultdicts to plain dicts for pickle cleanliness
    for entry in lookup.values():
        entry["topics_by_relation"] = dict(entry["topics_by_relation"])

    print(f"  -> {len(lookup):,} CUIs indexed ({time.time()-t:.1f}s)")

    # Stats
    if lookup:
        n_relations = sum(len(e["topics_by_relation"]) for e in lookup.values())
        n_topics = sum(
            sum(len(v) for v in e["topics_by_relation"].values())
            for e in lookup.values()
        )
        print(f"    Total relation buckets across CUIs: {n_relations:,}")
        print(f"    Total (source, relation, topic) triples: {n_topics:,}")

        rel_counts = defaultdict(int)
        for entry in lookup.values():
            for rel, topics in entry["topics_by_relation"].items():
                rel_counts[rel] += len(topics)
        top_relations = sorted(rel_counts.items(), key=lambda x: -x[1])[:15]
        print(f"\n    Top 15 relations by count:")
        for rel, n in top_relations:
            print(f"      {rel:35s}  {n:>10,}")

        # Per-CUI topic counts
        per_cui_counts = sorted(
            sum(len(v) for v in e["topics_by_relation"].values())
            for e in lookup.values()
        )
        n = len(per_cui_counts)
        print(f"\n    Per-CUI topic count: "
              f"min={per_cui_counts[0]}  median={per_cui_counts[n//2]}  "
              f"mean={sum(per_cui_counts)/n:.1f}  "
              f"p95={per_cui_counts[int(n*0.95)]}  max={per_cui_counts[-1]}")

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
