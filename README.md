"""
Stage 2 of 2: assign topics and sub-topics from MRREL data.

Reads:  cui_expansion_lookup.pkl   (built by run_builder.py)
Writes: cui_topic_lookup.pkl       (same data + topic_id + subtopics)

Topic = connected component of REL-relations within the same semantic type.
        CUIs are in the same topic if they're connected via PAR/CHD/RB/RN/SY/SIB
        edges (transitively).

Sub-topic = (RELA, target_cui) grouping within a topic. Each CUI gets one
            sub-topic ID per RELA-relation it has, so a CUI can belong to
            multiple sub-topics simultaneously.

This stage uses no clustering algorithm — both topic and sub-topic
assignments are direct lookups from MRREL data.

Topic ID format:
    "<semantic_type>::<int>"
    "<semantic_type>::singleton::<cui>"

Sub-topic ID format:
    "<topic_id>::<rela>::<target_cui>"
"""

import os
import pickle
import sys
import time
from collections import defaultdict

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


INPUT  = "cui_expansion_lookup.pkl"
OUTPUT = "cui_topic_lookup.pkl"


# ─────────────────────────────────────────────────────────────
# TOPIC: connected components on REL graph
# ─────────────────────────────────────────────────────────────

def assign_topics(lookup):
    """For each semantic type, build a graph from CUIs' expansion lists
    (which are REL-filtered, same-type) and find connected components.
    Each component is a topic.
    """
    print("  Assigning topics via connected components...")
    t = time.time()

    # Bucket CUIs by semantic type
    by_type = defaultdict(set)
    for cui, entry in lookup.items():
        by_type[entry["semantic_type"]].add(cui)

    topic_assignments = {}
    component_sizes = []
    n_types = len(by_type)

    for type_idx, (sem_type, cuis_in_type) in enumerate(by_type.items(), 1):
        n_cuis = len(cuis_in_type)
        if type_idx % 20 == 0 or type_idx == 1 or type_idx == n_types:
            print(f"    [{type_idx}/{n_types}] semantic_type={sem_type!r}  "
                  f"CUIs={n_cuis:,}", flush=True)

        # Union-Find for connected components
        parent = {c: c for c in cuis_in_type}

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for cui in cuis_in_type:
            for e in lookup[cui]["expansion"]:
                other = e["cui"]
                if other in parent:
                    union(cui, other)

        # Group by root
        components = defaultdict(list)
        for cui in cuis_in_type:
            components[find(cui)].append(cui)

        for comp_idx, members in enumerate(components.values()):
            component_sizes.append(len(members))
            if len(members) == 1:
                topic_assignments[members[0]] = (
                    f"{sem_type}::singleton::{members[0]}"
                )
            else:
                tid = f"{sem_type}::{comp_idx}"
                for cui in members:
                    topic_assignments[cui] = tid

    for cui, entry in lookup.items():
        entry["topic_id"] = topic_assignments.get(
            cui, f"{entry['semantic_type']}::unassigned"
        )

    distinct = set(topic_assignments.values())
    singletons = sum(1 for x in distinct if "::singleton::" in x)
    multi = len(distinct) - singletons

    print(f"  -> done ({time.time()-t:.1f}s)")
    print(f"    Topics: {len(distinct):,} distinct  "
          f"({multi:,} multi-CUI, {singletons:,} singletons)")
    if component_sizes:
        sizes = sorted(component_sizes)
        n = len(sizes)
        print(f"    Component size: min={sizes[0]:,}  median={sizes[n//2]:,}  "
              f"mean={sum(sizes)/n:.0f}  p95={sizes[int(n*0.95)]:,}  "
              f"max={sizes[-1]:,}")

    return lookup


# ─────────────────────────────────────────────────────────────
# SUB-TOPIC: (RELA, target_cui) groupings
# ─────────────────────────────────────────────────────────────

def assign_subtopics(lookup):
    """For each CUI, build sub-topic IDs from its rela_neighbors list.
    Each (RELA, target_cui) pair becomes one sub-topic.

    Sub-topic ID is scoped under the CUI's main topic so the same RELA
    target produces a different sub-topic ID for CUIs in different topics.
    """
    print("  Assigning sub-topics from RELA neighbors...")
    t = time.time()

    n_subtopics_total = 0
    cuis_with_subtopics = 0

    for cui, entry in lookup.items():
        topic_id = entry["topic_id"]
        rela_list = entry.get("rela_neighbors", [])
        subtopics = []
        seen = set()  # dedupe (rela, target_cui) pairs in case of duplicates
        for r in rela_list:
            key = (r["rela"], r["target_cui"])
            if key in seen:
                continue
            seen.add(key)
            subtopic_id = f"{topic_id}::{r['rela']}::{r['target_cui']}"
            subtopics.append({
                "rela":         r["rela"],
                "target_cui":   r["target_cui"],
                "target_label": r["target_label"],
                "subtopic_id":  subtopic_id,
            })
        entry["subtopics"] = subtopics
        n_subtopics_total += len(subtopics)
        if subtopics:
            cuis_with_subtopics += 1

    print(f"  -> done ({time.time()-t:.1f}s)")
    print(f"    Sub-topics: {n_subtopics_total:,} total  "
          f"({cuis_with_subtopics:,} CUIs have at least one sub-topic, "
          f"{len(lookup) - cuis_with_subtopics:,} have none)")

    # Distribution stats
    subtopic_counts = [len(e["subtopics"]) for e in lookup.values()]
    if subtopic_counts:
        sorted_c = sorted(subtopic_counts)
        n = len(sorted_c)
        print(f"    Per-CUI sub-topic count: min={sorted_c[0]}  "
              f"median={sorted_c[n//2]}  mean={sum(sorted_c)/n:.1f}  "
              f"p95={sorted_c[int(n*0.95)]}  max={sorted_c[-1]}")

    # Most common RELA values
    rela_counter = defaultdict(int)
    for entry in lookup.values():
        for s in entry["subtopics"]:
            rela_counter[s["rela"]] += 1
    if rela_counter:
        top_relas = sorted(rela_counter.items(), key=lambda x: -x[1])[:15]
        print(f"    Top 15 RELA values:")
        for rela, n in top_relas:
            print(f"      {rela:35s}  {n:>10,}")

    return lookup


# ─────────────────────────────────────────────────────────────
# IO + MAIN
# ─────────────────────────────────────────────────────────────

def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run run_builder.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = round(os.path.getsize(path) / 1e6, 1)
    print(f"  Saved {len(obj):,} entries -> {path}  ({size_mb} MB)")


if __name__ == "__main__":
    start = time.time()

    print(f"\n{'='*60}")
    print(f"  Stage 2: topic (REL components) + sub-topic (RELA)")
    print(f"  Input  : {INPUT}")
    print(f"  Output : {OUTPUT}")
    print(f"{'='*60}\n")

    print("[1/4] Loading expansion pickle...")
    t = time.time()
    lookup = load_pickle(INPUT)
    print(f"  -> {len(lookup):,} entries  ({time.time()-t:.1f}s)")
    print()

    print("[2/4] Assigning topics...")
    lookup = assign_topics(lookup)
    print()

    print("[3/4] Assigning sub-topics...")
    lookup = assign_subtopics(lookup)
    print()

    print("[4/4] Saving pickle...")
    save_pickle(lookup, OUTPUT)
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")
