def fetch_relation_data(client):
    """
    Fetch relation graph with semantic types
    """

    query = f"""
    WITH semantic_map AS (
        SELECT DISTINCT
            cui,
            semantic_type
        FROM `{SEMANTIC_TABLE}`
    )

    SELECT
        r.origin_cui,
        r.origin_cui_name,
        r.target_cui,
        r.target_cui_name,
        r.relation,

        s1.semantic_type AS origin_semantic_type,
        s2.semantic_type AS target_semantic_type

    FROM `{RELATION_TABLE}` r

    LEFT JOIN semantic_map s1
        ON r.origin_cui = s1.cui

    LEFT JOIN semantic_map s2
        ON r.target_cui = s2.cui
    """

    return client.query(query).to_dataframe()
def load_dataframe_to_bigquery(client, dataframe, table_name):
    """
    Load pandas dataframe into BigQuery
    """

    job = client.load_table_from_dataframe(
        dataframe,
        table_name
    )

    job.result()

    print(f"Loaded {len(dataframe)} rows into {table_name}")
def generate_cluster_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def build_clusters(df):
    """
    Logic:

    1. Every semantic type becomes a SUPER cluster
    2. Inside semantic type cluster:
         - if relation differs => create subcluster
         - if target semantic type differs => create cross-semantic cluster
    3. Preserve hierarchy through parent_cluster_id
    """

    clusters = []
    members = []
    mappings = []

    created_super_clusters = {}
    created_relation_clusters = {}
    created_cross_clusters = {}

    now = datetime.utcnow()

    for _, row in df.iterrows():

        origin_cui = row["origin_cui"]
        origin_name = row["origin_cui_name"]

        target_cui = row["target_cui"]
        target_name = row["target_cui_name"]

        relation = row["relation"]

        origin_sem = row["origin_semantic_type"]
        target_sem = row["target_semantic_type"]

        # =========================================================
        # 1. SUPER CLUSTER : semantic type
        # =========================================================

        if origin_sem not in created_super_clusters:

            super_cluster_id = generate_cluster_id("SUPER")

            created_super_clusters[origin_sem] = super_cluster_id

            clusters.append({
                "cluster_id": super_cluster_id,
                "cluster_name": f"SemanticType_{origin_sem}",
                "cluster_type": "SUPER_SET",
                "semantic_type": origin_sem,
                "relation": None,
                "parent_cluster_id": None,
                "created_at": now
            })

        super_cluster_id = created_super_clusters[origin_sem]

        # =========================================================
        # 2. SAME SEMANTIC TYPE + SAME RELATION
        # =========================================================

        if origin_sem == target_sem:

            relation_key = (origin_sem, relation)

            if relation_key not in created_relation_clusters:

                relation_cluster_id = generate_cluster_id("REL")

                created_relation_clusters[relation_key] = relation_cluster_id

                clusters.append({
                    "cluster_id": relation_cluster_id,
                    "cluster_name": f"{origin_sem}_{relation}",
                    "cluster_type": "RELATION_CLUSTER",
                    "semantic_type": origin_sem,
                    "relation": relation,
                    "parent_cluster_id": super_cluster_id,
                    "created_at": now
                })

                mappings.append({
                    "source_cluster_id": super_cluster_id,
                    "target_cluster_id": relation_cluster_id,
                    "mapping_type": "PARENT_CHILD",
                    "created_at": now
                })

            relation_cluster_id = created_relation_clusters[relation_key]

            # add origin member
            members.append({
                "cluster_id": relation_cluster_id,
                "cui": origin_cui,
                "cui_name": origin_name,
                "semantic_type": origin_sem,
                "role": "ORIGIN",
                "created_at": now
            })

            # add target member
            members.append({
                "cluster_id": relation_cluster_id,
                "cui": target_cui,
                "cui_name": target_name,
                "semantic_type": target_sem,
                "role": "TARGET",
                "created_at": now
            })

        # =========================================================
        # 3. CROSS SEMANTIC TYPE CLUSTER
        # =========================================================

        else:

            cross_key = (origin_sem, target_sem, relation)

            if cross_key not in created_cross_clusters:

                cross_cluster_id = generate_cluster_id("CROSS")

                created_cross_clusters[cross_key] = cross_cluster_id

                clusters.append({
                    "cluster_id": cross_cluster_id,
                    "cluster_name": f"{origin_sem}_TO_{target_sem}_{relation}",
                    "cluster_type": "CROSS_SEMANTIC_CLUSTER",
                    "semantic_type": origin_sem,
                    "relation": relation,
                    "parent_cluster_id": super_cluster_id,
                    "created_at": now
                })

                mappings.append({
                    "source_cluster_id": super_cluster_id,
                    "target_cluster_id": cross_cluster_id,
                    "mapping_type": "PARENT_CHILD",
                    "created_at": now
                })

            cross_cluster_id = created_cross_clusters[cross_key]

            members.append({
                "cluster_id": cross_cluster_id,
                "cui": origin_cui,
                "cui_name": origin_name,
                "semantic_type": origin_sem,
                "role": "ORIGIN",
                "created_at": now
            })

            members.append({
                "cluster_id": cross_cluster_id,
                "cui": target_cui,
                "cui_name": target_name,
                "semantic_type": target_sem,
                "role": "TARGET",
                "created_at": now
            })

    return clusters, members, mappings
def run_clustering():

    client = bigquery.Client(project=PROJECT_ID)


    print("Fetching relation graph...")
    df = fetch_relation_data(client)

    print(f"Fetched {len(df)} rows")

    print("Building clusters...")
    clusters, members, mappings = build_clusters(df)

    import pandas as pd

    cluster_df = pd.DataFrame(clusters).drop_duplicates()

    member_df = pd.DataFrame(members).drop_duplicates()

    map_df = pd.DataFrame(mappings).drop_duplicates()

    print("Loading cluster table...")
    load_dataframe_to_bigquery(
        client,
        cluster_df,
        CLUSTER_TABLE
    )

    print("Loading member table...")
    load_dataframe_to_bigquery(
        client,
        member_df,
        MEMBER_TABLE
    )

    print("Loading map table...")
    load_dataframe_to_bigquery(
        client,
        map_df,
        MAP_TABLE
    )

    print("Scenario 4 clustering completed")

    return {
        "clusters": len(cluster_df),
        "members": len(member_df),
        "mappings": len(map_df)
    }

# Execute
if __name__ == "__main__":

    result = run_clustering()

    print(result)
