# Day 7 - Databricks

## What Is It?

Databricks is a leading data and AI platform built around the lakehouse architecture - a design that combines the flexibility and scale of a data lake with the structure and governance of a data warehouse. It is where many data engineering and machine learning teams already store, process and serve their data, using Apache Spark for large-scale computation, Delta Lake for reliable storage and MLflow for experiment tracking.

Databricks SQL adds a familiar SQL interface on top of the lakehouse, letting analysts and data engineers query Delta tables without writing Spark code. The `VECTOR_COSINE_SIMILARITY` function extends this to vector similarity search, making it straightforward to add semantic search to data that already lives in the lakehouse - without introducing a separate vector database.

In 2025 Databricks acquired Neon, a serverless Postgres startup. The acquisition signals Databricks' intent to extend the lakehouse to cover the full spectrum of database workloads, including the transactional and vector search use cases that Neon's pgvector support provides. For teams already working in Databricks, this reinforces the "stay in the lakehouse" story that this chapter demonstrates.

## When Would You Reach for It?

The clearest signal is existing data and workflows in Databricks. If your data engineering pipelines, ML experiments and analytical queries already run in Databricks, adding semantic search to a Delta table is a natural extension of what you already do. There is no new system to learn, no data to move and no synchronization to maintain.

The second signal is a SQL and PySpark team. Data engineers and ML practitioners who work in Databricks already know Delta Lake, Databricks SQL and the Unity Catalog. `VECTOR_COSINE_SIMILARITY` slots into this existing toolchain as a function call in a familiar SQL query.

The third signal is analytics alongside search. Like Snowflake, Databricks is built for analytical workloads. If your use case requires not just "find the most relevant document" but also "how many documents exist per category" and "which authors have contributed the most", both questions can be answered in the same system with SQL queries.

## The Use Case

For this chapter we'll build a RAG retrieval layer over internal technical documents - architecture notes, runbooks, API guides, security policies and onboarding materials. Users ask questions in natural language - "how do I restart a service during an incident" or "what are the security requirements for storing sensitive data" - and the system finds the most relevant documents from the corpus.

This is a natural fit for Databricks. Internal technical documentation is exactly the kind of content that data and engineering teams generate, store and share within a data platform. Keeping it searchable in the same system where teams already work reduces friction and keeps governance centralized.

## The Data

We'll generate synthetic internal technical wiki documents from pools of categories, authors and content templates. Each document has a title, category, author and a prose content field - the content is what we'll embed and search over.

The eight categories represent typical internal documentation types:

- `Architecture` - system design documents and technical overviews
- `Runbooks` - step-by-step operational procedures for incidents
- `API` - API reference guides and integration documentation
- `Security` - access control, data classification and compliance policies
- `Onboarding` - guides for new team members joining the organization
- `Data Engineering` - pipeline documentation, data models and ETL guides
- `ML Platform` - model deployment, feature store and experiment tracking guides
- `Incident Reports` - post-incident reviews and root cause analyses

The dataset scales via `NUM_DOCS`. Each document is generated from category-specific templates filled with randomized but realistic-sounding values. All author names are fictional.

## Building the Application

### Prerequisites

To follow along you'll need:

- A Databricks community edition account with access to a SQL warehouse
- Ollama running locally with the `all-minilm` model pulled
- Python 3.12 with a virtual environment

### Generate a Personal Access Token

1. In the Databricks UI, click your username in the top bar and select **Settings**
2. Click **User > Developer**
3. Next to **Access tokens**, click **Manage**
4. Click **Generate new token**, give it a name, select **BI Tools** as the scope type and click **Generate**
5. Copy the token immediately - it is only shown once

### Find Your SQL Warehouse Connection Details

1. In the Databricks UI, go to **SQL > SQL Warehouses**
2. Click your warehouse and select **Connection details**
3. Note down the **Server hostname** and **HTTP path**

### Set Environment Variables

```bash
export DATABRICKS_SERVER_HOSTNAME="dbc-xxxx.cloud.databricks.com"
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/xxxx"
export DATABRICKS_TOKEN="your-personal-access-token"
```

### Configuration

```python
DATABRICKS_SERVER_HOSTNAME = os.environ["DATABRICKS_SERVER_HOSTNAME"]
DATABRICKS_HTTP_PATH       = os.environ["DATABRICKS_HTTP_PATH"]
DATABRICKS_TOKEN           = os.environ["DATABRICKS_TOKEN"]
CATALOG_NAME               = "workspace"
SCHEMA_NAME                = "wiki_docs"
TABLE_NAME                 = "documents"
LLM_EMBEDDING              = "all-minilm"
NUM_DOCS                   = 200
RANDOM_SEED                = 42
```

> **Note:** Databricks Community Edition does not have a `main` catalog. The default catalog is `workspace`. If you see a `NO_SUCH_CATALOG_EXCEPTION` error, check which catalogs are available with `SHOW CATALOGS` and update `CATALOG_NAME` accordingly.

### Connect to Databricks

```python
conn = sql.connect(
    server_hostname = DATABRICKS_SERVER_HOSTNAME,
    http_path       = DATABRICKS_HTTP_PATH,
    access_token    = DATABRICKS_TOKEN
)

cursor = conn.cursor()
print("Connected to Databricks.")
```

### Generate the Dataset

Documents are assembled from category-specific content templates filled with randomized values from component pools. Each category has its own set of templates so the content is relevant to the category:

```python
CONTENT_TEMPLATES = {
    "Architecture": [
        "This document describes the high-level architecture of the {system} platform. "
        "The system is built on a microservices model with {pattern} as the primary "
        "communication pattern. Services are deployed on {infra} and managed via {tool}. "
        "Key design decisions include the use of {decision} to ensure scalability and fault tolerance.",
        # ... further templates
    ],
    "Runbooks": [
        "This runbook covers the procedure for {task} in the {system} environment. "
        "Follow these steps in order during an incident. First, verify the {check} is "
        "responding correctly. If not, escalate to the {team} team and open a P{priority} "
        "incident ticket. Roll back using {rollback} if the issue persists after {timeout} minutes.",
        # ... further templates
    ],
    # ... further categories
}
```

### Generate Embeddings

We'll embed the content of each document locally using Ollama. The embedding is stored as a string and cast to `ARRAY<FLOAT>` at query time:

```python
embeddings = []
for doc in tqdm(documents, desc = "Generating embeddings"):
    embeddings.append(get_embedding(doc["CONTENT"]))

df["EMBEDDING"] = [str(e) for e in embeddings]
```

### Load Data into Databricks

We'll create a Delta table and load all documents in a single bulk `INSERT`. The embedding is stored as a `STRING` column rather than a native vector type, for reasons explained below.

> **Note:** Inserting one row per document over a network connection to Databricks is extremely slow - each `cursor.execute()` call is a separate round trip. Concatenating all rows into a single `VALUES` clause and sending one SQL statement is significantly faster:

```python
values = []
for i, doc in enumerate(documents):
    embedding_str = json.dumps(embeddings[i])
    title    = doc["TITLE"].replace("'", "\\'")
    category = doc["CATEGORY"].replace("'", "\\'")
    author   = doc["AUTHOR"].replace("'", "\\'")
    content  = doc["CONTENT"].replace("'", "\\'")
    emb      = embedding_str.replace("'", "\\'")
    values.append(
        f"('{doc['DOC_ID']}', '{title}', '{category}', '{author}', '{content}', '{emb}')"
    )

cursor.execute(f"""
    INSERT INTO {CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}
    (DOC_ID, TITLE, CATEGORY, AUTHOR, CONTENT, EMBEDDING)
    VALUES {', '.join(values)}
""")
```

Note that single quotes in text fields must be escaped before interpolation into the SQL string.

### Semantic Search

We'll embed the query with Ollama and pass it to Databricks SQL as a parameter. `VECTOR_COSINE_SIMILARITY` computes cosine similarity between the stored embedding and the query embedding.

> **Note:** The Databricks SQL connector does not support inserting data directly into a `VECTOR` typed column, so we'll store embeddings as `STRING`. At query time, `FROM_JSON` parses the string into an `ARRAY<FLOAT>` which `VECTOR_COSINE_SIMILARITY` can operate on. Both the stored embedding and the query embedding need this cast:

```python
def search_docs(query: str, top_k: int = 5):
    query_embedding = get_embedding(query)
    query_json      = json.dumps(query_embedding)

    cursor.execute(f"""
        SELECT
            DOC_ID,
            TITLE,
            CATEGORY,
            AUTHOR,
            CONTENT,
            VECTOR_COSINE_SIMILARITY(
                FROM_JSON(EMBEDDING, 'ARRAY<FLOAT>'),
                FROM_JSON(?, 'ARRAY<FLOAT>')
            ) AS similarity
        FROM {CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}
        ORDER BY similarity DESC
        LIMIT {top_k}
    """, (query_json,))
```

Sample results:

```text
Query: 'how do I restart a service during an incident'

  DOC00089 | Runbooks | Casey Petrov
  Title: Runbook: clearing a stuck queue in AnalyticsAPI
  Similarity: 0.742
  Content: This runbook covers the procedure for clearing a stuck queue in the AnalyticsAPI
  environment. Follow these steps in order during an incident...

  DOC00023 | Runbooks | Morgan Chen
  Title: StreamProcessor Operations Guide
  Similarity: 0.731
  Content: Use this runbook when high error rates occurs in production. The on-call engineer
  should first check Grafana for anomalies...
```

### Filtered Search

Because the documents live in a standard Delta table, filtering is plain SQL. Any column in the table can be used as a filter - no attribute declaration, no separate filter index:

```python
def search_docs_filtered(
    query:    str,
    category: str = None,
    author:   str = None,
    top_k:    int = 5
):
    filters = []
    if category: filters.append(f"CATEGORY = '{category}'")
    if author:   filters.append(f"AUTHOR = '{author}'")

    where_clause = "WHERE " + " AND ".join(filters) if filters else ""

    cursor.execute(f"""
        SELECT ...,
            VECTOR_COSINE_SIMILARITY(
                FROM_JSON(EMBEDDING, 'ARRAY<FLOAT>'),
                FROM_JSON(?, 'ARRAY<FLOAT>')
            ) AS similarity
        FROM {CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}
        {where_clause}
        ORDER BY similarity DESC
        LIMIT {top_k}
    """, (query_json,))
```

Filtering to runbooks during an incident gives an immediately useful result set:

```text
query = 'steps to roll back a failed deployment', category = 'Runbooks'

  DOC00012 | Runbooks | Taylor Osei
  Title: DataPlatform Runbook: rolling back a deployment
  Similarity: 0.798
  Content: This runbook covers the procedure for rolling back a deployment in the DataPlatform
  environment. Follow these steps in order during an incident...
```

### Analytics with SQL

The analytics query uses a window function to compute each category's share of the total document corpus alongside absolute counts:

```python
cursor.execute(f"""
    SELECT
        CATEGORY,
        COUNT(*) AS TOTAL_DOCS,
        COUNT(DISTINCT AUTHOR) AS UNIQUE_AUTHORS,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS PCT_OF_TOTAL
    FROM {CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}
    GROUP BY CATEGORY
    ORDER BY TOTAL_DOCS DESC
""")
```

No second system, no result merging, no data movement. The semantic search and the analytics run against the same Delta table in the same Databricks SQL warehouse.

## What You'd Hit in Production

**Community Edition credit limits.** The Databricks Community Edition runs on serverless compute with a daily credit allowance. The credit allowance can be exhausted quickly if you leave a SQL warehouse running or run large queries repeatedly.

**`FROM_JSON` overhead.** Casting `STRING` to `ARRAY<FLOAT>` via `FROM_JSON` at query time adds overhead compared to storing embeddings natively as a vector type. For production, investigate whether storing embeddings in a native `ARRAY<FLOAT>` column directly - bypassing the JSON serialization step - is possible with your version of the connector. The `FROM_JSON` approach is reliable and readable but not optimal for very large tables.

**Full table scan.** `VECTOR_COSINE_SIMILARITY` on a `STRING` column performs a full table scan to compute similarity for every row before sorting and limiting. For datasets with hundreds of thousands of documents this will become slow. Databricks Vector Search - available on paid tiers - uses approximate nearest neighbor indexing to avoid this. For the community edition, keeping the dataset to a smaller number of documents is a practical limit.

**Databricks Vector Search.** The paid tier offers a managed Vector Search service backed by Delta tables, with approximate nearest neighbor indexing and automatic sync as the underlying table changes. This is the production path for large-scale semantic search in Databricks and avoids the full scan limitation entirely. The community edition approach in this chapter demonstrates the same concept with standard SQL functions as a starting point.

**Bulk insert limit.** The single-statement bulk insert approach works well for datasets up to a few hundred documents. For larger datasets the SQL string becomes very large and may hit connector or warehouse limits. For production bulk loads, use Databricks' native data ingestion tools such as `COPY INTO` or the Auto Loader, which are designed for large-scale data movement into Delta tables.

**Lakebase and pgvector.** For production RAG applications that need low-latency reads and full Postgres semantics alongside lakehouse data, Databricks Lakebase provides a fully managed Postgres with pgvector integrated directly into the platform. See the Lakebase section below for a working example.

## Lakebase - pgvector Inside the Lakehouse

Databricks Lakebase is a fully managed Postgres database integrated directly into the Databricks platform. It is the product of Databricks' acquisition of Neon in 2025 and brings serverless Postgres with pgvector into the lakehouse context - available on the free tier with scale-to-zero compute and one project per account.

This is significant for the book's story. Day 1 started with pgvector on a local Postgres install. Day 7 ends with pgvector running inside the Databricks lakehouse as a fully managed service. The underlying technology is the same; the operational context is completely different.

### Connecting to Lakebase

Lakebase uses standard Postgres drivers. We'll connect with `psycopg2` using an OAuth token obtained from the Lakebase Connect dialog in the Databricks UI:

```python
lb_conn = psycopg2.connect(
    host     = LAKEBASE_HOST,
    user     = LAKEBASE_USER,
    password = LAKEBASE_TOKEN,
    dbname   = LAKEBASE_DBNAME,
    sslmode  = "require",
    port     = 5432
)
```

The OAuth token expires after one hour. For production use, the recommended approach is OAuth token rotation via a Databricks service principal and the `generate_database_credential()` method from the Databricks SDK, which generates a fresh token for each new connection automatically.

### pgvector on Lakebase

pgvector is available on Lakebase and is enabled programmatically in the notebook:

```python
lb_cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
```

### Native Vector Column

Unlike the Delta table approach earlier in this chapter, Lakebase supports the native vector type directly. There is no `FROM_JSON` cast required - embeddings are stored and queried as proper vector columns:

```python
lb_cursor.execute(f"""
    CREATE TABLE {LAKEBASE_TABLE} (
        doc_id    TEXT PRIMARY KEY,
        title     TEXT,
        category  TEXT,
        author    TEXT,
        content   TEXT,
        embedding vector({EMBEDDING_DIMS})
    )
""")
```

We'll reuse the embeddings already generated earlier in the notebook - no additional Ollama calls needed. An HNSW index is created after loading for fast similarity search:

```python
lb_cursor.execute(f"""
    CREATE INDEX ON {LAKEBASE_TABLE}
    USING hnsw (embedding vector_cosine_ops)
""")
```

Queries use the `<=>` cosine distance operator - identical to Day 1:

```python
lb_cursor.execute(f"""
    SELECT
        doc_id, title, category, author, content,
        1 - (embedding <=> %s::vector) AS similarity
    FROM {LAKEBASE_TABLE}
    ORDER BY embedding <=> %s::vector
    LIMIT %s
""", (str(query_embedding), str(query_embedding), top_k))
```

Sample output:

```text
Lakebase query: 'how do I restart a service during an incident'

  DOC00108 | Runbooks | Reese Fontaine
  Title: ReportingService Operations Guide
  Similarity: 0.553
  Content: This runbook covers the procedure for restarting the service
  in the EventBus environment. Follow these steps in order during an incident...
```

The Lakebase connection, the vector column and the `<=>` operator are all standard Postgres and pgvector - the same as Day 1. What has changed is the operational context: the database is fully managed, scales to zero when idle and lives inside the Databricks platform alongside Delta tables, ML models and pipelines.

## When to Look Elsewhere

Databricks is the right choice when your data and workflows already live there. Consider alternatives if:

- You have no existing Databricks footprint. Databricks is a paid platform and standing it up just for vector search is hard to justify when simpler options exist. Day 1 (pgvector) and Day 3 (Pinecone) are considerably easier starting points.
- You need sub-second semantic search at large scale without a paid tier. The full table scan approach in this chapter does not scale to millions of documents at interactive latency. Databricks Vector Search on a paid tier addresses this, but the community edition does not.
- Your team is not already working in SQL or PySpark. The Databricks connector-based approach adds friction compared to databases with richer Python SDKs designed for application developers.
- You need the embedding pipeline to be fully managed. This chapter generates embeddings locally with Ollama. Databricks does offer managed embedding models through its Foundation Model APIs on paid tiers, but the community edition requires an external embedding step.

For data and ML teams already living in Databricks, adding semantic search to a Delta table is a natural and low-friction extension of existing workflows. The ability to combine document retrieval with full analytical SQL in a single system - without moving data or managing a separate vector service - makes Databricks a compelling choice for the right team.
