# Day 7: Databricks

## What is it?

Databricks is a leading data and AI platform built around the lakehouse architecture - a design that combines the flexibility and scale of a data lake with the structure and governance of a data warehouse. It's where many data engineering and machine learning teams already store, process and serve their data, using Apache Spark for large-scale computation, Delta Lake for reliable storage and MLflow for experiment tracking.

Databricks SQL adds a familiar SQL interface on top of the lakehouse, letting analysts and data engineers query Delta tables without writing Spark code. The `VECTOR_COSINE_SIMILARITY` function extends this to vector similarity search over `ARRAY<FLOAT>` embeddings, making it straightforward to add semantic search to data that already lives in the lakehouse - without introducing a separate vector database.

In 2025 Databricks acquired Neon, a serverless Postgres startup. The acquisition signals Databricks' intent to extend the lakehouse to cover the full spectrum of database workloads, including the transactional and vector search use cases that Neon's `pgvector` support provides. For teams already working in Databricks, this reinforces the "stay in the lakehouse" story that this chapter demonstrates.

## When would you reach for it?

The clearest signal is existing data and workflows in Databricks. If your data engineering pipelines, ML experiments and analytical queries already run in Databricks, adding semantic search to a Delta table is a natural extension of what you already do. There's no new system to learn, no data to move and no synchronization to maintain.

The second signal is a SQL and PySpark team. Data engineers and ML practitioners who work in Databricks already know Delta Lake, Databricks SQL and the Unity Catalog. `VECTOR_COSINE_SIMILARITY` slots into this existing toolchain as a function call in a familiar SQL query.

The third signal is analytics alongside search. Like Snowflake, Databricks is built for analytical workloads. If your use case requires not just "find the most relevant document" but also "how many documents exist per category" and "how many authors write in each one", both questions can be answered in the same system with SQL queries.

## The use case

For this chapter we'll build a RAG retrieval layer over internal technical documents - architecture notes, runbooks, API guides, security policies and onboarding materials. Users ask questions in natural language - "how do I restart a service during an incident" or "what are the security requirements for storing sensitive data" - and the system finds the most relevant documents from the corpus.

This is a natural fit for Databricks. Internal technical documentation is exactly the kind of content that data and engineering teams generate, store and share within a data platform. Keeping it searchable in the same system where teams already work reduces friction and keeps governance centralized.

## The data

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

The dataset scales via `NUM_DOCS`. Each document is generated from category-specific templates filled with randomized but realistic-sounding values. Titles and content are filled independently from the same pools, so a title and its content can mention different tasks or systems. All author names are fictional.

## Building the application

### Prerequisites

To follow along you'll need:

- A Databricks Free Edition account with access to a SQL warehouse
- Ollama running locally with the `all-minilm` model pulled
- Python 3.12 with a virtual environment

### Generate a personal access token

1. In the Databricks UI, click your username in the top bar and select **Settings**
2. Click **Developer**
3. Next to **Access tokens**, click **Manage**
4. Click **Generate new token**, give it a name, select **BI Tools** as the scope type and click **Generate**
5. Copy the token immediately, as it's only shown once

### Find your SQL warehouse connection details

1. In the Databricks UI, go to **SQL > SQL Warehouses**
2. Click your warehouse and select **Connection details**
3. Note down the **Server hostname** and **HTTP path**

### Configuration

We'll set the following environment variables before running the notebook:

```bash
export DATABRICKS_SERVER_HOSTNAME="dbc-xxxx.cloud.databricks.com"
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/xxxx"
export DATABRICKS_TOKEN="your-personal-access-token"
```

Then in the notebook:

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

> **Note:** `NUM_DOCS` controls the size of the generated dataset. 200 is the recommended default for this chapter - embedding generation runs locally via Ollama and is single-threaded, so larger values will work but will take proportionally longer. Production pipelines would typically use a hosted embedding endpoint with async or batched generation to handle scale.

> **Note:** The default catalog in a Free Edition workspace is `workspace`. If you see a `NO_SUCH_CATALOG_EXCEPTION` error, check which catalogs are available with `SHOW CATALOGS` and update `CATALOG_NAME` accordingly.

### Determine embedding dimensions

We'll determine the embedding dimensions dynamically from a test embedding:

```python
def get_embedding(text: str) -> list:
    response = ollama.embeddings(model = LLM_EMBEDDING, prompt = text)
    return response["embedding"]

test_embedding = get_embedding("internal technical documentation")
EMBEDDING_DIMS = len(test_embedding)
print(f"Embedding dimensions: {EMBEDDING_DIMS}")
```

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

### Generate the dataset

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

### Generate embeddings

We'll embed the content of each document locally using Ollama. The embedding is stored as a string and cast to `ARRAY<FLOAT>` at query time:

```python
embeddings = []
for doc in tqdm(documents, desc = "Generating embeddings"):
    embeddings.append(get_embedding(doc["CONTENT"]))

df["EMBEDDING"] = [str(e) for e in embeddings]
```

### Load data into Databricks

We'll create a Delta table and load all documents in a single bulk `INSERT`. The embedding is stored as a `STRING` column holding JSON text, for reasons explained below.

> **Note:** Inserting one row per document over a network connection to Databricks is extremely slow - each `cursor.execute()` call is a separate round trip. Concatenating all rows into a single `VALUES` clause and sending one SQL statement is significantly faster.

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

### Semantic search

We'll embed the query with Ollama and pass it to Databricks SQL as a parameter. `VECTOR_COSINE_SIMILARITY` computes cosine similarity between the stored embedding and the query embedding.

> **Note:** `VECTOR_COSINE_SIMILARITY` operates on `ARRAY<FLOAT>` values. We store the embeddings as JSON text in a `STRING` column, which keeps the bulk `INSERT` simple, and `FROM_JSON` parses the text into an `ARRAY<FLOAT>` at query time. Both the stored embedding and the query embedding need this parsing step.

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

Sample results (top two of five):

```text
Query: 'how do I restart a service during an incident'

  DOC00108 | Runbooks | Reese Fontaine
  Title: ReportingService Operations Guide
  Similarity: 0.553
  Content: This runbook covers the procedure for restarting the service in the EventBus environment. Follow these steps in order during an incident. First, verif...

  DOC00039 | Runbooks | Casey Petrov
  Title: Runbook: database connection exhaustion in EventBus
  Similarity: 0.508
  Content: This runbook documents the steps to restarting the service for the DataVault service. Prerequisites: access to the admin panel. Steps: (1) Verify aler...
```

### Filtered search

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

Filtering to runbooks during an incident gives an immediately useful result set (top two of five):

```text
query = 'steps to roll back a failed deployment', category = 'Runbooks'

  DOC00024 | Runbooks | Jordan Blake
  Title: AnalyticsAPI Runbook: scaling the cluster
  Similarity: 0.574
  Content: This runbook covers the procedure for rolling back a deployment in the SearchEngine environment. Follow these steps in order during an incident. First...

  DOC00138 | Runbooks | Riley Kumar
  Title: Runbook: high error rates in DataVault
  Similarity: 0.547
  Content: This runbook covers the procedure for rolling back a deployment in the DataVault environment. Follow these steps in order during an incident. First, v...
```

The first title says "scaling the cluster" while the content is about rolling back a deployment. That's the independent title and content filling described earlier, not a search error.

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

## What you'd hit in production

**Free Edition usage limits.** Databricks Free Edition runs on serverless compute with a usage quota. The quota can be used up quickly if you run large queries repeatedly.

**`FROM_JSON` overhead.** Parsing a `STRING` into an `ARRAY<FLOAT>` via `FROM_JSON` adds overhead for every row on every query. For production, store the embeddings in an `ARRAY<FLOAT>` column and convert them once at load time. The `FROM_JSON` approach is reliable and readable but isn't optimal for very large tables.

**Full table scan.** Every query parses the stored embedding and computes the similarity for every row before sorting and limiting. For datasets with hundreds of thousands of documents this will become slow. Databricks Vector Search uses approximate nearest neighbor indexing to avoid this, but check whether it's available on your plan. On Free Edition, keeping the dataset to a smaller number of documents is a practical limit.

**Databricks Vector Search.** Databricks also offers a managed Vector Search service backed by Delta tables, with approximate nearest neighbor indexing and automatic sync as the underlying table changes. Availability depends on your plan. This is the production path for large-scale semantic search in Databricks and avoids the full scan limitation entirely. The approach in this chapter demonstrates the same concept with standard SQL functions as a starting point.

**Bulk insert limit.** The single-statement bulk insert approach works well for datasets up to a few hundred documents. For larger datasets the SQL string becomes very large and may hit connector or warehouse limits. For production bulk loads, use Databricks' native data ingestion tools such as `COPY INTO` or the Auto Loader, which are designed for large-scale data movement into Delta tables.

**Lakebase and pgvector.** For production RAG applications that need low-latency reads and full Postgres semantics alongside lakehouse data, Databricks Lakebase provides a fully managed Postgres with `pgvector` integrated directly into the platform. See the Lakebase section below for a working example.

## Lakebase - pgvector inside the Lakehouse

Databricks Lakebase is a fully managed Postgres database integrated directly into the Databricks platform. It's the product of Databricks' acquisition of Neon in 2025 and brings serverless Postgres with `pgvector` into the lakehouse context, with scale-to-zero compute.

This is significant for the book's story. Day 1 started with `pgvector` on a local Postgres install. Day 7 ends with `pgvector` running inside the Databricks lakehouse as a fully managed service. The underlying technology is the same; the operational context is completely different.

### Configuration

We'll set the following environment variables before running the notebook:

```bash
export LAKEBASE_HOST="ep-xxxx.database.xxxx.cloud.databricks.com"
export LAKEBASE_USER="your-email-address"
export LAKEBASE_TOKEN="your-oauth-token"
export LAKEBASE_DBNAME="databricks_postgres"
```

Then in the notebook:

```python
LAKEBASE_HOST   = os.environ["LAKEBASE_HOST"]
LAKEBASE_USER   = os.environ["LAKEBASE_USER"]
LAKEBASE_TOKEN  = os.environ["LAKEBASE_TOKEN"]
LAKEBASE_DBNAME = os.environ["LAKEBASE_DBNAME"]
LAKEBASE_TABLE  = "wiki_documents"
```

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

> **Note:** The `OAuth` token expires after one hour. For production use, the recommended approach is `OAuth` token rotation via a Databricks service principal and the `generate_database_credential()` method from the Databricks SDK, which generates a fresh token for each new connection automatically.

### pgvector on Lakebase

`pgvector` is available on Lakebase and is enabled programmatically in the notebook:

```python
lb_cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
```

### Native vector column

Unlike the Delta table approach earlier in this chapter, Lakebase supports the `pgvector` `VECTOR` type directly. There's no `FROM_JSON` cast required - embeddings are stored and queried as proper vector columns:

```python
lb_cursor.execute(f"""
    CREATE TABLE {LAKEBASE_TABLE} (
        doc_id    TEXT PRIMARY KEY,
        title     TEXT,
        category  TEXT,
        author    TEXT,
        content   TEXT,
        embedding VECTOR({EMBEDDING_DIMS})
    )
""")
```

We'll reuse the embeddings already generated earlier in the notebook - no additional Ollama calls needed. An `HNSW` index is created after loading for fast similarity search:

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

Sample output (top result of five):

```text
Lakebase query: 'how do I restart a service during an incident'

  DOC00108 | Runbooks | Reese Fontaine
  Title: ReportingService Operations Guide
  Similarity: 0.553
  Content: This runbook covers the procedure for restarting the service in the EventBus environment. Follow these steps in order during an incident. First, verif...
```

All five results and their scores are identical to the Delta table search earlier in the chapter, which is a useful cross-check that both approaches compute the same cosine similarity.

The Lakebase connection, the vector column and the `<=>` operator are all standard Postgres and `pgvector` - the same as Day 1. What has changed is the operational context: the database is fully managed, scales to zero when idle and lives inside the Databricks platform alongside Delta tables, ML models and pipelines.

## When to look elsewhere

Databricks is the right choice when your data and workflows already live there. Consider alternatives if:

- You have no existing Databricks footprint. Databricks is a commercial platform and standing it up just for vector search is hard to justify when simpler options exist. Day 1 (`pgvector`) and Day 3 (Pinecone) are considerably easier starting points.
- You need sub-second semantic search at large scale. The full table scan approach in this chapter doesn't scale to millions of documents at interactive latency. Databricks Vector Search addresses this with approximate nearest neighbor indexing, but it's a separate service to set up.
- Your team is not already working in SQL or PySpark. The Databricks connector-based approach adds friction compared to databases with richer Python SDKs designed for application developers.
- You need the embedding pipeline to be fully managed. This chapter generates embeddings locally with Ollama. Databricks does offer managed embedding models through its Foundation Model APIs, and using them would remove the external embedding step.

For data and ML teams already living in Databricks, adding semantic search to a Delta table is a natural and low-friction extension of existing workflows. The ability to combine document retrieval with full analytical SQL in a single system - without moving data or managing a separate vector service - makes Databricks a compelling choice for the right team.
