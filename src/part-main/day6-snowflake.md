# Day 6: Snowflake

## What Is It?

Snowflake is a leading cloud data platform. It's where many organizations already store their operational and analytical data - data warehouse exports, CRM records, support tickets, event logs and more. Rather than moving that data to a separate vector database, Snowflake's native `VECTOR` data type and `VECTOR_COSINE_SIMILARITY` function bring vector similarity search directly to the data where it already lives.

This is a different proposition from the previous five days. Snowflake is not a vector database - it is a cloud data warehouse that has added vector search as a first-class capability. The argument is data gravity: if your support tickets, customer records or product data already live in Snowflake, the simplest path to semantic search is to stay there rather than copy data into a dedicated vector system and build a synchronization pipeline to keep the two in sync.

Snowflake's vector support consists of two main pieces. The `VECTOR` data type stores fixed-dimension floating-point vectors natively. The `VECTOR_COSINE_SIMILARITY` function computes cosine similarity between two vectors in a SQL query. Both are available on all Snowflake account types including trial accounts.

## When Would You Reach for It?

The clearest signal is existing data in Snowflake. If your organization's data is already in Snowflake - and for many data engineering and analytics teams it is - adding semantic search is a matter of storing embeddings in a new column and writing a SQL query. There is no new system to provision, no data to move and no synchronization to maintain.

The second signal is a SQL-native team. Data engineers and analysts who already work in Snowflake will find `VECTOR_COSINE_SIMILARITY` immediately familiar. It slots into existing query patterns, works alongside standard `WHERE` clauses, `GROUP BY` aggregations and window functions, and integrates naturally with Snowflake's access control and governance features.

The third signal is analytics alongside search. Snowflake is built for analytical workloads. If your use case requires not just "find similar tickets" but also "how many critical tickets are unresolved this week" and "what is the average resolution time by category", both questions can be answered in the same database with the same SQL toolchain.

## The Use Case

For this chapter we'll build a customer support analytics system. Support teams can describe an issue in natural language - "customer cannot log in after forgetting their password" or "application is slow and pages take a long time to load" - and find the most relevant historical tickets from the database. Filters by category, priority and status let teams narrow results to the most actionable matches. A summary analytics query shows ticket volumes and resolution rates by category alongside the semantic search.

This is a natural fit for Snowflake. Support ticket data lives in data warehouses and CRM systems. The people who work with it - support operations, customer success and analytics teams - are SQL users. Adding semantic search without leaving SQL is a meaningful reduction in operational complexity.

## The Data

We generate synthetic customer support tickets from pools of categories, priorities, products, statuses and description templates. Each ticket has a free-text description and an optional resolution note.

The key fields are:

- `TICKET_ID` - a unique ticket identifier, e.g. "TKT00042"
- `CATEGORY` - e.g. "Billing", "Technical", "Account", "Security"
- `PRIORITY` - "Low", "Medium", "High" or "Critical"
- `STATUS` - "Open", "In Progress", "Resolved" or "Closed"
- `PRODUCT` - one of ten fictional product names
- `DESCRIPTION` - a prose description of the issue; this is what we embed and search over
- `RESOLUTION` - a resolution note, present only for resolved and closed tickets
- `EMBEDDING` - the embedding vector stored as a `VARCHAR`, cast to `VECTOR` at query time

Resolution notes are only present for tickets with a status of "Resolved" or "Closed", which reflects real-world ticket data. This makes resolved tickets particularly useful for finding past solutions to current problems.

## Building the Application

### Prerequisites

To follow along you'll need:

- A Snowflake free trial account
- Ollama running locally with the `all-minilm` model pulled
- Python 3.12 with a virtual environment

> **Note:** Snowflake offers a managed vector search service called Cortex Search, which handles embedding generation internally. However, the embedding functions it relies on are not available on trial accounts. For this chapter we use Ollama to generate embeddings locally and store them in Snowflake using the native `VECTOR` data type, which is available on all account types.

### Configuration

We'll set the following environment variables before running the notebook:

```bash
export SNOWFLAKE_ACCOUNT="your-account-identifier"
export SNOWFLAKE_USER="your-username"
export SNOWFLAKE_PASSWORD="your-password"
```

Then in the notebook:

```python
SNOWFLAKE_ACCOUNT   = os.environ["SNOWFLAKE_ACCOUNT"]
SNOWFLAKE_USER      = os.environ["SNOWFLAKE_USER"]
SNOWFLAKE_PASSWORD  = os.environ["SNOWFLAKE_PASSWORD"]
SNOWFLAKE_DATABASE  = "SUPPORT_DB"
SNOWFLAKE_SCHEMA    = "SUPPORT_SCHEMA"
SNOWFLAKE_WAREHOUSE = "SUPPORT_WH"
SNOWFLAKE_ROLE      = "ACCOUNTADMIN"
TABLE_NAME          = "SUPPORT_TICKETS"
LLM_EMBEDDING       = "all-minilm"
NUM_TICKETS         = 200
RANDOM_SEED         = 42
```

> **Note:** `NUM_TICKETS` controls the size of the generated dataset. 200 is the recommended default for this chapter - embedding generation runs locally via Ollama and is single-threaded, so larger values will work but will take proportionally longer. Production pipelines would typically use a hosted embedding endpoint with async or batched generation to handle scale.

### Determine Embedding Dimensions

We'll determine the embedding dimensions dynamically from a test embedding:

```python
def get_embedding(text: str) -> list:
    response = ollama.embeddings(model = LLM_EMBEDDING, prompt = text)
    return response["embedding"]

test_embedding = get_embedding("customer support ticket about billing issue")
EMBEDDING_DIMS = len(test_embedding)
print(f"Embedding dimensions: {EMBEDDING_DIMS}")
```

### Connect to Snowflake

```python
conn = snowflake.connector.connect(
    account  = SNOWFLAKE_ACCOUNT,
    user     = SNOWFLAKE_USER,
    password = SNOWFLAKE_PASSWORD,
    role     = SNOWFLAKE_ROLE,
)

cursor = conn.cursor()
print(f"Connected to Snowflake: {conn.account}")
```

### Create Database, Schema and Warehouse

We'll create all infrastructure programmatically using `cursor.execute()`. The warehouse is set to `X-SMALL` with `AUTO_SUSPEND = 60` to minimize credit consumption:

```python
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {SNOWFLAKE_DATABASE}")
cursor.execute(f"USE DATABASE {SNOWFLAKE_DATABASE}")
cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {SNOWFLAKE_SCHEMA}")
cursor.execute(f"USE SCHEMA {SNOWFLAKE_SCHEMA}")
cursor.execute(f"""
    CREATE WAREHOUSE IF NOT EXISTS {SNOWFLAKE_WAREHOUSE}
    WITH WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME  = TRUE
""")

cursor.execute(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}")
```

### Generate the Dataset

Description templates are category-specific so each ticket gets a relevant, realistic description:

```python
DESCRIPTION_TEMPLATES = {
    "Billing": [
        "Customer was charged twice for the same subscription period and is requesting a refund for the duplicate charge.",
        "Invoice amount does not match the quoted price. Customer is disputing the difference and requesting a corrected invoice.",
        # ... further templates
    ],
    "Technical": [
        "Customer reports the application is crashing on startup after the latest update was applied to their system.",
        "API calls are returning 500 errors intermittently. Customer has provided request IDs and timestamps for investigation.",
        # ... further templates
    ],
    # ... further categories
}

def generate_ticket(ticket_id: int) -> dict:
    category = random.choice(CATEGORIES)
    status   = random.choice(STATUSES)
    return {
        "TICKET_ID":   f"TKT{ticket_id:05d}",
        "CATEGORY":    category,
        "PRIORITY":    random.choice(PRIORITIES),
        "STATUS":      status,
        "PRODUCT":     random.choice(FICTIONAL_PRODUCTS),
        "DESCRIPTION": random.choice(DESCRIPTION_TEMPLATES[category]),
        "RESOLUTION":  RESOLUTION_TEMPLATES[category] if status in ["Resolved", "Closed"] else None,
    }
```

### Generate Embeddings

We embed each ticket description locally with Ollama and store the result as a string in a new `EMBEDDING` column on the DataFrame:

```python
embeddings = []
for ticket in tqdm(tickets, desc = "Generating embeddings"):
    embeddings.append(get_embedding(ticket["DESCRIPTION"]))

df["EMBEDDING"] = [str(e) for e in embeddings]
```

The embedding is stored as a string because `write_pandas` does not support the `VECTOR` type directly. We cast it to `VECTOR` at query time.

### Load Data into Snowflake

We'll create the table with an `EMBEDDING VARCHAR(16000)` column and load the data using `write_pandas`:

```python
cursor.execute(f"""
    CREATE TABLE {TABLE_NAME} (
        TICKET_ID   VARCHAR(20),
        CATEGORY    VARCHAR(50),
        PRIORITY    VARCHAR(20),
        STATUS      VARCHAR(20),
        PRODUCT     VARCHAR(100),
        DESCRIPTION VARCHAR(1000),
        RESOLUTION  VARCHAR(1000),
        EMBEDDING   VARCHAR(16000)
    )
""")

success, num_chunks, num_rows, _ = write_pandas(
    conn = conn,
    df = df,
    table_name = TABLE_NAME,
    auto_create_table = False
)
```

### Semantic Search

We'll embed the query with Ollama, pass it to Snowflake as a SQL parameter and use `VECTOR_COSINE_SIMILARITY` to rank results.

> **Note:** Snowflake cannot cast a `VARCHAR` directly to `VECTOR`. The stored embedding string must first be parsed into a `VARIANT` (Snowflake's JSON type) using `TRY_PARSE_JSON`, which can then be cast to `VECTOR`. The same applies to the query embedding passed as a parameter.

```python
def search_tickets(query: str, top_k: int = 5):
    query_embedding = get_embedding(query)
    query_vec_str = str(query_embedding)

    cursor.execute(f"""
        SELECT
            TICKET_ID,
            CATEGORY,
            PRIORITY,
            STATUS,
            PRODUCT,
            DESCRIPTION,
            RESOLUTION,
            VECTOR_COSINE_SIMILARITY(
                TRY_PARSE_JSON(EMBEDDING)::VECTOR(FLOAT, {EMBEDDING_DIMS}),
                TRY_PARSE_JSON(%s)::VECTOR(FLOAT, {EMBEDDING_DIMS})
            ) AS similarity
        FROM {TABLE_NAME}
        ORDER BY similarity DESC
        LIMIT {top_k}
    """, (query_vec_str,))
```

Sample results:

```text
Query: 'customer cannot log in after forgetting their password'

  TKT00001 | Account | Medium | Resolved
  Product: InsightEngine
  Similarity: 0.918
  Description: Customer has forgotten their password and is not receiving the password reset email in their inbox.
  Resolution: Account team verified the customer's identity and restored access. Security review completed.

  TKT00012 | Account | High | Open
  Product: SecurePortal
  Similarity: 0.918
  Description: Two-factor authentication is locked and customer cannot access their account from a new device.
```

### Filtered Search

Because the tickets live in a standard Snowflake table, filtering is plain SQL. We'll add `WHERE` clauses to the similarity query - no special filter syntax, no filter index declaration, no attribute list defined at index creation time. Any column in the table can be used as a filter:

```python
def search_tickets_filtered(
    query:    str,
    category: str = None,
    priority: str = None,
    status:   str = None,
    product:  str = None,
    top_k:    int = 5
):
    filters = []
    if category: filters.append(f"CATEGORY = '{category}'")
    if priority: filters.append(f"PRIORITY = '{priority}'")
    if status:   filters.append(f"STATUS = '{status}'")
    if product:  filters.append(f"PRODUCT = '{product}'")

    where_clause = "WHERE " + " AND ".join(filters) if filters else ""

    cursor.execute(f"""
        SELECT ...,
            VECTOR_COSINE_SIMILARITY(
                TRY_PARSE_JSON(EMBEDDING)::VECTOR(FLOAT, {EMBEDDING_DIMS}),
                TRY_PARSE_JSON(%s)::VECTOR(FLOAT, {EMBEDDING_DIMS})
            ) AS similarity
        FROM {TABLE_NAME}
        {where_clause}
        ORDER BY similarity DESC
        LIMIT {top_k}
    """, (query_vec_str,))
```

Finding resolved technical issues is particularly useful for support teams - a query that returns past tickets with resolutions gives agents an immediate starting point for troubleshooting:

```text
query = 'API returning errors and integration not working', category = 'Technical', status = 'Resolved'

  TKT00003 | Technical | Low | Resolved
  Product: ConnectAPI
  Similarity: 0.836
  Description: Integration with a third-party service stopped working after a configuration change on the customer's side.
  Resolution: Engineering team identified the root cause and deployed a fix. Customer confirmed the issue is resolved.
```

### Analytics with SQL

The real advantage of keeping data in Snowflake is the ability to run analytical queries alongside semantic search in the same system. Here we summarize ticket volumes, resolution counts and resolution rates by category:

```python
cursor.execute(f"""
    SELECT
        CATEGORY,
        COUNT(*)                                                            AS TOTAL_TICKETS,
        SUM(CASE WHEN STATUS IN ('Resolved', 'Closed') THEN 1 ELSE 0 END) AS RESOLVED,
        SUM(CASE WHEN PRIORITY = 'Critical' THEN 1 ELSE 0 END) AS CRITICAL,
        ROUND(
            100.0 * SUM(CASE WHEN STATUS IN ('Resolved', 'Closed') THEN 1 ELSE 0 END) / COUNT(*),
            1
        ) AS RESOLUTION_RATE_PCT
    FROM {TABLE_NAME}
    GROUP BY CATEGORY
    ORDER BY TOTAL_TICKETS DESC
""")

summary = cursor.fetch_pandas_all()
```

No second system, no result merging, no data movement. The semantic search and the analytics live in the same table and run against the same Snowflake warehouse.

## What You'd Hit in Production

**Cortex Search on trial accounts.** Snowflake Cortex Search handles embedding generation internally and provides a managed search service. However, the underlying embedding functions are not available on trial accounts. For production use with a paid Snowflake account, Cortex Search is worth evaluating - it removes the need for an external embedding pipeline and manages index updates as the table changes. For trial accounts or teams with an existing embedding pipeline, the `VECTOR` + `VECTOR_COSINE_SIMILARITY` approach shown in this chapter is an alternative.

**`TRY_PARSE_JSON` overhead.** Casting the stored `VARCHAR` to `VECTOR` via `TRY_PARSE_JSON` at query time adds overhead compared to storing embeddings natively as `VECTOR`. For production, consider storing embeddings directly in a `VECTOR(FLOAT, N)` column rather than as a string. The `write_pandas` function does not currently support the `VECTOR` type, so you would need to use a `PUT` and `COPY INTO` pattern or insert rows individually.

**Warehouse costs.** Every query that runs `VECTOR_COSINE_SIMILARITY` across a large table requires the warehouse to be running. With `AUTO_SUSPEND = 60`, the warehouse suspends after a minute of inactivity. For low-traffic use cases this is cost-effective; for high-traffic search applications the cost of keeping the warehouse running can add up. Cortex Search manages this differently by running search on a dedicated infrastructure.

**Full table scan.** The `VECTOR_COSINE_SIMILARITY` query performs a full table scan to compute similarity for every row before sorting and limiting. For datasets with hundreds of thousands of tickets this works well. For millions of rows, query latency will increase. Cortex Search uses approximate nearest neighbor indexing to avoid full scans at scale.

**Warehouse size for large datasets.** At 200 tickets an `X-SMALL` warehouse is more than sufficient. For larger datasets or faster query times, increase the warehouse size. Snowflake auto-scales compute independently of storage, so you can use a larger warehouse for bulk embedding inserts and scale back down for queries.

## When to Look Elsewhere

Snowflake is the right choice when your data are already there. Consider alternatives if:

- You have no existing Snowflake footprint. Snowflake is not free - it operates on a credit model and production usage costs money. Standing up Snowflake just for vector search would be hard to justify when purpose-built vector databases or `pgvector` on a managed Postgres are simpler and cheaper.
- You need sub-second similarity search at very large scale. Full table scans do not scale indefinitely. Cortex Search on a paid account addresses this with approximate nearest neighbor indexing, but the free vector search approach in this chapter is inherently a scan-based approach.
- Your embedding pipeline is not already managed. This chapter requires generating embeddings externally with Ollama and loading them manually. If you are starting from scratch with no existing pipeline, a database that handles embeddings internally - like Snowflake's Cortex Search on a paid account - reduces operational complexity.
- Your team is not SQL-native. The Snowflake approach is cleanest for teams already working in SQL. For teams building Python applications, the connector-based approach adds friction compared to databases with richer Python SDKs.

For data teams already working in Snowflake, adding vector search to an existing table is a low-friction, high-value addition. The ability to combine semantic search with full SQL analytics in a single system - without moving data or introducing a new operational dependency - is a meaningful advantage for the right team and the right use case.
