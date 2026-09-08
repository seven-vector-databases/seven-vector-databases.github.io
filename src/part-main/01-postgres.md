# Day 1: PostgreSQL + pgvector

## What Is It?

PostgreSQL needs little introduction. It is the world's most widely deployed open-source relational database, with a history stretching back to the late 1980s and an ecosystem that spans virtually every programming language, cloud platform and deployment model imaginable. Developers reach for Postgres because it is reliable, standards-compliant and extraordinarily capable - it handles relational data, JSON documents, full-text search and geospatial queries, all without leaving the database.

The `pgvector` extension adds one more capability to that list: native vector similarity search. Released in 2021 and now available on every major managed Postgres platform, `pgvector` introduces a `VECTOR` data type and the indexing structures needed to search across it efficiently. With a single `CREATE EXTENSION` command, a Postgres database becomes a vector database.

What makes this interesting is not that `pgvector` is the fastest or most feature-rich vector search implementation - it's not. What makes it interesting is that it requires no new infrastructure, no new operational skills and no new mental model. If your application already runs on Postgres, vector search is one extension away.

## When Would You Reach for It?

The clearest signal is existing Postgres data. If your application already stores its data in Postgres - user profiles, product catalogs, support tickets, documents - adding `pgvector` means vector search lives alongside that data in the same database. You can filter by salary, location, date or any other column in the same query that computes semantic similarity. No synchronization between databases, no dual writes, no operational overhead of a second system.

The second signal is team familiarity. A team that knows SQL and knows Postgres can be productive with `pgvector` immediately. There is no new query language to learn, no new SDK to integrate and no new infrastructure to operate.

The third signal is scale. `pgvector` handles millions of vectors comfortably, which covers the majority of real-world use cases. If you are building a general-purpose search engine over billions of documents, you will eventually need something more specialized. But for most applications - internal tools, product search, recommendation features, RAG pipelines over a bounded corpus - `pgvector` is more than sufficient.

## The Use Case

For this chapter we'll build a semantic job listing search engine. Users describe what they are looking for in natural language - "I want to work on machine learning models in production" or "looking for a data pipeline and ETL role" - and the system returns the most relevant listings from the database.

This use case is a natural fit for Postgres. Job listings are structured data: they have titles, locations, salary ranges and skills lists, all of which are natural relational columns. But the richest signal for relevance is the free-text description, which is where vector search earns its place. By embedding each description and indexing the resulting vectors, we can match a user's query against the meaning of a listing rather than just its keywords.

The relational advantage becomes clear when we add filters. A user who wants a data engineering role in New York with a minimum salary of $130,000 should not have to choose between semantic relevance and structured constraints. With `pgvector`, both happen in a single SQL query.

## The Data

We generate a synthetic dataset of job listings programmatically rather than using a fixed hardcoded list. This keeps the notebook self-contained and lets us scale to any number of listings by changing a single configuration variable.

Each listing is assembled from pools of roles, companies, US cities, salary bands, skills and description templates. The templates are domain-aware - data roles get data-flavored descriptions, machine learning roles get ML-flavored ones - which gives enough variation for meaningful semantic search.

The key fields are:

- `title` - the job title, e.g. "Senior Data Engineer"
- `company` - the hiring company
- `location` - a US city or "Remote"
- `salary_min` and `salary_max` - salary band in US dollars
- `skills` - an array of relevant technologies
- `description` - a short prose description of the role; this is what we embed

The `description` field is the one that carries semantic meaning. Everything else is structured metadata that we use for filtering.

## Building the Application

### Prerequisites

To follow along you will need:

- PostgreSQL 16 installed locally (use Homebrew on a Mac)
- `pgvector` installed from source (see the note below)
- Ollama running locally with the `all-minilm` model pulled
- Python 3.12 with a virtual environment

> **Note:** The Homebrew formula for `pgvector` installs cleanly but leaves an empty package on Apple Silicon Macs - the extension files are simply not present after installation. The workaround is to install from source, as shown below.

```shell
brew uninstall pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

This compiles `pgvector` against the local Postgres installation and places the extension files in the correct location.

> **Note:** A Homebrew Postgres installation does not create a `postgres` superuser. Instead, it creates a role matching the Mac username. When connecting from Python, use `os.environ.get("USER")` rather than hardcoding `"postgres"`.

### Configuration

We'll keep all tuneable parameters in a single cell at the top of the notebook. `NUM_JOBS` controls the size of the generated dataset - 500 is a reasonable default for exploring search quality. `RANDOM_SEED` ensures the same dataset is generated on every run.

```python
LLM_EMBEDDING = "all-minilm"
NUM_JOBS      = 500
RANDOM_SEED   = 42
```

### Determine Embedding Dimensions

We'll determine the embedding dimensions dynamically from a test embedding:

```python
def get_embedding(text: str) -> list:
    response = ollama.embeddings(model = LLM_EMBEDDING, prompt = text)
    return response["embedding"]

test_embedding = get_embedding("senior data engineer with Python and Spark")
EMBEDDING_DIMS = len(test_embedding)
print(f"Embedding dimensions: {EMBEDDING_DIMS}")
```

### Connect to PostgreSQL and Enable pgvector

```python
conn = psycopg2.connect(
    dbname = "postgres",
    user   = os.environ.get("USER"),
    host   = "localhost",
    port   = 5432
)

conn.autocommit = True
cursor = conn.cursor()

cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
print("pgvector extension enabled.")
```

### Generate the Dataset

Rather than a hardcoded list, we'll assemble listings from component pools. Each description is built from a template appropriate to the role's domain:

```python
random.seed(RANDOM_SEED)

ROLES = [
    {"title": "Senior Data Engineer",     "domain": "data", "salary_band": (130000, 160000)},
    {"title": "Machine Learning Engineer","domain": "ml",   "salary_band": (150000, 190000)},
    # ... further roles
]

DESCRIPTION_TEMPLATES = {
    "data": [
        "Build and maintain {system} for a {company_type} specializing in {domain}. "
        "Work closely with {team} to ensure data quality and reliability.",
        # ... further templates
    ],
    # ... further domains
}

def generate_description(domain: str) -> str:
    template = random.choice(DESCRIPTION_TEMPLATES[domain])
    return template.format(
        system       = random.choice(SYSTEMS_BY_DOMAIN[domain]),
        company_type = random.choice(COMPANY_TYPES),
        platform     = random.choice(PLATFORMS),
        team         = random.choice(TEAMS),
        domain       = random.choice(DOMAINS),
    )

def generate_job_listings(n: int) -> list:
    listings = []
    for _ in range(n):
        role   = random.choice(ROLES)
        domain = role["domain"]
        sal_min, sal_max = role["salary_band"]
        offset = random.choice([-10000, -5000, 0, 5000, 10000])
        listings.append({
            "title":       role["title"],
            "company":     random.choice(COMPANIES),
            "location":    random.choice(LOCATIONS),
            "salary_min":  sal_min + offset,
            "salary_max":  sal_max + offset,
            "skills":      random.choice(SKILLS_BY_DOMAIN[domain]),
            "description": generate_description(domain),
        })
    return listings

job_listings = generate_job_listings(NUM_JOBS)
```

### Create the Table

The table schema is straightforward. The `embedding` column uses the `VECTOR({EMBEDDING_DIM})` type, which matches the output dimensionality of `all-minilm`.

```python
cursor.execute("DROP TABLE IF EXISTS job_listings;")

cursor.execute(f"""
    CREATE TABLE job_listings (
        id          SERIAL PRIMARY KEY,
        title       TEXT,
        company     TEXT,
        location    TEXT,
        salary_min  INTEGER,
        salary_max  INTEGER,
        skills      TEXT[],
        description TEXT,
        embedding   VECTOR({EMBEDDING_DIMS})
    );
""")
```

### Generate Embeddings and Load Data

We embed each job description using Ollama and insert the result alongside the structured fields. The `tqdm` progress bar gives useful feedback when loading larger datasets.

```python
def get_embedding(text: str) -> list:
    response = ollama.embeddings(model = LLM_EMBEDDING, prompt = text)
    return response["embedding"]

for job in tqdm(job_listings, desc = "Inserting listings"):
    embedding = get_embedding(job["description"])
    cursor.execute("""
        INSERT INTO job_listings
            (title, company, location, salary_min, salary_max, skills, description, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        job["title"], job["company"], job["location"],
        job["salary_min"], job["salary_max"], job["skills"],
        job["description"], str(embedding)
    ))
```

### Create the Vector Index

`pgvector` supports two index types: `IVFFlat` and `HNSW`. We'll use `HNSW`, which gives better recall and is the recommended default for most workloads. Note that we'll create the index after loading data - building it incrementally during inserts is significantly slower for bulk loads.

```python
cursor.execute("""
    CREATE INDEX ON job_listings
    USING hnsw (embedding vector_cosine_ops);
""")
```

### Semantic Search

We embed the user's query and find the most similar job descriptions using the `<=>` cosine distance operator.

```python
def search_jobs(query: str, top_k: int = 5):
    query_embedding = get_embedding(query)
    cursor.execute("""
        SELECT title, company, location, salary_min, salary_max,
               1 - (embedding <=> %s::vector) AS similarity
        FROM job_listings
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (str(query_embedding), str(query_embedding), top_k))

    results = cursor.fetchall()
    print(f"\nQuery: '{query}'\n")
    for title, company, location, sal_min, sal_max, similarity in results:
        print(f"  {title} @ {company} - {location}")
        print(f"  ${sal_min:,} - ${sal_max:,} | Similarity: {similarity:.3f}\n")
```

Running a few queries against the 500-listing dataset:

```text
search_jobs("I want to work on machine learning models in production")

Query: 'I want to work on machine learning models in production'

  ML Ops Engineer @ RetailIQ - Phoenix, AZ
  $135,000 - $165,000 | Similarity: 0.717

  Machine Learning Engineer @ EdgeCore - Portland, OR
  $155,000 - $195,000 | Similarity: 0.700

  Prompt Engineer @ InsightCo - Portland, OR
  $110,000 - $140,000 | Similarity: 0.692

  Computer Vision Engineer @ GrowthLab - Austin, TX
  $140,000 - $175,000 | Similarity: 0.669

  ML Ops Engineer @ AlphaEdge - Austin, TX
  $140,000 - $170,000 | Similarity: 0.635
```

### The Relational Advantage - Filtered Search

This is where Postgres genuinely earns its place. We combine semantic similarity with standard SQL filters in a single query. The key subtlety in the implementation is parameter ordering - the first embedding parameter appears in the `SELECT` clause, the filter parameters follow in the `WHERE` clause and the second embedding parameter appears in the `ORDER BY`. Getting this wrong produces an `InvalidTextRepresentation` error as Postgres tries to interpret an embedding string as an integer.

```python
def search_jobs_filtered(query: str, location: str = None, min_salary: int = None, top_k: int = 5):
    query_embedding = get_embedding(query)
    embedding_str   = str(query_embedding)

    filters       = []
    filter_params = []

    if location:
        filters.append("location ILIKE %s")
        filter_params.append(f"%{location}%")
    if min_salary:
        filters.append("salary_min >= %s")
        filter_params.append(min_salary)

    where_clause = "WHERE " + " AND ".join(filters) if filters else ""

    # First embedding param goes before the WHERE filters; second after (ORDER BY)
    params = [embedding_str] + filter_params + [embedding_str, top_k]

    cursor.execute(f"""
        SELECT title, company, location, salary_min, salary_max,
               1 - (embedding <=> %s::vector) AS similarity
        FROM job_listings
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, params)
```

A filtered search for technical leadership roles in New York with a minimum salary of $130,000:

```text
search_jobs_filtered(
    "technical leadership and system design",
    location = "New York",
    min_salary = 130000
)

query = 'technical leadership and system design', location = 'New York', min_salary = $130,000

  Engineering Manager @ GrowthLab - New York, NY
  $175,000 - $215,000 | Similarity: 0.449

  Engineering Manager @ BridgeIT - New York, NY
  $175,000 - $215,000 | Similarity: 0.449

  Site Reliability Engineer @ DocuCraft - New York, NY
  $140,000 - $175,000 | Similarity: 0.336

  Search Engineer @ Orbis Cloud - New York, NY
  $135,000 - $165,000 | Similarity: 0.294

  NLP Engineer @ Orbis Cloud - New York, NY
  $135,000 - $170,000 | Similarity: 0.272
```

The SQL filter and the vector similarity operate together in a single round trip to the database. There is no second system to query, no results to merge and no metadata to synchronize.

## What You'd Hit in Production

**Index build time.** `HNSW` indexes are built at insert time, which slows bulk loads considerably. For large datasets, always load data first and create the index afterwards. The difference can be an order of magnitude.

**Dimensionality limit.** `pgvector` supports up to 2,000 dimensions. Most embedding models are well within this - `all-minilm` produces 384 dimensions, OpenAI's `text-embedding-3-small` produces 1,536. The exception is `text-embedding-3-large` at 3,072 dimensions, which requires dimensionality reduction before storage.

**Approximate vs exact search.** `HNSW` is an approximate nearest neighbor algorithm. For the vast majority of applications this is fine - recall is high and the speed improvement over exact search is substantial. If you need guaranteed exact results, omit the index and use a sequential scan, but be aware this does not scale beyond a few hundred thousand vectors.

**Connection pooling.** Vector queries are memory-intensive, particularly at higher dimensions. In production, use `PgBouncer` or a connection pooler to avoid connection exhaustion under load.

## When to Look Elsewhere

`pgvector` is a strong default, but it is not the right choice for every situation. Consider a dedicated vector database if:

- You are storing tens of millions of vectors and need sub-10ms retrieval at scale. Dedicated vector databases are built around this problem in a way that a general-purpose database with an extension is not.
- You need advanced filtering on high-cardinality metadata without index performance trade-offs. `pgvector`'s `HNSW` index applies the vector search first and filters afterwards, which can degrade precision when filters are highly selective.
- Your team has no existing Postgres footprint and no appetite for managing it. The operational simplicity argument only holds if Postgres is already in your stack.
- You need built-in embedding model integrations, multi-tenancy, namespacing or other features that dedicated vector databases provide out of the box.

For most applications - particularly those already running on Postgres - `pgvector` is the right place to start. It is free, battle-tested and keeps your architecture simple. The chapters that follow explore what you gain by moving to a dedicated vector database and when that trade-off is worth making.
