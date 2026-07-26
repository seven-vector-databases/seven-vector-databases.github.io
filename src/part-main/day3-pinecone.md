# Day 3: Pinecone

## What Is It?

Pinecone is a fully managed, purpose-built vector database. Unlike Day 1 and Day 2, where vector search was an extension or add-on to an existing database, Pinecone exists solely to store and search vectors. There is no relational layer, no document model and no infrastructure to manage - just an API.

This is a deliberate design choice. Pinecone's thesis is that vector search is a distinct enough problem to warrant its own dedicated system, and that the operational overhead of running and tuning a general-purpose database is unnecessary friction when all you need is similarity search. You provision an index, upsert vectors, query them and let Pinecone handle everything else.

Pinecone organizes vectors in **indexes**. Each index stores vectors of a fixed dimension alongside optional **metadata** - structured fields like category, price or brand that can be used for filtering at query time. There are no tables, no collections and no schemas. The data model is intentionally minimal: an `id`, a vector and a metadata dictionary.

## When Would You Reach for It?

The clearest signal is a pure vector search use case with no need for a broader data model. If your application needs to find similar items - products, documents, images, user profiles - and the structured filtering you need can be expressed as metadata on the vector record, Pinecone is a strong fit.

The second signal is managed simplicity. Pinecone requires no infrastructure decisions, no index tuning and no operational runbook. You do not choose instance sizes, manage disk or worry about replication. For teams that want vector search without a dedicated infrastructure engineer, this is the value proposition.

The third signal is scale. Pinecone is built to handle hundreds of millions of vectors with consistent low-latency retrieval. If your use case will grow beyond what a general-purpose database with a vector extension can handle, Pinecone is designed for that ceiling.

## The Use Case

For this chapter we'll build a semantic product search engine for an electronics store. Users describe what they are looking for in natural language - "a laptop for video editing under $1,500" or "noise canceling headphones for travel" - and get relevant products in return.

This is a natural fit for Pinecone. Product search is a pure similarity problem: given a query, find the most semantically relevant items. The structured constraints a user might apply - category, price ceiling, performance tier - map naturally to Pinecone's metadata filtering. There is no relational data to join, no document hierarchy to navigate and no schema to maintain.

## The Data

We generate electronics product listings programmatically from pools of categories, brands, performance tiers, use cases and description templates. Each product has a name, category, brand, price, performance tier and a short prose description. The description is what we embed and search over. Everything else is stored as metadata on the vector record for filtering.

The key fields are:

- `name` - the product name, e.g. "VisionCore Laptop 576"
- `category` - e.g. "Laptop", "Headphones", "Camera"
- `brand` - one of fifteen fictional brands
- `price` - a realistic price for the category in US dollars
- `performance` - "entry-level", "mid-range", "high-performance", "professional-grade" or "flagship"
- `use_case` - the intended audience, e.g. "remote workers", "content creators", "travelers"
- `description` - a short prose summary assembled from templates; this is what we embed

## Building the Application

### Prerequisites

To follow along you will need:

- A Pinecone account with an API key
- Ollama running locally with the `all-minilm` model pulled
- Python 3.12 with a virtual environment

### Configuration

We'll set the `PINECONE_API_KEY` environment variable before running the notebook:

```bash
export PINECONE_API_KEY="your-api-key-here"
```

Then in the notebook:

```python
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = "electronics"
CLOUD            = "aws"
REGION           = "us-east-1"
LLM_EMBEDDING    = "all-minilm"
NUM_PRODUCTS     = 200
RANDOM_SEED      = 42
```

> **Note:** `NUM_PRODUCTS` controls the size of the generated dataset. 200 is the recommended default for this chapter - embedding generation runs locally via Ollama and is single-threaded, so larger values will work but will take proportionally longer. Production pipelines would typically use a hosted embedding endpoint with async or batched generation to handle scale.

### Determine Embedding Dimensions

We'll determine the embedding dimensions dynamically from a test embedding:

```python
def get_embedding(text: str) -> list:
    response = ollama.embeddings(model = LLM_EMBEDDING, prompt = text)
    return response["embedding"]

test_embedding = get_embedding("noise canceling headphones for travel")
EMBEDDING_DIMS = len(test_embedding)
print(f"Embedding dimensions: {EMBEDDING_DIMS}")
```

### Connect to Pinecone

```python
pc = Pinecone(api_key = PINECONE_API_KEY)
print("Connected to Pinecone.")
```

### Generate the Dataset

We assemble products from component pools. Description templates are parameterized to produce varied prose across all 200 products:

```python
DESCRIPTION_TEMPLATES = [
    "A {performance} {category} designed for {use_case}, featuring {feature1} and {feature2}.",
    "Built for {use_case}, this {performance} {category} delivers {feature1} alongside {feature2}.",
    "The {brand} {category} is a {performance} option for {use_case}, offering {feature1} and {feature2}.",
    # ... further templates
]

def generate_product() -> dict:
    category    = random.choice(CATEGORIES)
    brand       = random.choice(BRANDS)
    performance = random.choice(PERFORMANCE)
    use_case    = random.choice(USE_CASES)
    features    = random.sample(FEATURES[category], 2)
    price_min, price_max = PRICE_BANDS[category]
    price       = round(random.randint(price_min, price_max) / 10) * 10

    description = random.choice(DESCRIPTION_TEMPLATES).format(
        category    = category,
        brand       = brand,
        performance = performance,
        use_case    = use_case,
        feature1    = features[0],
        feature2    = features[1],
    )

    return {
        "name":        f"{brand} {category} {random.randint(100, 999)}",
        "category":    category,
        "brand":       brand,
        "price":       price,
        "performance": performance,
        "use_case":    use_case,
        "description": description,
    }

products = [generate_product() for _ in range(NUM_PRODUCTS)]
```

### Create the Pinecone Index

We create a serverless index programmatically. If the index already exists from a previous run, we delete it and recreate it for a clean start. The `EMBEDDING_DIMS` variable ensures the index dimension matches the embedding model.

```python
existing_indexes = [idx.name for idx in pc.list_indexes()]

if INDEX_NAME in existing_indexes:
    print(f"Deleting existing index '{INDEX_NAME}'...")
    pc.delete_index(INDEX_NAME)

print(f"Creating index '{INDEX_NAME}'...")
pc.create_index(
    name      = INDEX_NAME,
    dimension = EMBEDDING_DIMS,
    metric    = "cosine",
    spec      = ServerlessSpec(cloud = CLOUD, region = REGION)
)
```

We then poll until the index is ready before proceeding:

```python
print("Waiting for index to be ready...")
while True:
    status = pc.describe_index(INDEX_NAME).status
    if status.get("ready"):
        print(f"Index '{INDEX_NAME}' is ready.")
        break
    print(f"  Status: {status} - waiting...")
    time.sleep(5)

index = pc.Index(INDEX_NAME)
```

### Generate Embeddings and Load Data

In Pinecone, each record consists of three parts: an `id`, a `values` list (the embedding vector) and a `metadata` dictionary. The metadata holds all the structured fields we want to filter on or display in search results.

We generate embeddings and upsert records in batches of 50. Pinecone's `upsert` operation inserts new records and updates existing ones if the `id` already exists.

```python
BATCH_SIZE = 50

records = []
for i, product in enumerate(tqdm(products, desc = "Generating embeddings")):
    embedding = get_embedding(product["description"])
    records.append({
        "id":     str(i),
        "values": embedding,
        "metadata": {
            "name":        product["name"],
            "category":    product["category"],
            "brand":       product["brand"],
            "price":       product["price"],
            "performance": product["performance"],
            "use_case":    product["use_case"],
            "description": product["description"],
        }
    })

for i in range(0, len(records), BATCH_SIZE):
    index.upsert(vectors = records[i:i + BATCH_SIZE])
```

After upserting we verify the index using `describe_index_stats`:

```python
stats = index.describe_index_stats()
print(f"Total vectors in index: {stats['total_vector_count']}")
print(f"Dimensions: {stats['dimension']}")
```

### Semantic Search

We embed the user's query and pass it to Pinecone's `query` method. The `include_metadata = True` parameter ensures the structured fields come back with each result alongside the similarity score.

```python
def search_products(query: str, top_k: int = 5):
    query_embedding = get_embedding(query)

    results = index.query(
        vector           = query_embedding,
        top_k            = top_k,
        include_metadata = True
    )

    print(f"\nQuery: '{query}'\n")
    for match in results["matches"]:
        m = match["metadata"]
        print(f"  {m['name']} ({m['category']})")
        print(f"  ${m['price']:,.0f} | {m['performance']} | Score: {match['score']:.3f}")
        print(f"  {m['description']}")
        print()
```

Running an example query:

```text
search_products("noise canceling headphones for travel")

Query: 'noise canceling headphones for travel'

  CloudSync Headphones 979 (Headphones)
  $50 | mid-range | Score: 0.732
  The CloudSync Headphones is a mid-range option for travelers, offering wireless connectivity and foldable design.

  VisionCore Headphones 777 (Headphones)
  $180 | high-performance | Score: 0.721
  A high-performance Headphones with active noise cancellation and wireless connectivity, perfect for home entertainment.
```

### Filtered Search

Pinecone supports metadata filtering at query time using a MongoDB-style filter syntax. Filters are applied before the vector search, meaning only matching records are considered as candidates. This is the same pre-filtering approach we saw in MongoDB Atlas on Day 2.

```python
def search_products_filtered(
    query:       str,
    category:    str   = None,
    max_price:   float = None,
    performance: str   = None,
    top_k:       int   = 5
):
    query_embedding = get_embedding(query)

    filter_doc = {}
    if category:
        filter_doc["category"]    = {"$eq": category}
    if max_price:
        filter_doc["price"]       = {"$lte": max_price}
    if performance:
        filter_doc["performance"] = {"$eq": performance}

    results = index.query(
        vector           = query_embedding,
        top_k            = top_k,
        include_metadata = True,
        filter           = filter_doc if filter_doc else None
    )
```

A filtered search for high-performance laptops for creative work:

```text
search_products_filtered(
    "laptop for video editing and creative work",
    category    = "Laptop",
    performance = "high-performance"
)

query = 'laptop for video editing and creative work', category = 'Laptop', performance = 'high-performance'

  NexaDisplay Laptop 997 (Laptop)
  $1,030 | high-performance | Score: 0.583
  A high-performance Laptop with fast SSD storage and long battery life, perfect for content creators.

  NovaByte Laptop 710 (Laptop)
  $680 | high-performance | Score: 0.561
  A high-performance Laptop designed for students, featuring high-resolution display and fast SSD storage.
```

And a cross-category search with a strict price cap:

```text
search_products_filtered(
    "portable device for travel",
    max_price = 300
)

query = 'portable device for travel', max_price = $300

  SwiftTech Tablet 259 (Tablet)
  $280 | high-performance | Score: 0.580
  A high-performance Tablet designed for travelers, featuring detachable keyboard and cellular connectivity.

  NovaByte E-Reader 820 (E-Reader)
  $300 | mid-range | Score: 0.529
  A mid-range E-Reader with weeks of battery life and waterproof design, perfect for travelers.
```

## What You'd Hit in Production

**Metadata filtering limitations.** Pinecone's metadata filters work well for low-cardinality fields like category or performance tier. For high-cardinality fields such as free-text tags or arbitrary user-defined attributes, the filtering model can become cumbersome. Plan your metadata schema upfront.

**Metadata storage costs.** Every piece of metadata stored alongside a vector counts toward your storage usage. For large datasets with rich metadata, this can add up. Store only what you need for filtering and display.

**No joins or aggregations.** Pinecone is not a general-purpose database. There is no way to group results, compute aggregates or join across indexes. If your use case requires these, you will need a second system alongside Pinecone.

**Index deletion on recreation.** Deleting a Pinecone index removes everything. If you need to reload data, you either upsert into the existing index or delete and recreate it. The `upsert` approach is preferable for production since it avoids downtime.

**Serverless vs pod-based indexes.** The free tier uses serverless indexes, which are optimized for variable workloads and scale to zero when idle. Pod-based indexes offer more predictable latency for high-throughput production workloads but come at a fixed cost. For most use cases serverless is the right starting point.

**Free tier limits.** The free tier allows one project with up to five serverless indexes and a total of 2GB of storage. This is sufficient for development and small production workloads.

## When to Look Elsewhere

Pinecone is an excellent choice for pure vector search with metadata filtering. Consider alternatives if:

- You already have data in PostgreSQL or MongoDB. Adding a dedicated vector database introduces a second system to operate and a synchronization problem to solve. Days 1 and 2 showed that both can handle vector search natively.
- You need relational queries, aggregations or joins. Pinecone has no query language beyond vector search and metadata filtering.
- You need full control over your infrastructure. Pinecone is fully managed and closed-source. If data residency, self-hosting or auditability are requirements, a self-hosted alternative may be a better fit.
- Your metadata filtering requirements are complex. For highly selective filters on many fields, the pre-filtering approach can return too few candidates, degrading recall. Dedicated filtering systems handle this more gracefully.

For teams that want managed simplicity and are building a use case that is genuinely about similarity search, Pinecone is hard to beat. The operational overhead is close to zero and the API is clean and well-documented. The question is whether your use case is pure enough to justify a dedicated system, or whether an existing database with vector support is the simpler path.
