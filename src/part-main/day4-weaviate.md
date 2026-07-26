# Day 4: Weaviate

## What Is It?

Weaviate is an open-source vector database with a managed cloud offering. Unlike the pure-play managed approach of Day 3, Weaviate can be self-hosted or run as a fully managed service on Weaviate Cloud. Its defining feature is hybrid search - the ability to combine vector similarity, BM25 keyword matching and metadata filtering in a single query, without needing to orchestrate separate systems.

Weaviate organizes data in **collections**. Each collection has a defined schema with typed properties - text, integers, arrays and more. This schema-aware approach means data are validated at write time and queries can take advantage of strong typing when filtering. Vectors are stored alongside the document properties on the same object, so there is no separate metadata store to maintain.

## When Would You Reach for It?

The clearest signal is a hybrid search requirement. If your use case benefits from both the semantic meaning of a query and the presence of specific technical terms - and most knowledge-heavy retrieval tasks do - Weaviate handles this natively through a single query parameter called `alpha`. At `alpha = 0`, queries are pure BM25 keyword search. At `alpha = 1`, queries are pure vector search. Values in between blend the two in proportion. No separate indexes, no result merging in application code.

The second signal is rich structured metadata alongside free-text content. Research papers, legal documents, product manuals and support tickets all have structured fields, such as dates, categories, authors, ratings, that users want to filter on alongside semantic search. In Weaviate, any property defined in the collection schema can be used as a filter at query time without any additional index declaration.

The third signal is open-source flexibility. Weaviate is Apache 2.0 licensed and can be self-hosted on your own infrastructure. For teams with data residency requirements or a preference for running their own stack, this is a meaningful advantage over fully managed proprietary options.

## The Use Case

For this chapter we'll build a research paper discovery system. Users describe their topic of interest in natural language - "transformer models for natural language understanding" or "reinforcement learning for robotic manipulation" - and get relevant papers in return. Filters let users narrow results by research field, publication year or citation count.

This is a natural fit for Weaviate. Research papers have rich structured metadata - field, venue, year, citation count, authors - alongside a free-text abstract that carries the semantic meaning. Users searching for papers often include specific technical terms ("BM25", "HNSW", "transformer") that benefit from keyword matching, while the overall intent of the query benefits from semantic search. Hybrid search handles both simultaneously.

## The Data

We'll generate research papers programmatically from pools of fields, venues, author names and abstract templates. Each abstract is assembled from domain-appropriate components - methods, datasets and contribution statements - giving enough variation for meaningful hybrid search across hundreds of papers.

The key fields are:

- `title` - the paper title, e.g. "Efficient Transformer Models for Natural Language Processing"
- `authors` - a list of author names of varying length
- `year` - publication year between 2015 and 2024
- `field` - research field, e.g. "Natural Language Processing", "Computer Vision"
- `venue` - conference or journal, e.g. "International Conference on Machine Learning Systems", "Journal of Advanced Artificial Intelligence"
- `citation_count` - an integer between 0 and 500
- `abstract` - a short prose summary assembled from templates; this is what we embed and search over

## Building the Application

### Prerequisites

To follow along you'll need:

- A Weaviate Cloud account
- Ollama running locally with the `all-minilm` model pulled
- Python 3.12 with a virtual environment

### Create a Free Weaviate Cloud Cluster

If you do not already have a cluster:

1. Go to [Weaviate Console](https://console.weaviate.cloud) and create an account
2. Click **Create new cluster > Free**
3. Give the cluster a name (e.g. `papers`)
4. Accept all the other default options and and click **Create cluster**
5. Click the **How to connect** button and note down the `WEAVIATE_URL`
6. From the cluster page select **API Keys**, create a new **Admin** key and note it down

### Configuration

We'll set the following Weaviate environment variables before running the notebook:

```bash
export WEAVIATE_URL="your-cluster-name.c0.region.cloud-provider.weaviate.cloud"
export WEAVIATE_API_KEY="your-api-key"
```

Then in the notebook:

```python
WEAVIATE_URL     = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
COLLECTION_NAME  = "ResearchPaper"
LLM_EMBEDDING    = "all-minilm"
NUM_PAPERS       = 200
RANDOM_SEED      = 42
```

### Determine Embedding Dimensions

We'll determine the embedding dimensions dynamically from a test embedding:

```python
def get_embedding(text: str) -> list:
    response = ollama.embeddings(model = LLM_EMBEDDING, prompt = text)
    return response["embedding"]

test_embedding = get_embedding("transformer architecture for natural language processing")
EMBEDDING_DIMS = len(test_embedding)
print(f"Embedding dimensions: {EMBEDDING_DIMS}")
```

### Connect to Weaviate Cloud

```python
client = weaviate.connect_to_weaviate_cloud(
    cluster_url      = WEAVIATE_URL,
    auth_credentials = Auth.api_key(WEAVIATE_API_KEY)
)

print(f"Connected to Weaviate: {client.is_ready()}")
```

### Generate the Dataset

We'll assemble papers from component pools. Each abstract is built from a template parameterized with a field-appropriate method, dataset and contribution statement:

```python
ABSTRACT_TEMPLATES = [
    "We present a new approach to {field} using {method}. Our method addresses key limitations "
    "of prior work by introducing a novel architecture trained on {dataset}. "
    "Experiments demonstrate {contribution} standard benchmarks, with ablation studies "
    "confirming the importance of each component.",
    # ... further templates
]

def generate_paper() -> dict:
    field   = random.choice(FIELDS)
    method  = random.choice(METHODS[field])
    dataset = random.choice(DATASETS[field])

    title = random.choice(TITLE_TEMPLATES).format(
        Method = method.title(),
        Field  = field,
    )

    abstract = random.choice(ABSTRACT_TEMPLATES).format(
        field        = field,
        method       = method,
        dataset      = dataset,
        contribution = random.choice(CONTRIBUTIONS),
    )

    return {
        "title":          title,
        "authors":        generate_authors(),
        "year":           random.randint(2015, 2024),
        "field":          field,
        "venue":          random.choice(VENUES),
        "citation_count": random.randint(0, 500),
        "abstract":       abstract,
    }

papers = [generate_paper() for _ in range(NUM_PAPERS)]
```

### Create the Weaviate Collection

Weaviate organizes data in **collections**. Each collection has a defined schema with typed properties. We'll use `Configure.Vectors.self_provided()` since we are supplying our own embeddings from Ollama rather than using one of Weaviate's built-in vectorizers.

> **Note** The Weaviate Cloud free tier only supports the `hfresh` vector index type. `HNSW` is not available on the free tier. Use `Configure.VectorIndex.hfresh()` rather than `Configure.VectorIndex.hnsw()`.

> **Note:** The free tier allows only one collection per cluster. If the collection already exists from a previous run, delete it before recreating:

```python
if client.collections.exists(COLLECTION_NAME):
    client.collections.delete(COLLECTION_NAME)

collection = client.collections.create(
    name = COLLECTION_NAME,
    vector_config = Configure.Vectors.self_provided(
        vector_index_config = Configure.VectorIndex.hfresh(
            distance_metric = wvc.config.VectorDistances.COSINE
        )
    ),
    properties = [
        Property(name = "title",          data_type = DataType.TEXT),
        Property(name = "authors",        data_type = DataType.TEXT_ARRAY),
        Property(name = "year",           data_type = DataType.INT),
        Property(name = "field",          data_type = DataType.TEXT),
        Property(name = "venue",          data_type = DataType.TEXT),
        Property(name = "citation_count", data_type = DataType.INT),
        Property(name = "abstract",       data_type = DataType.TEXT),
    ]
)

print(f"Collection '{COLLECTION_NAME}' created.")
```

### Generate Embeddings and Load Data

We'll embed each abstract and insert papers using Weaviate's batch context manager. The `batch.dynamic()` mode automatically adjusts batch size based on server response times:

```python
collection = client.collections.get(COLLECTION_NAME)

with collection.batch.dynamic() as batch:
    for paper in tqdm(papers, desc = "Inserting papers"):
        embedding = get_embedding(paper["abstract"])
        batch.add_object(
            properties = {
                "title":          paper["title"],
                "authors":        paper["authors"],
                "year":           paper["year"],
                "field":          paper["field"],
                "venue":          paper["venue"],
                "citation_count": paper["citation_count"],
                "abstract":       paper["abstract"],
            },
            vector = embedding
        )

print(f"\nInserted {collection.aggregate.over_all().total_count} papers.")
```

### Semantic Search

We'll start with pure vector search using `collection.query.near_vector()` to establish a baseline before introducing hybrid search. The `MetadataQuery(distance = True)` parameter returns the cosine distance alongside each result:

```python
def search_papers(query: str, top_k: int = 5):
    query_embedding = get_embedding(query)

    results = collection.query.near_vector(
        near_vector     = query_embedding,
        limit           = top_k,
        return_metadata = MetadataQuery(distance = True),
    )

    print(f"\nQuery: '{query}'\n")
    for obj in results.objects:
        p = obj.properties
        print(f"  {p['title']}")
        print(f"  {p['field']} | {p['venue']} | {p['year']} | Citations: {p['citation_count']}")
        print(f"  Authors: {', '.join(p['authors'])}")
        print(f"  Distance: {obj.metadata.distance:.3f}")
        print()
```

A sample result:

```text
search_papers("transformer models for natural language understanding")

Query: 'transformer models for natural language understanding'

  Efficient Transformer Models for Natural Language Processing
  Natural Language Processing | Transactions on Machine Learning and Data Mining | 2020 | Citations: 196
  Authors: Sarah Kumar, Ahmed Brown
  Distance: 0.241

  Efficient Transformer Models for Natural Language Processing
  Natural Language Processing | Symposium on Knowledge Discovery and Data Mining | 2022 | Citations: 18
  Authors: Chen Kumar, Fatima Kim, Fatima Li, Fatima Smith, Carlos Kim
  Distance: 0.272
```

### Hybrid Search

Hybrid search combines vector similarity with BM25 keyword matching using the `alpha` parameter. We'll use `HybridFusion.RELATIVE_SCORE` as the fusion type, which normalizes scores from both searches before combining them:

```python
def hybrid_search_papers(query: str, alpha: float = 0.5, top_k: int = 5):
    query_embedding = get_embedding(query)

    results = collection.query.hybrid(
        query        = query,
        vector       = query_embedding,
        alpha        = alpha,
        limit        = top_k,
        fusion_type  = HybridFusion.RELATIVE_SCORE,
        return_metadata = MetadataQuery(score = True),
    )

    print(f"\nHybrid search: '{query}' (alpha = {alpha})\n")
    for obj in results.objects:
        p = obj.properties
        print(f"  {p['title']}")
        print(f"  {p['field']} | {p['venue']} | {p['year']} | Citations: {p['citation_count']}")
        print(f"  Score: {obj.metadata.score:.3f}")
        print()
```

### Comparing Alpha Values

Running the same query at `alpha = 0.0`, `0.5` and `1.0` illustrates how the blend shifts results. At `alpha = 0.0` (pure BM25), results are ranked by keyword overlap with the query terms. At `alpha = 1.0` (pure vector), results are ranked by semantic similarity. At `alpha = 0.5`, the two signals are blended using relative score normalization:

```text
query = "large language models for question answering"

alpha = 0.0 (pure BM25)
  Towards Better Natural Language Processing via Question Answering
  Natural Language Processing | Journal of Advanced Artificial Intelligence | 2024 | Citations: 269
  Score: 1.000

alpha = 0.5 (balanced)
  Efficient Question Answering for Natural Language Processing
  Natural Language Processing | Transactions on Machine Learning and Data Mining | 2018 | Citations: 485
  Score: 0.999

alpha = 1.0 (pure vector)
  Efficient Question Answering for Natural Language Processing
  Natural Language Processing | Transactions on Machine Learning and Data Mining | 2018 | Citations: 485
  Score: 1.000
```

For research paper discovery, a balanced blend works well because users often include specific technical terms alongside more general intent.

### Filtered Hybrid Search

Any property defined in the collection schema can be used as a filter - no separate index declaration is needed. Filters are expressed using Weaviate's `Filter` class and chained with the `&` operator:

```python
def hybrid_search_filtered(
    query:         str,
    alpha:         float = 0.5,
    field:         str   = None,
    min_year:      int   = None,
    min_citations: int   = None,
    top_k:         int   = 5
):
    query_embedding = get_embedding(query)

    filters = None
    if field:
        f = Filter.by_property("field").equal(field)
        filters = f if filters is None else filters & f
    if min_year:
        f = Filter.by_property("year").greater_or_equal(min_year)
        filters = f if filters is None else filters & f
    if min_citations:
        f = Filter.by_property("citation_count").greater_or_equal(min_citations)
        filters = f if filters is None else filters & f

    results = collection.query.hybrid(
        query        = query,
        vector       = query_embedding,
        alpha        = alpha,
        limit        = top_k,
        fusion_type  = HybridFusion.RELATIVE_SCORE,
        filters      = filters,
        return_metadata = MetadataQuery(score = True),
    )
```

Natural Language Processing papers from 2020 onwards with at least 100 citations:

```text
hybrid_search_filtered(
    "attention mechanisms and transformer architectures",
    field         = "Natural Language Processing",
    min_year      = 2020,
    min_citations = 100
)

query = 'attention mechanisms and transformer architectures', alpha = 0.5,
field = 'Natural Language Processing', min_year = 2020, min_citations = 100

  Efficient Transformer Models for Natural Language Processing
  Natural Language Processing | Transactions on Machine Learning and Data Mining | 2020 | Citations: 196
  Score: 0.918

  Transformer Models in Natural Language Processing: Challenges and Opportunities
  Natural Language Processing | International Journal of Deep Learning | 2024 | Citations: 337
  Score: 0.500
```

Recent computer vision papers on object detection:

```text
hybrid_search_filtered(
    "object detection in real time",
    field    = "Computer Vision",
    min_year = 2022
)

query = 'object detection in real time', alpha = 0.5,
field = 'Computer Vision', min_year = 2022

  Rethinking Computer Vision with Object Detection
  Computer Vision | International Journal of Deep Learning | 2024 | Citations: 269
  Score: 1.000

  Towards Better Computer Vision via Vision Transformers
  Computer Vision | Conference on Intelligent Data Analysis | 2022 | Citations: 304
  Score: 0.440
```

## What You'd Hit in Production

**Free tier limitations.** The Weaviate Cloud free tier supports one collection per cluster and only the `hfresh` vector index type. `HNSW`, which offers better recall at scale, requires a paid tier. For production workloads with more than a few hundred thousand vectors, upgrade to a paid plan.

**hfresh vs HNSW.** The `hfresh` index type is optimized for fresh data and low-latency insertions. `HNSW` offers better approximate nearest neighbor recall at large scale. On the free tier you'll not notice a difference at a few hundred papers, but it's worth understanding the distinction before moving to production.

**Alpha tuning.** The right `alpha` value depends on the data and query patterns. For queries with specific technical terms, a lower alpha (more BM25) improves recall of papers that contain those exact terms. For more exploratory queries, a higher alpha (more vector) surfaces semantically related content even when the exact terms are absent. Experiment with your own queries to find the right balance.

**Collection limit on free tier.** The free tier allows only one collection per cluster. If you need to reload data, delete the existing collection first using `client.collections.delete(COLLECTION_NAME)` before recreating it. Deleting a Weaviate collection removes everything including the schema.

**Batch insert behavior.** Weaviate's `batch.dynamic()` mode adjusts the batch size automatically based on server response times. For large datasets this is more efficient than fixed-size batches. Errors during batch insert are collected rather than raised immediately - check `batch.failed_objects` after the context manager exits to catch any insertion failures.

**Self-hosting.** Weaviate can be run locally via Docker if you prefer not to use the managed cloud service. The self-hosted version supports `HNSW` on all tiers and has no collection limits. This is worth considering for production deployments with data residency requirements.

## When to Look Elsewhere

Weaviate is a strong choice for hybrid search and knowledge-heavy retrieval. Consider alternatives if:

- You have no need for hybrid search and your queries are purely semantic. Days 1 through 3 all handle pure vector search well, often more simply.
- You need the absolute highest vector search performance at hundreds of millions of vectors. Weaviate is performant but purpose-built vector databases like Pinecone are optimized for that ceiling.
- You already have data in PostgreSQL or MongoDB. Adding a second database introduces synchronization complexity. Days 1 and 2 showed that both can handle vector search natively alongside your existing data.
- Your data have no meaningful keyword signal. For use cases where queries are entirely conceptual and no specific technical terms matter, the BM25 component adds little value and a simpler pure-vector setup is cleaner.

For use cases where meaning and keywords both matter - research discovery, legal document search, technical support retrieval - Weaviate's hybrid search is genuinely differentiated. The ability to tune the blend between keyword and semantic relevance in a single parameter, without building a separate retrieval pipeline, is a meaningful advantage for knowledge-intensive applications.
