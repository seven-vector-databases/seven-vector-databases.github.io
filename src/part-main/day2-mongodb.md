# Day 2: MongoDB Atlas

## What Is It?

MongoDB is the world's most popular document database. Rather than storing data in rows and columns, MongoDB stores it as documents - flexible, JSON-like structures that can hold nested objects, arrays and variably-shaped data without a fixed schema. This makes it a natural fit for content that does not conform cleanly to a relational table, such as product catalogs with varying attributes, user profiles with optional fields or recipes with ingredient lists of different lengths.

MongoDB Atlas is the fully managed cloud version of MongoDB, available on AWS, GCP and Microsoft Azure. Atlas Vector Search extends Atlas with native vector similarity search, letting us store and query embeddings directly alongside our existing document data. There is no separate vector store to maintain, no synchronization between systems and no new operational model to learn.

## When Would You Reach for It?

The clearest signal is an existing MongoDB footprint. If an application already stores its data as documents in MongoDB, Atlas Vector Search is the path of least resistance - vector search becomes a field on existing documents rather than a reason to introduce a new database.

The second signal is variably structured content. Documents whose shape varies from record to record - recipes with different numbers of ingredients, products with different attribute sets, support tickets with optional fields - fit the document model naturally. In a relational database, this kind of variation requires either nullable columns, separate tables or JSON columns. In MongoDB it is just how documents work.

The third signal is the aggregation pipeline. MongoDB's query model is built around composable pipeline stages. The `$vectorSearch` stage plugs into this pipeline naturally, meaning we can chain vector search with filtering, grouping, lookup joins and projection in a single query. Developers already familiar with MongoDB's aggregation model will find Atlas Vector Search intuitive to work with.

## The Use Case

For this chapter we'll build a semantic recipe finder. Users describe what they feel like eating in natural language - "something warm and spicy for a cold evening" or "a light fresh dish with vegetables for summer" - and get relevant recipes in return.

Recipes are a natural fit for the document model. Each recipe has a name, a cuisine, a difficulty rating, a prep time and an ingredient list - but the ingredient list varies in length from recipe to recipe and the set of attributes that matter differs by dish type. In a relational database this variability requires workarounds. In MongoDB it is simply a document with an array field.

The `description` field on each recipe is a short prose summary of the dish. This is what we'll embed and search over. The structured fields - cuisine, difficulty, prep time - serve as filters.

## The Data

We'll generate recipes programmatically from pools of cuisines, cooking methods, ingredients and description templates. Each description is assembled from role-appropriate components, giving enough variation for meaningful semantic search across hundreds of recipes.

The key fields are:

- `name` - the recipe name, e.g. "Korean Chicken And Green Beans"
- `cuisine` - the cuisine type, e.g. "Korean", "Italian", "Thai"
- `difficulty` - "Easy", "Medium" or "Hard"
- `prep_time` - preparation time in minutes
- `ingredients` - a list of ingredients of varying length
- `description` - a short prose summary of the dish; this is what is embedded

Note that `ingredients` is a list field with no fixed length. This is the document model working as intended - no schema changes are needed to accommodate recipes with five ingredients or fifteen.

## Building the Application

### Prerequisites

To follow along you'll need:

- A MongoDB Atlas account
- Ollama running locally with the `all-minilm` model pulled
- Python 3.12 with a virtual environment

### Create a Free MongoDB Atlas Cluster

If you do not already have a cluster:

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) and create a free account
2. Once logged-in, select **Create a free starter database**
3. Give the cluster a name (e.g. `recipes`)
4. Click **Set it up for me**
5. Copy and save database user credentials
6. Select **Choose a connection method > Drivers > Python**
7. Note down the connection string that looks like `mongodb+srv://<username>:<password>@recipes.xxxxx.mongodb.net/`
8. From the left navigation pane select **SECURITY > Database & Network Access**
9. From the left navigation pane select **NETWORK ACCESS > IP Access List**
10. Add `0.0.0.0/0` for temporary open access during development

### Create the Vector Search Index

The vector search index is created programmatically in the notebook - no manual steps required in the Atlas UI. The index will only become active once data have been inserted and the embeddings are present.

> **Note:** The collection must exist before the vector index can be created. On each run we drop the index first, then clear the documents with `delete_many({})`, before recreating the index programmatically.

### Configuration

We'll set the `MONGODB_URI` environment variable before running the notebook:

```bash
export MONGODB_URI='mongodb+srv://<username>:<password>@<cluster>.mongodb.net'
```

Then in the notebook:

```python
MONGODB_URI   = os.environ["MONGODB_URI"]
DB_NAME       = "recipes_db"
COLLECTION    = "recipes"
INDEX_NAME    = "vector_index"
LLM_EMBEDDING = "all-minilm"
NUM_RECIPES   = 200
RANDOM_SEED   = 42
```

> **Note:** `NUM_RECIPES` controls the size of the generated dataset. 200 is the recommended default for this chapter - embedding generation runs locally via Ollama and is single-threaded, so larger values will work but will take proportionally longer. At 1,000 recipes expect a few minutes, at 20,000 expect significantly longer. Production pipelines would typically use a hosted embedding endpoint with async or batched generation to handle scale.

### Generate the Dataset

We'll assemble recipes from component pools - cuisines, cooking methods, proteins, vegetables, bases and description templates. The description templates are parameterized so each recipe gets a unique prose summary:

```python
DESCRIPTION_TEMPLATES = [
    "{method} {protein} served over {base} with {vegetable}, {flavor} and perfect for {occasion}.",
    "A {flavor} {cuisine} dish featuring {method} {protein} with {vegetable} on a bed of {base}.",
    "{cuisine} classic: {method} {protein} with {vegetable}, {flavor} flavors ideal for {occasion}.",
    # ... further templates
]

def generate_recipe() -> dict:
    cuisine  = random.choice(CUISINES)
    method   = random.choice(COOKING_METHODS)
    protein  = random.choice(PROTEINS)
    veg      = random.choice(VEGETABLES)
    base     = random.choice(BASES)

    name = random.choice(RECIPE_NAME_TEMPLATES).format(
        cuisine = cuisine, method = method.capitalize(),
        protein = protein, vegetable = veg, base = base
    )

    description = random.choice(DESCRIPTION_TEMPLATES).format(
        cuisine = cuisine, method = method, protein = protein,
        vegetable = veg, base = base,
        flavor = random.choice(FLAVORS),
        occasion = random.choice(OCCASIONS),
    )

    return {
        "name":        name.title(),
        "cuisine":     cuisine,
        "difficulty":  random.choice(DIFFICULTIES),
        "prep_time":   random.choice(PREP_TIMES),
        "ingredients": list(set([protein, veg, base] + random.sample(VEGETABLES + PROTEINS, random.randint(2, 9)))),
        "description": description,
    }

recipes = [generate_recipe() for _ in range(NUM_RECIPES)]
```

### Generate Embeddings and Load Data

We'll embed each recipe description and store the vector as an `embedding` field on the document:

```python
def get_embedding(text: str) -> list:
    response = ollama.embeddings(model = LLM_EMBEDDING, prompt = text)
    return response["embedding"]

test_embedding = get_embedding("a warm spicy dish with chicken")
EMBEDDING_DIMS = len(test_embedding)
print(f"Embedding dimensions: {EMBEDDING_DIMS}")

collection.drop_search_index(INDEX_NAME)
collection.delete_many({})

documents = []
for recipe in tqdm(recipes, desc = "Generating embeddings"):
    doc = recipe.copy()
    doc["embedding"] = get_embedding(recipe["description"])
    documents.append(doc)

collection.insert_many(documents)
```

### Create the Vector Index

We create the index programmatically using `SearchIndexModel`. The `EMBEDDING_DIMS` variable ensures the index definition stays in sync with whichever embedding model is in use:

```python
search_index_model = SearchIndexModel(
    definition = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": EMBEDDING_DIMS,
                "similarity": "cosine"
            },
            {
                "type": "filter",
                "path": "cuisine"
            },
            {
                "type": "filter",
                "path": "difficulty"
            },
            {
                "type": "filter",
                "path": "prep_time"
            }
        ]
    },
    name = INDEX_NAME,
    type = "vectorSearch"
)

collection.create_search_index(model = search_index_model)
print(f"Index '{INDEX_NAME}' creation initiated.")
```

### Wait for the Vector Index to be Ready

> **Note:** Atlas Vector Search reports the index status as `READY` before queries will actually return results. A fixed delay is not reliable - a small collection may be ready in five seconds, a larger one may need longer. The most robust approach is to poll the index status and then confirm with a test query.

After inserting data, confirm that the vector index we created is active.

```python
print("Waiting for vector index to be active...")
while True:
    indexes = list(collection.list_search_indexes())
    status  = next((idx["status"] for idx in indexes if idx["name"] == INDEX_NAME), None)
    if status == "READY":
        print(f"Index '{INDEX_NAME}' is active.")
        break
    print(f"  Status: {status} - waiting...")
    time.sleep(5)

# Confirm the index is genuinely ready by running a test query
print("Confirming index is ready...")
while True:
    test = list(collection.aggregate([
        {
            "$vectorSearch": {
                "index":         INDEX_NAME,
                "path":          "embedding",
                "queryVector":   get_embedding("test"),
                "numCandidates": 10,
                "limit":         1,
            }
        }
    ]))
    if test:
        print("Index is ready.")
        break
    print("  Index not yet propagated - waiting...")
    time.sleep(5)
```

The second loop issues a real vector query and waits until it returns a result. This is the most reliable signal that the index is ready for use.

### Semantic Search

Atlas Vector Search uses MongoDB's aggregation pipeline. The `$vectorSearch` stage takes a query vector and returns the most similar documents, ranked by cosine similarity. The `$project` stage controls which fields are returned and adds the similarity score via `$meta: "vectorSearchScore"`.

```python
def search_recipes(query: str, top_k: int = 5):
    query_embedding = get_embedding(query)

    pipeline = [
        {
            "$vectorSearch": {
                "index":         INDEX_NAME,
                "path":          "embedding",
                "queryVector":   query_embedding,
                "numCandidates": top_k * 10,
                "limit":         top_k,
            }
        },
        {
            "$project": {
                "_id":         0,
                "name":        1,
                "cuisine":     1,
                "difficulty":  1,
                "prep_time":   1,
                "description": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        }
    ]

    results = list(collection.aggregate(pipeline))
    print(f"\nQuery: '{query}'\n")
    for r in results:
        print(f"  {r['name']} ({r['cuisine']})")
        print(f"  {r['difficulty']} - {r['prep_time']} mins | Score: {r['score']:.3f}")
        print(f"  {r['description']}")
        print()
```

The `numCandidates` parameter controls how many documents Atlas considers before returning the top `limit` results. A higher value improves recall at the cost of latency. A ratio of 10x is a reasonable starting point.

### The Document Advantage - Filtered Search

Atlas Vector Search supports pre-filtering directly in the `$vectorSearch` stage. Filters are applied before the vector search, meaning only matching documents are considered as candidates. This is more precise than post-filtering and produces better results when filters are selective.

The filter fields must be declared in the index definition. We included `cuisine`, `difficulty` and `prep_time` when we created the index, so all three are available as filters:

```python
def search_recipes_filtered(
    query:      str,
    cuisine:    str = None,
    difficulty: str = None,
    max_time:   int = None,
    top_k:      int = 5
):
    query_embedding = get_embedding(query)

    filter_doc = {}
    if cuisine:
        filter_doc["cuisine"]    = {"$eq": cuisine}
    if difficulty:
        filter_doc["difficulty"] = {"$eq": difficulty}
    if max_time:
        filter_doc["prep_time"]  = {"$lte": max_time}

    vector_search_stage = {
        "$vectorSearch": {
            "index":         INDEX_NAME,
            "path":          "embedding",
            "queryVector":   query_embedding,
            "numCandidates": top_k * 10,
            "limit":         top_k,
        }
    }
    if filter_doc:
        vector_search_stage["$vectorSearch"]["filter"] = filter_doc

    pipeline = [vector_search_stage, {"$project": {...}}]
    results  = list(collection.aggregate(pipeline))
```

Here is an example filtered search - "easy Italian recipes ready in 30 minutes or less":

```text
search_recipes_filtered(
    "a comforting pasta dish",
    cuisine    = "Italian",
    difficulty = "Easy",
    max_time   = 30
)

query = 'a comforting pasta dish', cuisine = 'Italian', difficulty = 'Easy', max_time = 30 mins

  Italian Style Salmon With Rice (Italian)
  Easy - 30 mins | Score: 0.748
  Simple and satisfying - poached salmon tossed with kale and rice, creamy and comforting.
```

### Querying the Document Model

One of MongoDB's strengths is querying nested and array fields directly. We can find all recipes containing a specific ingredient using a standard MongoDB query - no joins, no separate table, no additional index, as follows:

```python
def find_by_ingredient(ingredient: str, limit: int = 5):
    results = collection.find(
        {"ingredients": {"$in": [ingredient]}},
        {"_id": 0, "name": 1, "cuisine": 1, "difficulty": 1, "prep_time": 1, "ingredients": 1}
    ).limit(limit)
```

This works because `ingredients` is a native array field on the document. MongoDB indexes array fields automatically, so this query is efficient even at scale.

## What You'd Hit in Production

**Index propagation delay.** As noted above, Atlas reports the index status as `READY` before queries will return results. Budget for this in any automated pipeline that creates a collection and immediately queries it.

**numCandidates tuning.** The ratio between `numCandidates` and `limit` affects both recall and latency. A ratio of 10x is a reasonable starting point, but high-precision use cases may need higher values. Atlas documentation recommends at least `limit * 10` and no more than 10,000.

**Filter field declaration.** Filter fields must be declared in the index definition at creation time. Adding a new filterable field requires rebuilding the index. Plan your filter fields upfront.

**Free tier limits.** Atlas Free clusters support vector search but are intended primarily for development and testing. A Free cluster supports one vector search index per collection and vector indexes can contain up to 8,192 dimensions. For production deployments, MongoDB recommends using a dedicated cluster (M10 or higher).

**Index management.** The vector search index can be created, updated and dropped entirely from Python using `SearchIndexModel` and `create_search_index()` - no Atlas UI required. This makes the full lifecycle scriptable and repeatable.

## When to Look Elsewhere

MongoDB Atlas is a strong choice when your application already runs on MongoDB or when your data is naturally document-shaped. Consider a dedicated vector database if:

- You have no existing MongoDB footprint and no other reason to run it. The operational simplicity argument only holds if MongoDB is already in your stack.
- You need the highest possible vector search performance at very large scale. Dedicated vector databases are built exclusively around this problem and can outperform a general-purpose database with vector search added on.
- Your data are highly relational with many joins. MongoDB handles references between documents, but deeply relational data is more naturally expressed in a relational database.
- You need advanced vector index configuration beyond what Atlas exposes. Dedicated vector databases offer more granular control over index parameters, distance metrics and quantization.

For teams already building on MongoDB, Atlas Vector Search is the natural path - it keeps the stack simple and lets vector search grow with the application without introducing a second system to operate.
