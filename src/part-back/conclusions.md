# Conclusions

## What We Built

Over seven days we built seven working applications, each demonstrating vector similarity search in a different database against a different use case. We searched job listings, found recipes, searched electronics products, discovered research papers, detected fraud, analyzed support tickets and retrieved internal documents. Every example used the same embedding model, the same general chapter structure and the same honest framing: here is what this database is genuinely good at and here is when you should look elsewhere.

The notebooks are real. The gotchas are real. The "when to look elsewhere" sections are meant. This is not a marketing guide.

## What We Learned

### The right database depends on where your data already lives

The most consistent theme across all seven chapters is data gravity. The best vector database for your use case is often the one that already holds your data.

If your application runs on Postgres, `pgvector` is one command away. If your data is in MongoDB, Atlas Vector Search is a field on your existing documents. If your analytics team lives in Snowflake or Databricks, `VECTOR_COSINE_SIMILARITY` is a SQL function call. Adding a dedicated vector database to a stack that already has one of these systems means synchronization, dual writes and operational overhead. That overhead is only justified if the dedicated system offers something genuinely unavailable in what you already have.

### Purpose-built databases earn their place at scale and in specific scenarios

Pinecone's managed simplicity is real. Weaviate's hybrid search - the ability to blend BM25 keyword matching with vector similarity in a single `alpha` parameter - is genuinely differentiated for knowledge-heavy retrieval. Neo4j's ability to combine graph traversal with vector search in a single Cypher query is something no other database in this book can match.

These capabilities matter when they are the right fit. Pure-play vector search at hundreds of millions of vectors, hybrid search over technical content where exact terminology matters, fraud ring detection where relationships between entities are the signal - these are the scenarios where a purpose-built system earns its complexity.

### Vector search and relational filtering are natural partners

Across all seven chapters, the most compelling demonstrations were not the pure vector searches but the filtered ones. Semantic similarity combined with structured constraints - category, salary, publication year, priority, fraud status - is more useful than either alone. Every database in this book handles this combination, but in different ways and with different trade-offs.

Postgres and SQL-native systems handle it most naturally because the filter is just a `WHERE` clause. Pinecone and Weaviate use pre-filtering, which applies constraints before the vector search and produces better results when filters are selective. MongoDB applies filters within the `$vectorSearch` aggregation stage. Understanding which approach a database uses matters when your filters are highly selective.

### Gotchas are part of the story

Every chapter encountered real friction. `pgvector` had to be installed from source on Apple Silicon. MongoDB Atlas reports an index as ready before queries will actually return results. Weaviate's free tier supports only the `hfresh` index type, not HNSW. Neo4j's `db.index.vector.queryNodes()` procedure is deprecated but its replacement is not yet available on AuraDB Free. Snowflake Cortex Search is not available on trial accounts. Databricks Community Edition uses `workspace` as the default catalog, not `main`.

These are not edge cases. They are the real experience of setting up these systems for the first time and documenting them honestly is one of the things that makes this book useful rather than just promotional.

### pgvector is everywhere

One observation that cuts across the whole book: pgvector has spread far beyond standalone Postgres. It powers the Neon integration that became Databricks Lakebase. It is available via Crunchy Data inside Snowflake. It is the default vector layer in dozens of managed Postgres offerings including Supabase and Neon. The book starts with pgvector on a local Mac install and ends with pgvector running inside the Databricks lakehouse as a fully managed service. The underlying technology is the same. The operational context has changed completely.

## Choosing the Right Database

Rather than a ranking, here is a practical decision guide based on what we learned across the seven chapters:

**Start with what you have.** If you are already running Postgres, MongoDB, Snowflake or Databricks, try adding vector search there first. The operational simplicity argument is strong and the capabilities are sufficient for most use cases.

**Reach for Pinecone when simplicity is the priority.** If your use case is pure similarity search with metadata filtering and you want zero infrastructure overhead, Pinecone's managed service is hard to beat. The data model is minimal by design.

**Reach for Weaviate when keywords matter alongside meaning.** If your users search for specific technical terms as well as concepts, hybrid search with a tunable `alpha` parameter is a meaningful capability. Weaviate is also the open-source option if self-hosting matters.

**Reach for Neo4j when connections are the signal.** If the interesting questions in your domain involve networks of entities - fraud rings, recommendation graphs, knowledge graphs, supply chains - graph traversal combined with vector search is a capability that no relational or document database can replicate naturally.

**Stay in the data platform when your data is already there.** For Snowflake and Databricks users, `VECTOR_COSINE_SIMILARITY` brings semantic search to existing tables without new infrastructure. The full-scan limitation is real but manageable at the scales where teams are typically starting out.

## What Comes Next

The vector database landscape is moving quickly. Several things will have changed by the time you read this:

**Cortex Search** will likely become available on more Snowflake account tiers, removing the trial account limitation we encountered on Day 6.

**Databricks Lakebase** is actively developing. The Neon acquisition has produced a fully managed Postgres with pgvector integrated into the lakehouse. How deeply it integrates with Delta Lake and Unity Catalog will shape whether it becomes the default choice for Databricks teams that need vector search.

**Embedding models** are improving rapidly. The `all-minilm` model we used throughout this book is fast and free but not state of the art. Production systems should evaluate embedding quality as carefully as they evaluate the database.

**Approximate nearest neighbor algorithms** continue to improve. HNSW is currently the default for most systems, but research into better index structures is active and the landscape may shift.

The databases themselves will also evolve. Weaviate's free tier may gain HNSW support. Neo4j's `VECTOR SEARCH` Cypher syntax will reach AuraDB Free. Pinecone will add capabilities. MongoDB will refine its pre-filtering model. The specific gotchas in this book will become outdated; the underlying evaluation framework will not.

## A Final Note

The goal of this book was not to pick a winner. It was to give you enough honest, hands-on experience with seven different approaches that you can make a good decision for your own situation. Vector search is not a feature you bolt on - it is an architectural choice that affects your data model, your query patterns, your operational overhead and your costs.

The best database for your use case is the one that fits your data, your team and your scale. We hope this book has made that choice a little clearer.
