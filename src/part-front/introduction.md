# Introduction

## Why Vector Databases?

Something changed in software development around 2022. Applications began needing to search not just for exact matches - a user ID, a product SKU, a keyword - but for meaning. "Find me something like this." "What documents are relevant to this question?" "Which past support tickets are similar to this one?" These aren't keyword search problems. They're similarity problems and the data structure that makes similarity search practical at scale is the vector embedding.

A vector embedding is a list of numbers - typically hundreds or thousands of them - that encodes the semantic meaning of a piece of text, an image or any other content. Two pieces of content with similar meanings will have embeddings that are close together in vector space, even if they share no words in common. This property is what makes it possible to ask "find me jobs related to building machine learning models in production" and get back relevant results even when the job listings use entirely different vocabulary.

Vector databases are systems built to store these embeddings and search across them efficiently. Some are purpose-built for the task. Others are general-purpose databases that have added vector search as a capability. The field has expanded rapidly and evaluating the options is now a real problem for teams building AI-powered features.

## What This Book Is About

This book isn't a comparison of vector databases. It won't tell you which one is "best." It won't produce a ranking or a scorecard.

What it'll do is show you what each of seven databases is genuinely good at and when you'd reach for it over the others. Each chapter focuses on a different database and a different use case, chosen specifically because the use case plays to that database's strengths. The goal isn't to pit them against each other but to give you the context to make a good decision when you're evaluating tooling for your own problem.

The seven databases covered are:

| Day | Database | Use Case |
|-----|----------|----------|
| 1 | PostgreSQL + pgvector | Semantic job listing search |
| 2 | MongoDB Atlas | Recipe finder |
| 3 | Pinecone | E-commerce product search |
| 4 | Weaviate | Research paper discovery |
| 5 | Neo4j | Fraud detection |
| 6 | Snowflake | Customer support analytics |
| 7 | Databricks | RAG over internal documents |

The progression is deliberate. We start with the familiar - Postgres, which most developers already have - and move through a document database, two dedicated vector databases, a graph database and, finally, the major data platforms. By the end, you'll have a clear picture of where vector search fits in each architecture and what trade-offs each approach involves.

## Who This Book Is For

This book is written for developers, data engineers and architects who are evaluating vector database options for a real project. It assumes you're comfortable with Python and SQL. It doesn't assume any prior experience with vector databases, embeddings or machine learning.

If you're building a RAG application and wondering whether to use Pinecone, pgvector or something else - this book is for you. If you have data in Snowflake or Databricks and are wondering whether you need a separate vector database - this book is for you. If you're new to vector search and want a practical grounding before making architectural decisions - this book is for you.

## How the Book Is Structured

Each chapter follows the same structure:

- **What is it** - an honest characterization of the database, not a marketing summary
- **When would you reach for it** - the evaluator's question answered upfront
- **The use case** - the specific scenario for that chapter, chosen to showcase the database's strengths
- **The data** - the dataset used and why it fits
- **Building the application** - hands-on walkthrough with code, including gotchas we encountered along the way
- **What you'd hit in production** - honest notes on limitations, operational considerations and costs
- **When to look elsewhere** - the cases where this database is probably not the right choice

Every chapter comes with a Jupyter notebook. The notebooks are self-contained and independently runnable. They use `all-minilm` via Ollama for local embedding generation, which means no embedding API keys or costs in any chapter. Where a managed cloud service is required - MongoDB Atlas, Pinecone, Weaviate Cloud, Neo4j AuraDB, Snowflake and Databricks - a free tier or trial account is sufficient for all the examples in the book.

## A Note on the "Seven Days" Format

The "seven days" framing is a nod to the tradition of books like *Seven Languages in Seven Weeks* and *Seven Databases in Seven Weeks*. The format carries a specific promise: enough depth to form a genuine opinion, structured as a journey rather than a reference. You're not expected to read this book in seven calendar days, but each chapter is written to stand alone, so you can read them in order or jump to the chapters most relevant to your situation.

## A Note on Embeddings

Throughout the book we use the `all-minilm` model via Ollama to generate embeddings. This model produces 384-dimensional vectors and runs entirely locally on your machine. It isn't the most powerful embedding model available - OpenAI's `text-embedding-3-large` or Cohere's embed models would produce higher-quality embeddings - but it's free, fast and consistent across all seven chapters, which makes it ideal for a tutorial context.

In production, the choice of embedding model matters and deserves its own evaluation. The model, the chunking strategy and the indexing approach all interact. This book focuses on the database layer; the embedding layer is a separate concern that we deliberately keep constant so the database differences are the variable.

## Code and Notebooks

All notebooks and source code are available at:

[seven-vector-databases.github.io](https://seven-vector-databases.github.io)

Each notebook is self-contained. You'll need Python, a virtual environment, classic Jupyter and Ollama installed locally. Specific prerequisites for each chapter, including versions, are documented at the top of the relevant notebook.
