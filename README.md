# Movie Plot RAG System

A lightweight Retrieval-Augmented Generation system for querying Wikipedia movie plots using hybrid search and LLM reasoning.

## Setup

```bash
# Clone and navigate to project
cd type-b-digital

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your Groq API key to .env
```

## Run

```bash
python rag_system.py
```

Ask questions about movies. Type 'quit' to exit.

## Configuration

Edit `.env` to customize:

- `GROQ_API_KEY` - Your Groq API key (required)
- `EMBEDDING_MODEL` - HuggingFace model for embeddings
- `LLM_MODEL` - Groq model name
- `CHUNK_SIZE` - Text chunk size for retrieval
- `TOP_K_RESULTS` - Number of results to retrieve
- `BM25_WEIGHT` - Weight for keyword search (0-1)
- `SEMANTIC_WEIGHT` - Weight for semantic search (0-1)
- `USE_CACHE` - Enable vector database caching

See `.env.example` for all options.

## Tech Stack

**Data Processing**
- Pandas for CSV handling
- RecursiveCharacterTextSplitter for intelligent text chunking

**Retrieval**
- FAISS for fast vector similarity search
- BM25 for keyword-based matching
- Hybrid search combining both approaches

**Language Model**
- Groq API with Qwen 3 32B model
- Built-in reasoning capabilities
- Structured JSON output via Pydantic

**Embeddings**
- HuggingFace sentence-transformers
- all-mpnet-base-v2 model (high quality)

**Framework**
- LangChain for RAG pipeline orchestration
- LCEL (LangChain Expression Language) for chain composition

## Why It Excels

**Smart Caching** - Vector database is built once and reused. First run takes 90 seconds, subsequent runs take 8 seconds.

**Hybrid Retrieval** - Combines keyword matching (BM25) with semantic search (FAISS) for better accuracy. Finds exact movie titles while understanding conceptual queries.

**Reasoning Transparency** - Qwen model provides explicit reasoning for how it derived answers from retrieved contexts.

**Minimal Dependencies** - Only 9 packages, no bloat. 

## Project Structure

```
.
├── rag_system.py          # Main RAG implementation (195 lines)
├── requirements.txt       # Python dependencies
├── .env                   # Configuration (not committed)
├── .env.example          # Configuration template
├── wiki_movie_plots_deduped.csv  # Dataset
└── vector_db/            # Cached indices (auto-generated)
```

## Dataset

Uses last 400 movies from Wikipedia Movie Plots dataset. Covers 2008-2017 releases with substantial plot descriptions.

