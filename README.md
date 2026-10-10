# ACRE — Adaptive Cognitive Retrieval Engine

ACRE is a retrieval-augmented generation (RAG) research project exploring how different retrieval strategies can be combined to answer complex, multi-hop questions more effectively.

Instead of relying on a single retrieval method for every query, ACRE uses intent routing to select a retrieval path, combines complementary retrieval techniques, and applies a relevance-grading stage before generating an answer with a locally hosted language model.

## Overview

Traditional RAG systems retrieve relevant document chunks and pass them to an LLM for answer generation. However, retrieving context does not guarantee that the context is sufficient, relevant, or useful for answering a complex question.

ACRE explores three ideas:

- **Adaptive retrieval:** Route queries through different retrieval paths based on their intent.
- **Complementary retrieval strategies:** Explore techniques that capture semantic similarity, relationships between entities, and information across different levels of abstraction.
- **Retrieval quality control:** Grade retrieved chunks before using them for answer generation.

The goal is to investigate how these components can work together in a modular RAG pipeline.

## Architecture

```text
                    User Query
                         |
                         v
                   Intent Router
                         |
              +----------+----------+
              |                     |
              v                     v
         Simple Query         Relational Query
              |                     |
              v                     v
        FAISS Retrieval      Advanced Retrieval
                                    |
                          +---------+---------+
                          |         |         |
                          v         v         v
                        HyDE +   HyDE +   Other configured
                       GraphRAG  RAPTOR   retrieval paths
                          |         |         |
                          +---------+---------+
                                    |
                                    v
                           Self-RAG-Style Gate
                                    |
                         +----------+----------+
                         |          |          |
                         v          v          v
                      Relevant   Partially   Irrelevant
                                 Relevant
                         |          |          |
                         +----------+----------+
                                    |
                                    v
                          Context Selection
                                    |
                                    v
                         Llama 3.2 via Ollama
                                    |
                                    v
                             Final Answer
```

*Conceptual overview of the pipeline. The exact execution path depends on the query and the implemented configuration.*

## Key Components

### 1. Intent Routing

The router distinguishes between simpler queries and relational queries that may require more involved retrieval. This allows the system to explore different retrieval strategies instead of applying the same pipeline to every question.

### 2. FAISS-Based Retrieval

ACRE uses FAISS for vector similarity search.

- **Embedding model:** `nomic-embed-text`
- **Vector dimensionality:** 768
- **Index:** FAISS `IndexFlatL2`
- **Purpose:** Retrieve candidate chunks based on vector distance.

The index stores chunk embeddings, while the corresponding text is maintained separately for retrieval and context construction.

### 3. HyDE — Hypothetical Document Embeddings

HyDE explores query expansion through a hypothetical answer or document. The generated text is embedded and used to help retrieve documents that may be semantically relevant even when their wording differs from the original query.

### 4. GraphRAG

GraphRAG incorporates relationships between entities and concepts into retrieval. It is particularly relevant to questions where answering requires connecting information across multiple documents or entities.

ACRE explores combining HyDE with graph-based retrieval for relational queries.

### 5. RAPTOR

RAPTOR organizes information at multiple levels of abstraction using recursively clustered and summarized text. This can help retrieval access broader contextual information in addition to fine-grained chunks.

ACRE explores a HyDE + RAPTOR retrieval path as another approach to handling complex queries.

### 6. Self-RAG-Style Relevance Gate

Before generation, retrieved chunks are graded as:

- **Relevant:** Directly useful for answering the query.
- **Partially relevant:** Contains some useful information but may not fully support the answer.
- **Irrelevant:** Does not meaningfully support the query.

The purpose is to reduce the amount of irrelevant context passed to the generation stage.

This is a Self-RAG-inspired relevance-grading component, rather than a claim that the complete original Self-RAG training framework has been implemented.

### 7. Local LLM Generation

ACRE uses **Llama 3.2 (3B) through Ollama** to generate answers from the selected context.

Local inference makes experimentation possible without requiring a paid hosted LLM API for generation. It also allows the retrieval pipeline to be explored independently of a particular cloud model provider.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Vector search | FAISS |
| Embeddings | Nomic Embed Text |
| Retrieval framework | LlamaIndex |
| Graph processing | NetworkX |
| NLP | spaCy |
| Local LLM inference | Ollama |
| Generation model | Llama 3.2 (3B) |
| Evaluation datasets | HotpotQA, MultiHop-RAG |

## Evaluation

ACRE has been explored on multi-hop question-answering tasks using:

- **HotpotQA — Distractor setting:** A question-answering benchmark involving reasoning across multiple supporting documents.
- **MultiHop-RAG:** A benchmark designed to investigate retrieval and reasoning over information that requires multiple hops.

During development, a 300-record subset of HotpotQA was used for local experimentation. MultiHop-RAG was also used as part of the evaluation workflow.

The project investigates how retrieval choices and relevance filtering affect the context available for answer generation.

A comprehensive comparison against baseline RAG systems, full-dataset evaluation, and systematic latency and answer-quality measurements should be treated as separate evaluation work rather than assumed from the current implementation.

## Getting Started

### Prerequisites

- Python
- Git
- Ollama
- Sufficient memory and disk space for the embedding model, language model, and indexed data

### 1. Clone the repository

```bash
git clone https://github.com/be-hope/ACRE.git
cd ACRE
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up Ollama

Install Ollama from [ollama.com](https://ollama.com/), then pull the models used by the project:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Make sure the Ollama service is running before starting the pipeline.

### 5. Configure the project

Review `config.py` and the relevant configuration files to set the model names, data paths, and other required parameters.

Ensure that the required datasets and any preprocessed indexes or intermediate files are available before running the pipeline.

### 6. Run the project

The repository contains `main.py` as an entry point. Check its current configuration and argument requirements before executing:

```bash
python main.py
```

For evaluation, inspect the scripts and configuration under `evaluation/` to determine the appropriate dataset preparation and execution steps.

*Note: Exact dataset download commands, index-building commands, and evaluation invocations depend on the current scripts and configuration in the repository.*

## Project Structure

```text
ACRE/
├── config.py
├── main.py
├── requirements.txt
├── test_ollama.py
├── src/
├── data/
├── evaluation/
└── README.md
```

- `config.py` — Project configuration.
- `main.py` — Main entry point.
- `src/` — Core retrieval and generation components.
- `data/` — Dataset and intermediate data files.
- `evaluation/` — Evaluation-related code.
- `test_ollama.py` — Ollama integration testing.
- `requirements.txt` — Python dependencies.

## Future Work

Potential directions for further development include:

- Comparing retrieval strategies independently and in combination.
- Measuring answer correctness, retrieval recall, and context relevance.
- Evaluating the trade-off between answer quality, latency, and computational cost.
- Comparing local and hosted LLM generation backends.
- Adding reproducible baselines and more comprehensive evaluation across datasets.

## Repository

GitHub: https://github.com/be-hope/ACRE
