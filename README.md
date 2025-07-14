# 🧠 RAG-based Loan Approval Q&A Chatbot

This project implements a lightweight Retrieval-Augmented Generation (RAG) chatbot designed to answer questions related to a **Loan Approval Prediction Dataset**. It combines vector-based semantic search with generative language modeling using a TinyLlama model for fast, contextual, and informative responses.

## 🚀 Project Overview

The chatbot enables users to ask natural language questions about a tabular dataset, returning informed answers based on both statistics and sample data. It uses:

- `SentenceTransformers` for embedding dataset chunks.
- `FAISS` for vector similarity search.
- `TinyLlama-1.1B-Chat` for generating responses.
- `Gradio` for building a user-friendly chat interface.

## 📁 Dataset Used

The notebook expects a CSV file named `Training Dataset.csv`, representing a typical loan application dataset with features like:

- Income
- Credit History
- Employment Status
- And others...

Each column is described statistically using `pandas.describe()` and enriched with a few sample rows.

## 🧱 Tech Stack

| Component              | Description                                 |
|------------------------|---------------------------------------------|
| `pandas`               | Data loading and processing                 |
| `sentence-transformers`| Converts text chunks to embeddings          |
| `faiss-cpu`            | Efficient similarity search over embeddings |
| `transformers`         | Loads the TinyLlama text generation model   |
| `gradio`               | Provides a web-based chat UI                |

## 🔍 How It Works

1. **Data Chunking**: Each column's stats and a few sample rows are turned into textual "chunks" to form a knowledge base.
2. **Vector Indexing**: These chunks are embedded and stored using `FAISS` for nearest-neighbor search.
3. **Query Retrieval**: At runtime, the chatbot retrieves top-K relevant chunks based on semantic similarity to the query.
4. **Contextual Answering**: These chunks, along with chat history, are fed into a TinyLlama-based text generation pipeline to produce an answer.

## 🧠 Prompt Template

The prompt to the model includes:

- Last 3 turns of conversation (user + bot)
- Top-K retrieved dataset chunks
- A task-specific instruction:

```text
You are an assistant for questions about a loan approval dataset.
Previous conversation:
User: ...
Bot: ...
User: <query>
Based on the context below, answer the user’s question.

Context:
<retrieved chunks>

Answer:
```

## 🖥️ Launching the App

To run the chatbot interface locally with sharing enabled:

```bash
!pip install pandas sentence-transformers transformers faiss-cpu gradio
```

Then execute the notebook and follow the Gradio link:

```python
gr.ChatInterface(...).launch(share=True)
```

## ✅ Sample Query

```python
rag_answer("What features are most important for loan approval?", "")
```

This will produce a model-generated response grounded in dataset features like income, credit history, etc.

## 📌 Notes

- Ensure the dataset file is named correctly (`Training Dataset.csv`).
- TinyLlama is used for fast inference; this can be swapped with a larger model if needed.
- No external APIs or cloud resources are required—everything runs locally.

## 📜 License

This project is licensed under the [MIT License](LICENSE).
