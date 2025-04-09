# 🧠 SHL Assessment Recommendation System

Hiring managers often struggle to identify the right SHL assessments for specific roles. This project is an intelligent recommendation system powered by a **Retrieval-Augmented Generation (RAG)** pipeline. It enables natural language queries over SHL assessment data and returns curated, structured recommendations to help hiring teams make faster and more informed decisions.

---

## 🚀 Live Demo

- 🔗 Frontend: [shl-frontend-5o9v.onrender.com](https://shl-frontend-5o9v.onrender.com)
- 🔗 Backend API: [shl-rag-recommender-og.onrender.com](https://shl-rag-recommender-og.onrender.com)

---

## 🧠 Problem Addressed

Traditional keyword-based search is time-consuming and often inaccurate for selecting SHL tests based on vague job descriptions. This system allows users to ask:
- "What SHL assessments are best for a Java developer?"
- "Recommend SHL tests for a customer service role."

And get back a **structured table of tests** with links, duration, and reasoning.

---

## 🔍 RAG Architecture

### 1. Data Collection
- Scraped SHL test titles and metadata using automated tools and refined slug list.
- Converted scraped content into structured JSON.

### 2. Knowledge Base Storage
- Embedded job test data using `SentenceTransformers (all-MiniLM-L6-v2)`.
- Stored and indexed vectors in **Pinecone** for fast semantic search.

### 3. Query & Recommendation Flow
- ✅ User asks a natural language question.
- 🔎 Query is semantically matched with embedded documents.
- 💬 Gemini 1.5 Flash model generates a **table of 1–10 relevant SHL tests** with rationale.

---

## 🛠️ Tools & Libraries Used

| Tool | Purpose |
|------|---------|
| **FastAPI** | Backend APIs (`/context/`, `/response/`, `/search/`, `/push_docs/`) |
| **Streamlit** | Frontend for interactive querying and output display |
| **Google Gemini 1.5 Flash** | LLM for test recommendation generation |
| **Sentence Transformers** | Embedding model for semantic understanding |
| **Pinecone** | Vector database for fast, scalable similarity search |
| **dotenv** | Secure environment variable management |
| **pydantic** | Type-safe request/response models |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/context/` | Retrieve semantically similar job test descriptions |
| `POST` | `/response/` | Generate a Gemini-based table of SHL tests |
| `GET`  | `/search/?query=` | Semantic search of job tests for a query |
| `POST` | `/push_docs/` | Bulk upload job descriptions to Pinecone |

---

## 📁 Project Structure

```bash
SHL-RAG-Recommender/
├── backend/
│   └── api.py           # FastAPI server logic
├── frontend/
│   └── app.py           # Streamlit app
├── job_descriptions.json
└── requirements.txt
