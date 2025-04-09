from fastapi import FastAPI
from dotenv import load_dotenv
import os
import hashlib
import json
import threading
from pydantic import BaseModel
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ---------- Load Env & Keys ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GENAI_MODEL_ID = "models/gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------- Initialize Models ----------
genai.configure(api_key=GEMINI_API_KEY)
generation_model = genai.GenerativeModel(GENAI_MODEL_ID)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ---------- Initialize ChromaDB ----------
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("job_descriptions")

# ---------- Initialize FastAPI ----------
app = FastAPI()

# ---------- Utility Functions ----------
def generate_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prepare_jsons_for_rag(json_paths):
    items = []
    global_index = 0  # unique across all files and jobs

    for json_path in json_paths:
        filename = json_path.split("/")[-1]
        with open(json_path, "r", encoding="utf-8") as f:
            job_data = json.load(f)

        for job in job_data:
            slug = job.get("slug", f"job-{global_index}")
            for rec in job.get("recommendations", []):
                text = json.dumps(rec)
                unique_id = f"{slug}-{global_index}"
                items.append({
                    "id": unique_id,
                    "line": text,
                    "filename": filename,
                    "page_number": "1"
                })
                global_index += 1

    return {"items": items}


def upsert_documents(documents):
    ids = [doc["id"] for doc in documents] 
    texts = [doc["line"] for doc in documents]
    embeddings = [embedding_model.encode(text).tolist() for text in texts]
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings)
    return ids


# ---------- Pydantic Models ----------
class Item(BaseModel):
    id: str
    line: str
    filename: str
    page_number: str = "1"

class Docs(BaseModel):
    items: List[Item]

class Query(BaseModel):
    query: str

class QA(BaseModel):
    query: str
    context: str

# ---------- API Endpoints ----------
@app.post("/push_docs/")
async def push_docs(item: Docs):
    try:
        docs = item.dict()["items"]
        ids = upsert_documents(docs)
        return {"status": "success", "inserted_ids": ids}
    except Exception as e:
        return {"error": str(e)}

@app.post("/context/")
async def get_context(item: Query):
    try:
        query_embedding = embedding_model.encode(item.query).tolist()
        print("Query Embedding:", query_embedding)

        results = collection.query(
            query_embeddings=[query_embedding],  # use embedding
            n_results=5,
            include=["documents"]
        )
        print("Query Results:", results)

        return {"docs": results.get("documents", [[]])[0]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/response/")
async def get_response(item: QA):
    try:
        if not item.context.strip():
            return {"output": "I don't know."}

        prompt = f"""
        You are an expert AI assistant helping HR professionals recommend SHL assessments for a specific role.

        Use the provided context below to identify and recommend the most relevant individual test solutions.

        Context:
        {item.context}

        Task:
        Based on the context, recommend between 1 and 10 individual SHL test solutions (minimum 1, maximum 10). Present your recommendations in a table format with the following columns:

        | Test Name | URL | Duration | Test Type | Remote Testing (Yes/No) | Adaptive Support (Yes/No) | Reason for Selection |

        Ensure each recommended test is relevant to the role, and explain briefly in the final column *why* that test was selected (e.g., assesses critical thinking, relevant to customer service, evaluates technical skills, etc.).

        Only output the final table.
        """


        response = generation_model.generate_content(prompt)
        print("Response:", response.text)
        return {"output": response.text}
    except Exception as e:
        return {"error": str(e)}


@app.get("/search/")
async def search(query: str):
    """
    GET endpoint that returns relevant context for a query in JSON.
    Example: /search/?query=data+analyst+assessment
    """
    try:
        query_embedding = embedding_model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents"]
        )

        return {
            "query": query,
            "results": results.get("documents", [[]])[0]
        }
    except Exception as e:
        return {"error": str(e)}


# ---------- Startup Loader ----------
def auto_push_job_data():
    try:
        file_path = "backend/job_descriptions.json"
        if os.path.exists(file_path):
            print("📤 Indexing job_descriptions.json into ChromaDB...")
            json_files = ["backend/job_descriptions.json", "backend/job_descriptions_1.json"]
            data = prepare_jsons_for_rag(json_files)
            from fastapi.testclient import TestClient
            client = TestClient(app)
            response = client.post("/push_docs/", json=data)
            print("✅ Job data indexed:", response.json())
        else:
            print("⚠️ job_descriptions.json not found.")
    except Exception as e:
        print("❌ Error pushing job data:", str(e))

# Run auto-loader on startup
threading.Thread(target=auto_push_job_data, daemon=True).start()
