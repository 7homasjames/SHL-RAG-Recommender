from fastapi import FastAPI
from dotenv import load_dotenv
import os
import hashlib
import json
import threading
from pydantic import BaseModel
from typing import List
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# ---------- Load Env & Keys ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "shlrag"

# Use existing index (don't try to create)

GENAI_MODEL_ID = "models/gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------- Initialize Models ----------
genai.configure(api_key=GEMINI_API_KEY)
generation_model = genai.GenerativeModel(GENAI_MODEL_ID)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ---------- Initialize Pinecone (NEW SDK) ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)


# ---------- Initialize FastAPI ----------
app = FastAPI()

# ---------- Utility Functions ----------
def generate_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def prepare_jsons_for_rag(json_paths):
    items = []
    global_index = 0
    for json_path in json_paths:
        filename = os.path.basename(json_path)
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

def upsert_documents(documents, batch_size=50):
    vectors = []
    for doc in documents:
        embedding = embedding_model.encode(doc["line"]).tolist()
        vectors.append({
            "id": doc["id"],
            "values": embedding,
            "metadata": {"text": doc["line"]}
        })

    # Split into batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        print(f"Upserting batch {i//batch_size + 1} of {len(vectors)//batch_size + 1}")
        index.upsert(vectors=batch)

    return [doc["id"] for doc in documents]

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
        print("Inserted IDs:", ids)
        print("Documents:", docs)
        return {"status": "success", "inserted_ids": ids}
    except Exception as e:
        return {"error": str(e)}

@app.post("/context/")
async def get_context(item: Query):
    try:
        query_embedding = embedding_model.encode(item.query).tolist()
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        print("Query Embedding:", query_embedding)
        print("Results:", results)
        return {
            "docs": [match["metadata"]["text"] for match in results["matches"]]
        }
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
        return {"output": response.text}
    except Exception as e:
        return {"error": str(e)}

@app.get("/search/")
async def search(query: str):
    try:
        query_embedding = embedding_model.encode(query).tolist()
        print("Search Query Embedding:", query_embedding)
        print("Search Query:", query)
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        return {
            "query": query,
            "results": [match["metadata"]["text"] for match in results["matches"]]
        }
    except Exception as e:
        return {"error": str(e)}

# ---------- Startup Loader ----------
def auto_push_job_data():
    try:
        file_path = "job_descriptions.json"
        if os.path.exists(file_path):
            print("📤 Indexing job_descriptions.json into Pinecone...")
            json_files = ["job_descriptions.json", "job_descriptions_1.json"]
            data = prepare_jsons_for_rag(json_files)
            print("Job data prepared for indexing:", data)
            from fastapi.testclient import TestClient
            client = TestClient(app)
            response = client.post("/push_docs/", json=data)
            print("✅ Job data indexed:", response.json())
        else:
            print("⚠️ job_descriptions.json not found.")
    except Exception as e:
        print("❌ Error pushing job data:", str(e))

threading.Thread(target=auto_push_job_data, daemon=True).start()
