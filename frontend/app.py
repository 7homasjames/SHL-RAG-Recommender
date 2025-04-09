import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

# ---------- API Helpers ----------
def get_context(query):
    try:
        res = requests.post(f"{API_BASE}/context/", json={"query": query})
        res.raise_for_status()
        print("Response:", res.json())
        return res.json().get("docs", [])
    except Exception as e:
        st.error(f"Error fetching context: {e}")
        return []

def get_response(query, context):
    try:
        payload = {"query": query, "context": "\n".join(context)}
        res = requests.post(f"{API_BASE}/response/", json=payload)
        res.raise_for_status()
        return res.json().get("output", "No response.")
    except Exception as e:
        st.error(f"Error fetching response: {e}")
        return "Error occurred during generation."

# ---------- UI Setup ----------
st.set_page_config(page_title="Gemini RAG QA", page_icon="🤖")
st.title("💼 Job Description QA Assistant")
st.markdown(
    """
    Ask me anything about job assessments or hiring tests!
    This app uses **ChromaDB + Sentence Transformers** for retrieval,
    and **Google Gemini (1.5 Flash)** for response generation.
    """
)

# ---------- User Input ----------
query = st.text_input("🔍 Enter your question:", placeholder="e.g., What assessments are recommended for a Java Developer")
if query:
    with st.spinner("Fetching context from vector database..."):
        context_docs = get_context(query)
        print("Context Docs:", context_docs)

    if context_docs:
        st.subheader("📚 Retrieved Context")
        selected = st.selectbox(
        "Select a context document",
        options=[f"Context {i+1}" for i in range(len(context_docs))]
    )

        selected_index = int(selected.split(" ")[1]) - 1

        # If the context is plain text, wrap it for JSON display
        doc_to_display = context_docs[selected_index]
        if isinstance(doc_to_display, str):
            st.json({"context": doc_to_display})
        else:
            st.json(doc_to_display)
        with st.spinner("💡 Generating answer with Gemini..."):
            answer = get_response(query, context_docs)

        st.subheader("🤖 Answer")
        st.success(answer)
    else:
        st.warning("No context found in the vector database. Try rephrasing your question.")
