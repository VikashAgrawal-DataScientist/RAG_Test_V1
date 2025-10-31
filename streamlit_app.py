import streamlit as st
import requests
from ingest_pdf import ingest_pdf_bytes
from qa import answer_with_retrieval

st.set_page_config(page_title="RAG NOW", layout="centered")
st.title("RAG System")

api_url = st.sidebar.text_input("API base URL (optional)", value="")
use_api = bool(api_url.strip())

st.header("Upload and Ingest PDF")
uploaded = st.file_uploader("PDF file", type=["pdf"])
source_name = st.text_input("Source name (optional)")

if uploaded is not None:
    if st.button("Ingest PDF"):
        data = uploaded.read()
        if use_api:
            try:
                files = {"file": (uploaded.name, data, "application/pdf")}
                resp = requests.post(f"{api_url.rstrip('/')}/upload", files=files, data={"source": source_name})
                if resp.status_code == 200:
                    st.success(f"Ingested: {resp.json().get('ingested_chunks')}")
                else:
                    st.error(f"API error: {resp.status_code} {resp.text}")
            except Exception as e:
                st.exception(e)
        else:
            try:
                count = ingest_pdf_bytes(data, source=source_name or uploaded.name)
                if count == 0:
                    st.warning("No text extracted from PDF.")
                else:
                    st.success(f"Ingested {count} chunks from {uploaded.name}")
            except Exception as e:
                st.exception(e)

st.markdown("---")
st.header("Ask a question")
question = st.text_area("Question", height=160)
k = st.slider("Top-k documents", min_value=1, max_value=8, value=3)
model = st.selectbox("Model (local call)", ["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo"])

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Type a question first.")
    else:
        if use_api:
            try:
                resp = requests.post(f"{api_url.rstrip('/')}/ask", json={"question": question})
                if resp.status_code == 200:
                    data = resp.json()
                    st.subheader("Answer")
                    st.write(data.get("answer"))
                    st.subheader("Sources")
                    for s in data.get("sources", []):
                        st.write(f"- {s or '(unknown)'}")
                else:
                    st.error(f"API error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.exception(e)
        else:
            try:
                answer, sources = answer_with_retrieval(question, k=k, model_name=model)
                st.subheader("Answer")
                st.write(answer)
                st.subheader("Sources")
                if sources:
                    for s in sources:
                        st.write(f"- {s or '(unknown)'}")
                else:
                    st.write("No sources returned.")
            except Exception as e:
                st.exception(e)
