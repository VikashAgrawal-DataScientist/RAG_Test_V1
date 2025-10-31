# vector_store.py
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from embeddings import get_embeddings
from config import PERSIST_DIR

_DUMMY_SOURCE = "__RAG_INIT_DUMMY__"

def get_faiss_path():
    return os.path.join(PERSIST_DIR, "faiss_index")

def _create_initial_index(embeddings):

    dummy_doc = Document(page_content="__EMPTY_INIT_DOCUMENT__",
                         metadata={"source": _DUMMY_SOURCE})
    vs = FAISS.from_documents([dummy_doc], embeddings)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    vs.save_local(get_faiss_path())
    return vs

def load_or_create_vectorstore():
    embeddings = get_embeddings()
    faiss_path = get_faiss_path()
    if os.path.exists(faiss_path):
        try:
            vs = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
            return vs
        except Exception:
            pass

    vs = _create_initial_index(embeddings)
    return vs

def add_documents_and_persist(docs):

    if not docs:
        return load_or_create_vectorstore()
    vs = load_or_create_vectorstore()
    vs.add_documents(docs)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    vs.save_local(get_faiss_path())
    return vs

def retrieve(query: str, k: int = 3):

    vs = load_or_create_vectorstore()

    docs = []

    try:
        docs = vs.similarity_search(query, k=k)
    except Exception:
 
        try:
            results = vs.similarity_search_with_score(query, k=k)

            docs = [r[0] for r in results]
        except Exception:
 
            try:
                retriever = vs.as_retriever(search_kwargs={"k": k})

                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(query)
                elif hasattr(retriever, "retrieve"):
                    docs = retriever.retrieve(query)
                elif hasattr(retriever, "get_relevant_documents_by_score"):
                    docs = [d for d, _ in retriever.get_relevant_documents_by_score(query)]
                else:
                    docs = []
            except Exception:
                docs = []

    if not docs:
        return []

    filtered = [d for d in docs if d.metadata.get("source") != _DUMMY_SOURCE]
    return filtered
