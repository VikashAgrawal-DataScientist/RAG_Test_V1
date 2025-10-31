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
    """
    Create an initial FAISS index with a single tiny dummy document.
    This avoids calling from_documents([]) which causes an IndexError in some LangChain versions.
    """
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
            # fallback to creating a fresh index
            pass

    # create initial index safely
    vs = _create_initial_index(embeddings)
    return vs

def add_documents_and_persist(docs):
    """
    docs: list of langchain_core.documents.Document
    """
    if not docs:
        return load_or_create_vectorstore()
    vs = load_or_create_vectorstore()
    vs.add_documents(docs)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    vs.save_local(get_faiss_path())
    return vs

def retrieve(query: str, k: int = 3):
    """
    Retrieve top-k documents using the vectorstore similarity search API.
    Compatible across LangChain versions that expose similarity_search or similarity_search_with_score.
    Filters out the dummy initialization document.
    """
    vs = load_or_create_vectorstore()

    # Try common vectorstore search methods with fallbacks
    docs = []
    # 1) preferred: similarity_search (returns list[Document])
    try:
        docs = vs.similarity_search(query, k=k)
    except Exception:
        # 2) some versions name it similarity_search_with_score
        try:
            results = vs.similarity_search_with_score(query, k=k)
            # results is list of (Document, score) pairs
            docs = [r[0] for r in results]
        except Exception:
            # 3) fallback: use as_retriever then call .retrieve(...) if available
            try:
                retriever = vs.as_retriever(search_kwargs={"k": k})
                # try different method names
                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(query)
                elif hasattr(retriever, "retrieve"):
                    docs = retriever.retrieve(query)
                elif hasattr(retriever, "get_relevant_documents_by_score"):
                    docs = [d for d, _ in retriever.get_relevant_documents_by_score(query)]
                else:
                    # last resort: empty
                    docs = []
            except Exception:
                docs = []

    if not docs:
        return []

    # filter out dummy init docs
    filtered = [d for d in docs if d.metadata.get("source") != _DUMMY_SOURCE]
    return filtered
