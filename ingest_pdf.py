from chunker import pdf_to_text_chunks
from vector_store import add_documents_and_persist
from config import CHUNK_SIZE, CHUNK_OVERLAP

def ingest_pdf_bytes(pdf_bytes: bytes, source: str = None):
    docs = pdf_to_text_chunks(pdf_bytes, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not docs:
        return 0
    if source:
        for d in docs:
            md = d.metadata or {}
            md["source"] = source
            d.metadata = md
    add_documents_and_persist(docs)
    return len(docs)
