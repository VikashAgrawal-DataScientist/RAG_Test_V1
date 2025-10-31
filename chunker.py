import io
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def pdf_to_text_chunks(pdf_bytes: bytes, chunk_size: int = 1000, chunk_overlap: int = 200):
    if not pdf_bytes:
        return []

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        return []

    texts = []
    for page in getattr(reader, "pages", []):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text and text.strip():
            texts.append(text.strip())

    if not texts:
        return []

    full_text = "\n\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    parts = splitter.split_text(full_text)
    docs = [Document(page_content=p, metadata={}) for p in parts if p and p.strip()]
    return docs
