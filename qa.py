import os
from config import OPENAI_API_KEY
from vector_store import retrieve
from langchain_openai import ChatOpenAI

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

PROMPT_TEMPLATE = """You are an expert assistant. Use the provided context to answer the question concisely.
If the answer is not in the context, say "I couldn't find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:"""

def llm_call(prompt: str, model_name: str = "gpt-4"):
    llm = ChatOpenAI(model=model_name)
    try:
        return llm.predict(prompt)
    except Exception:
        try:
            out = llm(prompt)
            if isinstance(out, str):
                return out
            return getattr(out, "content", str(out))
        except Exception:
            try:
                res = llm.invoke(prompt)
                return getattr(res, "content", str(res))
            except Exception as e:
                raise RuntimeError(f"LLM call failed: {e}")

def answer_with_retrieval(question: str, k: int = 3, model_name: str = "gpt-4"):
    docs = retrieve(question, k=k)
    context = "\n\n---\n\n".join([d.page_content for d in docs]) if docs else ""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = llm_call(prompt, model_name=model_name)
    sources = [d.metadata.get("source", "") for d in docs]
    return answer, sources
