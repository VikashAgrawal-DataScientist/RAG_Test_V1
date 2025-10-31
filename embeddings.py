import os
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY

def get_embeddings():
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    return OpenAIEmbeddings()
