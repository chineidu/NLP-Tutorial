import os
from typing import Any

import pinecone
from langchain.vectorstores import Pinecone

from app.chat.embeddings.openai import embedding_model

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV_NAME"))

vector_store = Pinecone.from_existing_index(os.getenv("PINECONE_INDEX_NAME"), embedding_model)


def build_retriever(chat_args) -> Any:
    search_kwargs = {"filter": {"pdf_id": chat_args.pdf_id}}
    return vector_store.as_retriever(search_kwargs=search_kwargs)
