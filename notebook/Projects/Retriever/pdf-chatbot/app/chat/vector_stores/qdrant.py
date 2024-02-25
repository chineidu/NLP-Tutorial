import os
from typing import Any

from dotenv import load_dotenv
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from typeguard import typechecked

from app.chat.embeddings.openai import embedding_model
from app.chat.models import ChatArgs

_ = load_dotenv()

URL: str = "http://localhost:6333/"
COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME")

client = QdrantClient(url=URL)

doc_store = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model,
)


@typechecked
def build_retriever(chat_args: ChatArgs) -> Any:
    search_kwargs = {"filter": {"pdf_id": chat_args.pdf_id}}

    return doc_store.as_retriever(search_kwargs=search_kwargs)
