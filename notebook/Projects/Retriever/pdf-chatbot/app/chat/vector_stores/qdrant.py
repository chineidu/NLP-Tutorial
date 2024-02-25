import os
from typing import Any

from dotenv import load_dotenv
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from typeguard import typechecked

from app.chat.embeddings.openai import embedding_model
from app.chat.models import ChatArgs

_ = load_dotenv()

QDRANT_URL: str = os.getenv("QDRANT_URL")
COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME")
QDRANT_VECTOR_SIZE: int = int(os.getenv("QDRANT_VECTOR_SIZE"))

client = QdrantClient(url=QDRANT_URL)

# Create collection if it doesn't exist
try:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
    )
except Exception:
    pass

vector_store = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model,
)


@typechecked
def build_retriever(chat_args: ChatArgs) -> Any:
    """This is used to build a retriever for the conversational retriever chain."""
    search_kwargs = {"filter": {"pdf_id": chat_args.pdf_id}}

    return vector_store.as_retriever(search_kwargs=search_kwargs)
