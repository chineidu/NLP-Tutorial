import os
from typing import Any
from langchain.vectorstores import Qdrant

from dotenv import load_dotenv
from typeguard import typechecked
from app.chat.embeddings.openai import embedding_model
# from app.chat.create_embeddings import d

_ = load_dotenv()

URL: str = "http://localhost:6333/"
COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME")


@typechecked
def set_up_vector_db(documents: list[Any]) -> Any:
    """This is used to create embeddings and upsert the embeddings to the collection."""
    vector_db: Any = Qdrant.from_documents(
        documents,
        embedding_model,
        url=URL,
        prefer_grpc=False,
        collection_name=COLLECTION_NAME,
        force_recreate=False,
    )

    return vector_db
