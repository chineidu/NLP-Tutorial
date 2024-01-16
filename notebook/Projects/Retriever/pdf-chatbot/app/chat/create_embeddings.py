from typing import Any
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typeguard import typechecked
from rich.console import Console

from app.chat.vector_stores.qdrant import set_up_vector_db

console = Console()


@typechecked
def create_embeddings_for_pdf(pdf_id: str, pdf_path: str):
    """Generate and store embeddings for the given pdf."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    loader = PyPDFLoader(file_path=pdf_path)
    docs: list[Any] = loader.load_and_split(text_splitter=text_splitter)
    vector_db = set_up_vector_db(documents=docs)

    console.print(f"{vector_db}\n", style="green")

    return vector_db


# vector_store = create_embeddings_for_pdf()
# console.print(f"[INFO]: {cr}")
