from typing import Any

from app.chat.vector_stores.qdrant import set_up_vector_db
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from typeguard import typechecked

console = Console()


@typechecked
def create_embeddings_for_pdf(pdf_id: str, pdf_path: str) -> Any:
    """Generate and store embeddings for the given pdf."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    loader = PyPDFLoader(file_path=pdf_path)
    docs: list[Any] = loader.load_and_split(text_splitter=text_splitter)

    # Update metadata
    for doc in docs:
        doc.metadata: dict[str, Any] = {
            "page": doc.metadata.get("page"),
            "text": doc.page_content,
            "pdf_id": pdf_id,
        }
    vector_db = set_up_vector_db(documents=docs)

    console.print(f"{vector_db}\n", style="green")

    return vector_db
