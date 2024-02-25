from typing import Any

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from typeguard import typechecked

# from app.chat.vector_stores.qdrant import doc_store, COLLECTION_NAME
from app.chat.vector_stores.pinecone import vector_store

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
        doc.metadata: dict[str, Any] = {  # type: ignore
            "page": doc.metadata.get("page"),
            "text": doc.page_content,
            "pdf_id": pdf_id,
        }

    vector_store.add_documents(docs)
