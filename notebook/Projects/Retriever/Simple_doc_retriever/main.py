import os
from pathlib import Path
from typing import Any, Optional

from dotenv import find_dotenv, load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from pydantic import BaseModel
from rich import print
from typeguard import typechecked

# Params
OPENAI_CHAT_MODEL: str = "gpt-3.5-turbo"
TEMPERATURE: float = 0.8


@typechecked
def load_credentials() -> Optional[str]:
    """This is used to load the API key(s)"""
    _ = load_dotenv(find_dotenv())  # read local .env file
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    return OPENAI_API_KEY


class CustomFactsRetriever(BaseModel):
    chunk_size: int = 200
    chunk_overlap: int = 0
    embedding_model: Any = OpenAIEmbeddings()
    url: str = "http://localhost:6333/"
    collection_name: str
    file_path: Path

    @typechecked
    def process_document(self) -> Optional[list[Any]]:
        """This is used to load and chunk the documents."""
        print("[INFO]: Loading and splitting the docs ...")
        try:
            loader: TextLoader = TextLoader(file_path=self.file_path)
            # Used to chunk the texts
            text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,  # 1. dets the rel. number of chars per chunk
                separator="\n",  # 2. each hunk is separated by this!
                chunk_overlap=self.chunk_overlap,
            )
            # Load and split
            docs: list[Any] = loader.load_and_split(text_splitter=text_splitter)
            return docs
        except RuntimeError as err:
            print(f"[ERROR] {err}")
            return None

    @typechecked
    def set_up_vector_db(self, documents: list[Any]) -> Any:
        """This is used to create embeddings and upsert the embeddings to the collection."""
        vector_db: Any = Qdrant.from_documents(
            documents,
            self.embedding_model,
            url=self.url,
            prefer_grpc=False,
            collection_name=self.collection_name,
            force_recreate=False,
        )

        return vector_db

    @typechecked
    def retriever_chain(self, vector_db: Any) -> RetrievalQA:
        """Set up the retriever for question and answering."""
        # Create LLM
        chat_llm = ChatOpenAI(
            model=OPENAI_CHAT_MODEL,
            temperature=TEMPERATURE,
        )
        print("[INFO]: Setting up retriever ...")
        # Retriever DB
        retriever = vector_db.as_retriever()

        chain = RetrievalQA.from_chain_type(
            llm=chat_llm,
            retriever=retriever,
            chain_type="stuff",  # it feeds the LLM a prompt built from document list.
            return_source_documents=True,  # to easily detect hallucination
            input_key="query",
        )
        return chain

    @typechecked
    def run_query(self) -> None:
        """This is used to retrieve info from the database."""
        vector_db: Any = self.set_up_vector_db(documents=self.process_document())
        chain: RetrievalQA = self.retriever_chain(vector_db=vector_db)
        print("[INFO]: Fetching the retrieved information ...")

        while True:
            query: str = input(">> ")
            if query.lower() not in ["exit", "bye"]:
                result: dict[str, Any] = chain({"query": query})
                # Check for hallucination
                if len(result.get("source_documents", 0)) == 0:
                    print("[WARNING]: LLM is possibly hallucinating!!!")
                    print(result.get("result"))
                else:
                    print(result.get("result"))

            else:
                print("[INFO]: Exiting ...")
                break

        return None


if __name__ == "__main__":
    collection_name = "facts"
    file_path: Path = Path("./data/facts.txt")
    facts_app: CustomFactsRetriever = CustomFactsRetriever(
        collection_name=collection_name, file_path=file_path
    )
    facts_app.run_query()
