"""This module contains the implementation of the AI assistant chatbot"""

from pprint import pprint
from typing import Any, Dict, List

import click
from dotenv import find_dotenv, load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pydantic import BaseModel

_ = load_dotenv(find_dotenv())  # read local .env file


class DecideBot(BaseModel):
    """This is used to create the Decide AI Assistant."""

    # Constants
    PERSIST_DIRECTORY: str = "./data/doc_db"
    CHUNK_SIZE: int = 1_000
    CHUNK_OVERLAP: int = 50
    TEMPERATURE: float = 0.0

    # Variables
    filepath: str
    create_index: bool

    def _load_data(self) -> List[Any]:
        """This is used to load the JSON data."""
        loader = JSONLoader(
            file_path=self.filepath,
            jq_schema=".data",
            text_content=False,
        )
        data = loader.load()
        return data

    def _preprocess_data(self) -> Chroma:
        """This is used to split and store the embedded data in a vector store."""
        data = self._load_data()

        # === Embedding model ===
        embedding = OpenAIEmbeddings()

        # === Split the doc(s) ===
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", r"\{\}", ""],
        )
        splits = text_splitter.split_documents(documents=data)

        # === Create and save vectors tore to disk ===
        if self.create_index:
            vector_db = Chroma.from_documents(
                documents=splits,
                embedding=embedding,
                persist_directory=self.PERSIST_DIRECTORY,
            )

        # === Load data ===
        vector_db = Chroma(persist_directory=self.PERSIST_DIRECTORY, embedding_function=embedding)
        return vector_db

    def _create_RAG_model(self) -> RetrievalQA:
        """This is used to create a RAG model for question and answering."""
        llm = ChatOpenAI(temperature=self.TEMPERATURE)
        # === Create memory ===
        memory = ConversationBufferWindowMemory(
            k=5,  # window size
            input_key="question",
            memory_key="chat_history",
        )
        vector_db = self._preprocess_data()

        # === Init Chatbot ===
        decide_bot = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(),  # retrieve data from data source
            verbose=False,
            chain_type_kwargs={
                "document_separator": "<<<<>>>>>",
                "memory": memory,
            },
        )
        return decide_bot

    def generate_response(self, *, query: str) -> Dict[str, Any]:
        """This is a chat bot used for question and answering."""
        decide_bot = self._create_RAG_model()
        result = decide_bot({"query": query})
        return result


@click.command()
@click.option("--filepath", help="Filepath to the source data", type=str)
@click.option(
    "--create-index", help="Do you want to regenerate embeddings?", type=bool, default=False
)
@click.option("--query", help="Enter your query", type=str)
def main(filepath: str, create_index: bool, query: str) -> None:
    """This is the main function."""
    decide_ai_assistant = DecideBot(filepath=filepath, create_index=create_index)
    result = decide_ai_assistant.generate_response(query=query)
    click.echo(pprint(result))  # type: ignore


if __name__ == '__main__':
    main()
