from typing import Any

# from app.chat.create_embeddings import build_retriever
from app.chat.llms.chat_openai import build_llm
from app.chat.memories.sql_memory import build_memory
from app.chat.models import ChatArgs

from app.chat.vector_stores.qdrant import build_retriever
from langchain.chains import ConversationalRetrievalChain


def build_chat(chat_args: ChatArgs) -> None:
    retriever = build_retriever(chat_args=chat_args)
    llm = build_llm(chat_args=chat_args)
    memory = build_memory(chat_args=chat_args)

    return ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=retriever)
