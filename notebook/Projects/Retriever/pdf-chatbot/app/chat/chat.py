from app.chat.models import ChatArgs

from app.chat.vector_stores.qdrant import build_retriever


def build_chat(chat_args: ChatArgs) -> None:
    retriever = build_retriever(chat_args)
