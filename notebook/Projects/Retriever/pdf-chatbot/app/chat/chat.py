from langchain_openai import ChatOpenAI

from app.chat.chains.retrieval import StreamingConversationsalRetrievalChain
from app.chat.llms.chat_openai import build_llm
from app.chat.memories.sql_memory import build_memory
from app.chat.models import ChatArgs

# from app.chat.vector_stores.qdrant import build_retriever
from app.chat.vector_stores.pinecone import build_retriever


def build_chat(chat_args: ChatArgs) -> None:
    retriever = build_retriever(chat_args=chat_args)
    llm = build_llm(chat_args=chat_args)
    condense_question_llm = ChatOpenAI(streaming=False)
    memory = build_memory(chat_args=chat_args)

    return StreamingConversationsalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm=condense_question_llm,
        memory=memory,
        retriever=retriever,
    )
