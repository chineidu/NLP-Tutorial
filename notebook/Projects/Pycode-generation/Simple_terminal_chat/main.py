import os
from typing import Any, Optional

from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
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


@typechecked
def terminal_chat() -> None:
    """This function is used to create a code generation program using Langchain and an LLM."""
    OPENAI_API_KEY: str = load_credentials()
    # Create LLM
    chat_llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_CHAT_MODEL,
        temperature=TEMPERATURE,
    )

    memory = ConversationBufferMemory(
        memory_key="messages",  # key for storing the chat history
        return_messages=True,  # returns the promp classes. e.g. HumanMsg, AIMsg, etc
    )

    # Used for chat completion LLMs
    prompt = ChatPromptTemplate(
        input_variables=["content", "messages"],
        messages=[
            HumanMessagePromptTemplate.from_template("{content}"),
            MessagesPlaceholder(variable_name="messages"),  # used to track the conversations
        ],
    )
    chain: LLMChain = LLMChain(
        llm=chat_llm,
        prompt=prompt,
        memory=memory,  # Add memory (new!),
    )

    print("[INFO]: Starting chatbot ...")
    while True:
        content: str = input(">> ")
        if content.lower() not in ["exit", "bye"]:
            result: str = chain({"content": content}).get("text")
            print(result)
        else:
            print("[INFO]: Exiting ...")
            break
    return None


if __name__ == "__main__":
    terminal_chat()
