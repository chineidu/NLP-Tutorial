from collections.abc import Iterator
from typing import Any
from uuid import UUID
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from rich.console import Console

console = Console()
load_dotenv()


class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        console.print(token)


chat_llm = ChatOpenAI(
    # streaming=True,
    # callbacks=[StreamingHandler()],
)
output_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")

chain = prompt | chat_llm | output_parser

result: Any = chain.stream({"topic": "Jesus"})
for _r in result:
    console.print(_r)
