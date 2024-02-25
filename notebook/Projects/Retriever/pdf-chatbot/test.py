from queue import Queue
from threading import Thread
from typing import Any

from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from rich.console import Console

console = Console()
load_dotenv()


class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue: Queue) -> None:
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.queue.put(token)

    def on_llm_end(self, response: str, **kwargs: Any) -> Any:
        self.queue.put("</None/>")

    def on_llm_error(self, error: str, **kwargs: Any) -> Any:
        self.queue.put("</None/>")


class StreamingChain1(LLMChain):
    def stream(self, input):
        queue = Queue()
        handler = StreamingHandler(queue)

        def task():
            self(input, callbacks=[handler])

        Thread(target=task).start()

        while True:
            token = queue.get()
            if token == "</None/>":
                break
            yield token


# === Using Mixin ===


class StreamingChainMixin:
    def stream(self, input):
        queue = Queue()
        handler = StreamingHandler(queue)

        def task():
            self(input, callbacks=[handler])

        Thread(target=task).start()

        while True:
            token = queue.get()
            if token == "</None/>":
                break
            yield token


class StreamingChain2(StreamingChainMixin, LLMChain):
    pass


chat_llm = ChatOpenAI(streaming=True)
output_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")

# chain = prompt | chat_llm | output_parser
chain = StreamingChain2(llm=chat_llm, prompt=prompt)


# result: Any = chain({"topic": "Jesus"})
result: Any = chain.stream("Jesus")
for _r in result:
    console.print(_r)
