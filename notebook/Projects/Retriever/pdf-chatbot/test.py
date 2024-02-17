from collections.abc import Iterator
from typing import Any
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from rich.console import Console

console = Console()
load_dotenv()

chat_llm = ChatOpenAI()
output_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")

chain = prompt | chat_llm | output_parser

result: Any = chain.invoke({"topic": "Jesus"})
console.print(result)
