import os
from pathlib import Path
from typing import Any, Optional, Sequence

import openai
from dotenv import find_dotenv, load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from rich import print
from tools.report import run_write_html_report_tool
from tools.sql import list_DB_tables, run_describe_tables_tool, run_query_tool
from typeguard import typechecked


@typechecked
def load_credentials() -> Optional[str]:
    """This is used to load the API key(s)"""
    _ = load_dotenv(find_dotenv())  # read local .env file
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    return OPENAI_API_KEY


# Params
OPENAI_CHAT_MODEL: str = "gpt-3.5-turbo"
TEMPERATURE: float = 0.0
DB_PATH: Path = Path("./data/db/db.sqlite")
OPENAI_API_KEY: str = load_credentials()

tools: Sequence[Any] = [
    run_query_tool,
    run_describe_tables_tool,
    run_write_html_report_tool,
]


# Create LLM
chat_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model=OPENAI_CHAT_MODEL,
    temperature=TEMPERATURE,
)
db_tables: str = list_DB_tables()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=(
                "You are an AI assistant that has access to a SQL database "
                f"with the following tables: {db_tables}. Do not make any "
                "assumptions about the table or column names. Instead, always use "
                "the `describe_tables` function to fetch the database information. "
            )
        ),
        # track the conversations with MessagesPlaceholder
        # chat_history contains the Human and the AI Messages
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        # variable_name `agent_scratchpad` is compulsory
        # track the conversations with MessagesPlaceholder
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Agents: chains that know how to use tools
agent = OpenAIFunctionsAgent(
    llm=chat_llm,
    tools=tools,
    prompt=prompt,
)

# Executor: runs an agent until the response is NOT a func call
agent_executor = AgentExecutor(
    memory=memory,  # Add memory!
    verbose=True,
    agent=agent,
    tools=tools,
)

# query: str = "How many users have provided a shipping address?"
# query: str = "What are the top 10 expensive products?"
# query_1: str = "Summarize the top 5 most popular products. Create a report."
query_1: str = "How many order are there? Create a report."
query_2: str = "Repeat the exact process for the users."

try:
    agent_executor.invoke({"input": query_1})
    agent_executor.invoke({"input": query_2})
except openai.InvalidRequestError as err:
    print(f"[ERROR]: {err}")
