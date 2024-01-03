import os
from pathlib import Path
from typing import Any, Optional, Sequence

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
TEMPERATURE: float = 0.8
DB_PATH: Path = Path("./data/db/db.sqlite")
OPENAI_API_KEY: str = load_credentials()

tools: Sequence[Any] = [run_query_tool, run_describe_tables_tool]


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
                f"with the following tables: \n{db_tables}. Do not make any "
                "assumptions about the table or column names. Instead, always use "
                "the `describe_tables` function to get the table descriptions. "
            )
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        # variable_name `agent_scratchpad` is compulsory
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # used to track the conversations
    ]
)

# Agents: chains that know how to use tools
agent = OpenAIFunctionsAgent(
    llm=chat_llm,
    tools=tools,
    prompt=prompt,
)

# Executor: runs an agent until the response is NOT a func call
agent_executor = AgentExecutor(
    verbose=True,
    agent=agent,
    tools=tools,
)

query: str = "How many users have provided a shipping address?"
agent_executor(query)
