import sqlite3
from pathlib import Path
from typing import Any, Optional, Union

import langchain
from langchain.tools import Tool
from pydantic import BaseModel
from rich import print
from typeguard import typechecked

# langchain.debug = True
# Params
DB_PATH: Path = Path("./data/db/db.sqlite")

# Create connection to the DB
try:
    print(f"[INFO]: Creating connection to the DB ...")
    conn = sqlite3.connect(DB_PATH)
except sqlite3.OperationalError as err:
    print(f"[ERROR]: Error loading DB. {err}")


@typechecked
def run_sqlite_query(query: str) -> Union[list[Any], str]:
    """This is used to run sqlite queries."""
    print(f"[INFO]: Running `run_sqlite_query` ...")
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        result: list[Any] = cursor.fetchall()  # type: ignore
        return result

    except sqlite3.OperationalError as err:
        print(f"[ERROR]: {err}")
        return f"{err}"


@typechecked
def list_DB_tables() -> Optional[str]:
    """This returns the list of table names present in the database."""
    print(f"[INFO]: Running `list_DB_tables` ...")
    cursor = conn.cursor()
    query: str = """
                    SELECT name FROM sqlite_master
                        WHERE type='table';
                """
    cursor.execute(query)
    tables: Optional[list[str]] = cursor.fetchall()
    result: str = ", ".join([f"'{x[0]}'" for x in tables])  # type: ignore
    return result


@typechecked
def describe_tables(db_table: str) -> str:
    """This is used to obtain the schema of the tables in the database."""
    print(f"[INFO]: Running `describe_tables` ...")
    cursor = conn.cursor()
    query: str = f"""
                    SELECT sql FROM sqlite_master
                        WHERE type='table' AND name IN ('{db_table}');
                """
    queryy: str = f"""
                    SELECT sql FROM sqlite_master
                        WHERE type='table' AND name IN ('products');
                """
    try:
        cursor.execute(query)
        tables: Optional[list[str]] = cursor.fetchall()
        result: str = "\n\n".join([x[0] for x in tables])  # type: ignore
        return result

    except sqlite3.OperationalError as err:
        print(f"[ERROR]: {err}")
        return f"{err}"


# Schema
class SqliteQuerySchema(BaseModel):
    query: str


class DescribeTablesSchema(BaseModel):
    db_tables: str


# Create tools with one arg
run_query_tool: Tool = Tool.from_function(
    name="run_sqlite_query",
    description="This is used to run sqlite queries.",
    func=run_sqlite_query,
    args_schema=SqliteQuerySchema,  # Add schema!
)

run_describe_tables_tool: Tool = Tool.from_function(
    name="describe_tables",
    description=(
        "This is used to obtain the schema of the tables and "
        "the respective columns in the database."
    ),
    func=describe_tables,
    args_schema=DescribeTablesSchema,  # Add schema!
)


# print(describe_tables(db_tables=list_DB_tables()))

# print(list_DB_tables())
