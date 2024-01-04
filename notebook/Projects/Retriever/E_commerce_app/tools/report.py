from pathlib import Path
from typing import Any

from langchain.tools import StructuredTool
from pydantic import BaseModel
from rich import print
from typeguard import typechecked


@typechecked
def write_html_report(file_name: Path, html: str) -> None:
    """This is used to write the html report to the specified file_name."""
    print(f"[INFO]: Writing html report ...")
    with open(file_name, "w") as fp:
        fp.write(html)


class WriteHTMLReport(BaseModel):
    file_name: Path
    html: str


# Agents with more than one arg
run_write_html_report_tool = StructuredTool.from_function(
    name="write_html_report",
    description=(
        "This is used to write the html report to the specified"
        "file_name whenever someone asks for a report."
    ),
    func=write_html_report,
    args_schema=WriteHTMLReport,  # Add schema!
)
