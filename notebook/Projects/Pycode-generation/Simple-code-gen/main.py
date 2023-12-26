import os
from typing import Any, Optional

import click
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from typeguard import typechecked


@typechecked
def load_credentials() -> Optional[str]:
    """This is used to load the API key(s)"""
    _ = load_dotenv(find_dotenv())  # read local .env file
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    return OPENAI_API_KEY


@click.group()
@click.option("--language", help="Name of programming language", default="python")
@click.option("--task", help="The task you want to perform", default="generate a list of numbers")
@click.pass_context
def cli(ctx, language: str, task: str) -> None:
    """Click command object."""
    ctx.ensure_object(dict)
    ctx.obj["LANGUAGE"] = language
    ctx.obj["TASK"] = task


# @typechecked
# @cli.command()
# @click.option("--language", help="Name of programming language", default="python")
# @click.option("--task", help="The task you want to perform", default="generate a list of numbers")
# def code_gen(language: str, task: str) -> str:
#     """This function is used to create a code generation program using Langchain and an LLM."""
#     OPENAI_API_KEY: str = load_credentials()
#     # Create LLM
#     llm: OpenAI = OpenAI(openai_api_key=OPENAI_API_KEY)

#     # Used for text completion LLMs
#     prompt_template: str = """Write a very short {language} function that will {task}"""
#     code_prompt: PromptTemplate = PromptTemplate(
#         input_variables=["language", "task"], template=prompt_template
#     )
#     code_chain: LLMChain = LLMChain(llm=llm, prompt=code_prompt)

#     result = code_chain(inputs={"language": language, "task": task}).get("text")
#     click.secho(message=result, fg="green")
#     return result


@typechecked
@cli.command()
@click.pass_context
def code_gen(ctx: Any) -> str:
    """This function is used to create a code generation program using Langchain and an LLM."""
    OPENAI_API_KEY: str = load_credentials()
    # Create LLM
    llm: OpenAI = OpenAI(openai_api_key=OPENAI_API_KEY)

    # Used for text completion LLMs
    prompt_template: str = """Write a very short {language} function that will {task}"""
    code_prompt: PromptTemplate = PromptTemplate(
        input_variables=["language", "task"], template=prompt_template
    )
    code_chain: LLMChain = LLMChain(llm=llm, prompt=code_prompt)

    result = code_chain(
        inputs={"language": ctx.obj.get("LANGUAGE"), "task": ctx.obj.get("TASK")}
    ).get("text")
    click.secho(message=result, fg="green")
    return result


if __name__ == "__main__":
    cli(obj={})
