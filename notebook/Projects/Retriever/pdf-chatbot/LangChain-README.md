# LangChain Tutorial

## Table of Content

- [LangChain Tutorial](#langchain-tutorial)
  - [Table of Content](#table-of-content)
  - [1. Access An LLM Without Chains](#1-access-an-llm-without-chains)
  - [Access An LLM With Chains](#access-an-llm-with-chains)
    - [Advantages of LangChain](#advantages-of-langchain)
    - [LLM Chain](#llm-chain)
    - [Connecting Chains Together](#connecting-chains-together)
    - [ChatPrompt Tempplate](#chatprompt-tempplate)
      - [Example of ChatPrompt Template](#example-of-chatprompt-template)

## 1. Access An LLM Without Chains

[![image.png](https://i.postimg.cc/RVLp2WgH/image.png)](https://postimg.cc/0brCwyRk)

## Access An LLM With Chains

### Advantages of LangChain

- Ease of Use

- Simplified Development: LangChain provides a high-level API that abstracts away the complexity of working with LLMs. This allows developers with less machine learning expertise to build sophisticated applications.

- Wide Range of Applications: LangChain is adaptable and can be used to build various applications, from chatbots and question-answering systems to content generation and data analysis tools.

- Model Agnostic: LangChain is not tied to a specific LLM provider. You can integrate with different LLMs (e.g., OpenAI API, Google AI) within the same interface.

### LLM Chain

- It contains the required arguments:
  - `LLM`
  - `Prompt Template`

```py
# Used for text completion LLMs
prompt_template: str = """Write a very short {language} function that will {task}"""
code_prompt = PromptTemplate(
    input_variables=["language", "task"], template=prompt_template
)
code_chain = LLMChain(llm=llm, prompt=code_prompt)

# Hardcoded inputs!
language, task = ("Python", "generate a list of numbers")

result = code_chain(inputs={"language": language, "task": task})
```

### Connecting Chains Together

- e,g using `SequentialChain`

[![image.png](https://i.postimg.cc/8Cdgm9wV/image.png)](https://postimg.cc/GBpfcMZM)

```py
from langchain.chains import SequentialChain

# Used for text completion LLMs
code_prompt_template: str = (
    """Write a very short {language} function that will {task}"""
)
unittest_prompt_template: str = (
    """Write a unit test for the following {language} code: \n{code}"""
)

# Propmt(s)
code_prompt = PromptTemplate(
    input_variables=["language", "task"], template=code_prompt_template
)
unittest_prompt = PromptTemplate(
    input_variables=["language", "code"], template=unittest_prompt_template
)

# Chain(s)
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code",  # from text to code
)
unittest_chain = LLMChain(
    llm=llm,
    prompt=unittest_prompt,
    output_key="test",  # from text to test
)
# Feed the output of code_chain into unittest_chain
final_chain = SequentialChain(
    chains=[code_chain, unittest_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"], # output keys
)
# Hardcoded inputs!
language, task = ("Python", "generate a list of numbers")

result = final_chain(inputs={"language": language, "task": task})
print(result)
```

### ChatPrompt Tempplate

[![image.png](https://i.postimg.cc/s2nTM32Z/image.png)](https://postimg.cc/4K96MkVJ)

[![image.png](https://i.postimg.cc/FRgxXmJ6/image.png)](https://postimg.cc/YG0FQ5RN)

#### Example of ChatPrompt Template

```py
TEMPERATURE: float = 0.5
OPENAI_CHAT_MODEL: str = "gpt-3.5-turbo"

chat_llm = ChatOpenAI(
    model=OPENAI_CHAT_MODEL,
    temperature=TEMPERATURE,
)

# Used for chat completion LLMs
prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[HumanMessagePromptTemplate.from_template("{content}")],
)
chain: LLMChain = LLMChain(llm=chat_llm, prompt=prompt)
content: str = "Hello, who are you?"

result: dict[str, Any] = chain({"content": content})
print(result)
# {
#     'content': 'Hello, who are you?',
#     'text': 'Hello! I am an AI assistant created by OpenAI. I am here to help answer any questions you may have.
# How can I assist you today?'
# }
```
