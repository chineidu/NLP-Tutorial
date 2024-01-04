# Retriever Augmented Generation

> **Credits**: The course content and images can be found [here](https://www.udemy.com/course/chatgpt-and-langchain-the-complete-developers-masterclass/learn/lecture/40261290#overview).

## Table of Content

- [Retriever Augmented Generation](#retriever-augmented-generation)
  - [Table of Content](#table-of-content)
  - [Facts Retriver App](#facts-retriver-app)
  - [E-commerce App](#e-commerce-app)
    - [Over All Logic](#over-all-logic)
    - [E-commerce Flowchart](#e-commerce-flowchart)
    - [Behind The Scenes](#behind-the-scenes)
    - [E-commerce JSON Schema](#e-commerce-json-schema)
      - [Langchain Behind The Scenes](#langchain-behind-the-scenes)
    - [Error Handling 1](#error-handling-1)
    - [Error Handling 2](#error-handling-2)
    - [Add Better Descriptions For Tools Arguments Using Schema](#add-better-descriptions-for-tools-arguments-using-schema)
    - [Run App](#run-app)
    - [Adding Callbacks](#adding-callbacks)


## Facts Retriver App

[![image.png](https://i.postimg.cc/C1265dwx/image.png)](https://postimg.cc/ZBrLDb9G)

```sh
# Run Program
python notebook/Projects/Retriever/Simple_doc_retriever/main.py
```

## E-commerce App

### Over All Logic

[![image.png](https://i.postimg.cc/4yZJzxTv/image.png)](https://postimg.cc/1nChSPX4)

### E-commerce Flowchart

[![image.png](https://i.postimg.cc/L6bcS80y/image.png)](https://postimg.cc/9ry8tC2T)

### Behind The Scenes

[![image.png](https://i.postimg.cc/W4BrcgM2/image.png)](https://postimg.cc/Z9L0r97g)

### E-commerce JSON Schema

- [Validator](https://transform.tools/json-to-json-schema)

- **`JSON Schema`** is a vocabulary that defines the structure, constraints, and expectations for JSON data.

- It acts as a `blueprint` or a `set of rules` that describe how JSON data should be `formatted` and `structured`.

[![image.png](https://i.postimg.cc/MGH4FMwZ/image.png)](https://postimg.cc/4KCBK3nj)

#### Langchain Behind The Scenes

[![image.png](https://i.postimg.cc/90MgsQr2/image.png)](https://postimg.cc/phwBFxfc)

### Error Handling 1

- Send the error message to the LLM with the hope that it can suggest a possible solution.

[![image.png](https://i.postimg.cc/3r95qfH2/image.png)](https://postimg.cc/Z09MBcnq)

### Error Handling 2

- Add a descriptive System Message.

- Add a new tool (descriptive tool).

[![Screenshot-2024-01-03-at-1-45-44-PM.png](https://i.postimg.cc/BbPztpXn/Screenshot-2024-01-03-at-1-45-44-PM.png)](https://postimg.cc/ctZhbQRp)

```text
Table Names:
------------
'users', 'addresses', 'products', 'carts', 'orders', 'order_products'
```

### Add Better Descriptions For Tools Arguments Using Schema

```python
from pydantic import BaseModel

class WriteHTMLReport(BaseModel)
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
    args_schema=WriteHTMLReport, # Add schema!
```

### Run App

```sh
# Run app
python notebook/Projects/Retriever/E_commerce_app/main.py
```

### Adding Callbacks

[![image.png](https://i.postimg.cc/P56f5C1F/image.png)](https://postimg.cc/bZtf6NCQ)
