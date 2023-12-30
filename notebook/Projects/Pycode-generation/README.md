# CODE-GEN

## Table of Content

- [CODE-GEN](#code-gen)
  - [Table of Content](#table-of-content)
  - [Code Generator Program](#code-generator-program)
  - [Terminal Chatbot Program](#terminal-chatbot-program)

## Code Generator Program

```sh
# get help
python notebook/Projects/Pycode-generation/Simple_code_gen/main.py --help

# Run Program
python notebook/Projects/Pycode-generation/Simple_code_gen/main.py \
  --language rust --task 'a fibonaci sequence function' code-gen

python notebook/Projects/Pycode-generation/Simple_code_gen/main.py \
  --language go --task 'read a parquet file from S3' test-gen
```

## Terminal Chatbot Program

```sh
# get help
python main.py --help

# Run Program
# main.py --language rust --task 'a fibonaci sequuence function' code-gen
# python main.py --language 'typescript' --task 'a fibonaci sequuence function' test-gen
```
