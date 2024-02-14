# PDF Chat Bot

## Table of Content

- [PDF Chat Bot](#pdf-chat-bot)
  - [Table of Content](#table-of-content)
  - [Requirements (To Do)](#requirements-to-do)
  - [1. Initial Step](#1-initial-step)
  - [2. Using Poetry \[Recommended\]](#2-using-poetry-recommended)
  - [Using Venv \[Optional\]](#using-venv-optional)
  - [3. Running the app \[Poetry\]](#3-running-the-app-poetry)
    - [To run the Python server](#to-run-the-python-server)
    - [Run The Vector Database Server](#run-the-vector-database-server)
    - [To run the worker](#to-run-the-worker)
    - [To reset the database](#to-reset-the-database)
  - [Running the app \[Venv\]](#running-the-app-venv)
    - [To run the Python server \[Venv\]](#to-run-the-python-server-venv)
    - [To run the worker \[Venv\]](#to-run-the-worker-venv)
    - [To run Redis \[Venv\]](#to-run-redis-venv)
  - [4 To Do](#4-to-do)
    - [To reset the database \[Venv\]](#to-reset-the-database-venv)

## Requirements (To Do)

- Celery: used for parallel execution of tasks and provides the facility to run programs/jobs in the background when the CPU is idle.
- Add workers to handle multiple requests.
- Document retrieval needs to be scoped to a particular PDF. i.e. so that if a user uploads 2 or more documents, the retrieval system know the exact document to query.
- Messages and chats need to be persisted.
- Handle vague user messages.

## 1. Initial Step

- Create a `.env` file containing all the necessary environment variables by running:

```sh
cp example.env .env
```

## 2. Using Poetry [Recommended]

```sh
# Install dependencies
poetry install

# Create a virtual environment
poetry shell

# Initialize the database
flask --app app.web init-db

```

## Using Venv [Optional]

These instructions are included if you wish to use venv to manage your evironment and dependencies instead of Poetry.

```sh
# Create the venv virtual environment
python -m venv .venv

# On MacOS, WSL, Linux
source .venv/bin/activate

# On Windows
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize the database
flask --app app.web init-db
```

## 3. Running the app [Poetry]

There are three separate processes that need to be running for the app to work: the server, the worker, and Redis.

If you stop any of these processes, you will need to start them back up!

Commands to start each are listed below. If you need to stop them, select the terminal window the process is running in and press Control-C

### To run the Python server

Open a new terminal window and create a new virtual environment:

```sh
poetry shell
inv dev
```

### Run The Vector Database Server

- Run it locally using Docker.

```sh
# Run a Qdrant Docker image
docker run -p 6333:6333 --rm --name vector_db \
    -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### To run the worker

1.) Install redis

```sh
brew install redis

# Start the redis server
redis-server
```

2.) Start up the worker:

- Create a new terminal tab and run:

```sh
poetry shell
inv devworker
```

### To reset the database

Open a new terminal window and create a new virtual environment:

```sh
poetry shell
```

Then:

```sh
flask --app app.web init-db
```

## Running the app [Venv]

_These instructions are included if you wish to use venv to manage your evironment and dependencies instead of Pipenv._

There are three separate processes that need to be running for the app to work: the server, the worker, and Redis.

If you stop any of these processes, you will need to start them back up!

Commands to start each are listed below. If you need to stop them, select the terminal window the process is running in and press Control-C

### To run the Python server [Venv]

Open a new terminal window and create a new virtual environment:

```sh
# On MacOS, WSL, Linux
source .venv/bin/activate

# On Windows
.\.venv\Scripts\activate
```

Then:

```sh
inv dev
```

### To run the worker [Venv]

```sh
brew install redis

# Start the redis server
redis-server
```

- Open a new terminal window and create a new virtual environment:

```sh
# On MacOS, WSL, Linux
source .venv/bin/activate

# On Windows
.\.venv\Scripts\activate
```

Then:

```sh
inv devworker
```

### To run Redis [Venv]

```sh
redis-server
```

## 4 To Do

- Custom Message Histories (88)

### To reset the database [Venv]

Open a new terminal window and create a new virtual environment:

```sh
# On MacOS, WSL, Linux
source .venv/bin/activate

# On Windows
.\.venv\Scripts\activate
```

Then:

```sh
flask --app app.web init-db
```
