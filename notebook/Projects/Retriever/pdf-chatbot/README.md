# First Time Setup

## Table of Content

- [First Time Setup](#first-time-setup)
  - [Table of Content](#table-of-content)
  - [Using Poetry \[Recommended\]](#using-poetry-recommended)
  - [Using Venv \[Optional\]](#using-venv-optional)
  - [Running the app \[Poetry\]](#running-the-app-poetry)
    - [To run the Python server](#to-run-the-python-server)
    - [To run the worker](#to-run-the-worker)
    - [To run Redis](#to-run-redis)
    - [To reset the database](#to-reset-the-database)
  - [Running the app \[Venv\]](#running-the-app-venv)
    - [To run the Python server \[Venv\]](#to-run-the-python-server-venv)
    - [To run the worker \[Venv\]](#to-run-the-worker-venv)
    - [To run Redis \[Venv\]](#to-run-redis-venv)
    - [To reset the database \[Venv\]](#to-reset-the-database-venv)

## Using Poetry [Recommended]

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

## Running the app [Poetry]

There are three separate processes that need to be running for the app to work: the server, the worker, and Redis.

If you stop any of these processes, you will need to start them back up!

Commands to start each are listed below. If you need to stop them, select the terminal window the process is running in and press Control-C

### To run the Python server

Open a new terminal window and create a new virtual environment:

```sh
poetry shell
```

Then:

```sh
inv dev
```

### To run the worker

- Install redis

```sh
brew install redis

# Start the redis server
redis-server
```

- Open a new terminal window and create a new virtual environment:

```sh
poetry shell
```

Then:

```sh
inv devworker
```

### To run Redis

```sh
redis-server
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
