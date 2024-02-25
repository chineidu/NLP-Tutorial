import os

from invoke import task


@task
def dev(ctx) -> None:
    ctx.run(
        "flask --app app.web run --debug --port 8000",
        pty=os.name != "nt",
        env={"APP_ENV": "development"},
    )


@task
def devworker(ctx) -> None:
    ctx.run(
        (
            "watchmedo auto-restart --directory=./app --pattern=*.py --recursive"
            " -- celery -A app.celery.worker worker --concurrency=1 --loglevel=INFO"
        ),
        pty=os.name != "nt",
        env={"APP_ENV": "development"},
    )
