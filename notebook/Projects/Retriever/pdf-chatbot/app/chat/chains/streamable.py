from queue import Queue
from threading import Thread
from typing import Iterator

from flask import current_app

from app.chat.callbacks.stream import StreamingHandler


class StreamableChainMixin:
    def stream(self, input) -> Iterator:
        """This is used to stream the input token from the LLM."""
        queue = Queue()
        handler = StreamingHandler(queue)

        def task(app_context) -> None:
            """This is used to call/invoke the chain.

            Note:
            -----
            For Flask apps, it requires an app context.
            """
            app_context.push()  # Gives access to the context inside the thread.
            self(input, callbacks=[handler])

        Thread(target=task, args=[current_app.app_context()]).start()

        while True:
            token = queue.get()
            if token is None:
                break
            yield token
