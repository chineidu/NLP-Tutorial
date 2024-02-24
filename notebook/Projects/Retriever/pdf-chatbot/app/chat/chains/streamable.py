from queue import Queue
from threading import Thread
from app.chat.callbacks.stream import StreamingHandler


class StreamableChain:
    def stream(self, input):
        """This is used to stream the input token from the LLM."""
        queue = Queue()
        handler = StreamingHandler(queue)

        def task():
            """This is used to call/invoke the chain."""
            self(input, callbacks=[handler])

        Thread(target=task).start()

        while True:
            token = queue.get()
            if token is None:
                break
            yield token
