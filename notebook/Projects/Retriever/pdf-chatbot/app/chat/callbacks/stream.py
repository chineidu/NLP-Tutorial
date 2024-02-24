from typing import Any
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult


class StreamingHandler(BaseCallbackHandler):
    """This is a handler that handles streaming."""

    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.queue.put(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.queue.put(None)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        self.queue.put(None)
