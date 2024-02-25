from typing import Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


class StreamingHandler(BaseCallbackHandler):
    """This is a handler that handles streaming."""

    def __init__(self, queue):
        self.queue = queue
        self.streaming_ids = set()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        run_id: str,
        **kwargs: Any,
    ) -> Any:
        # Select ONLY the streaming LLM
        if serialized.get("kwargs").get("streaming"):
            self.streaming_ids.add(run_id)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.queue.put(token)

    def on_llm_end(self, response: LLMResult, run_id: str, **kwargs: Any) -> Any:
        if run_id in self.streaming_ids:
            self.queue.put(None)
            self.streaming_ids.remove(run_id)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        self.queue.put(None)
