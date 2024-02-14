from typing import Union

from app.chat.models import ChatArgs
from app.web.api import add_message_to_conversation, get_messages_by_conversation_id
from app.web.db.models import Message
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class SqlMessageHistory(BaseChatMessageHistory, BaseModel):
    conversation_id: str

    @property
    def messages(self) -> Union[AIMessage, HumanMessage, SystemMessage]:
        return get_messages_by_conversation_id(conversation_id=self.conversation_id)

    def add_message(self, message: BaseMessage) -> Message:
        return add_message_to_conversation(
            conversation_id=self.conversation_id, role=message.type, content=message.content
        )

    def clear(self) -> None:
        return NotImplementedError


def build_memory(chat_args: ChatArgs):
    return ConversationBufferMemory(
        chat_memory=SqlMessageHistory(conversation_id=chat_args.conversation_id),
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
    )
