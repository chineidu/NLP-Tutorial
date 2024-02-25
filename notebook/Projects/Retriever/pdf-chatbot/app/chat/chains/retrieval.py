from langchain.chains import ConversationalRetrievalChain

from app.chat.chains.streamable import StreamableChainMixin


class StreamingConversationsalRetrievalChain(StreamableChainMixin, ConversationalRetrievalChain):
    """This is a ConversationalRetrievalChain with streaming capabilities."""

    pass
