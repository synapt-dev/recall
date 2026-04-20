"""LangChain integration for synapt recall memory.

Thin wrapper that re-exports SynaptChatMessageHistory from the synapt
package. Install via ``pip install langchain-synapt`` or
``pip install synapt[langchain]``.

Usage::

    from langchain_synapt import SynaptChatMessageHistory

    history = SynaptChatMessageHistory(session_id="user-123")
    history.add_messages([HumanMessage(content="hello")])

    # Semantic search across all recall-indexed sessions
    results = history.search("deployment config")

    # Persist a decision as a durable knowledge node
    history.save_to_recall("Always use UTC timestamps", category="convention")
"""

from synapt.integrations.langchain import SynaptChatMessageHistory

__all__ = ["SynaptChatMessageHistory"]
__version__ = "0.1.0"
