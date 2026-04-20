# langchain-synapt

LangChain chat message history backed by [synapt](https://synapt.dev) recall memory.

## Install

```bash
pip install langchain-synapt
```

Or as an extra on the main package:

```bash
pip install synapt[langchain]
```

## Usage

```python
from langchain_core.messages import HumanMessage
from langchain_synapt import SynaptChatMessageHistory

history = SynaptChatMessageHistory(session_id="user-123")
history.add_messages([HumanMessage(content="hello")])
print(history.messages)

# Semantic search across all recall-indexed sessions
results = history.search("deployment config")

# Persist a decision as a durable knowledge node
history.save_to_recall("Always use UTC timestamps", category="convention")
```

`SynaptChatMessageHistory` implements LangChain's `BaseChatMessageHistory` and works with `RunnableWithMessageHistory`, `ConversationChain`, and any component that accepts a chat history backend.

## Features

- SQLite storage with WAL mode for concurrent access
- Session isolation with multi-session support
- Semantic search across all recall-indexed sessions
- Knowledge persistence via recall's durable knowledge nodes
- Full message metadata preservation (tool calls, response metadata, IDs)

## Links

- [synapt documentation](https://synapt.dev/guide.html)
- [GitHub](https://github.com/synapt-dev/recall)
