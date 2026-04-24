"""synapt.integrations — framework adapters for recall memory.

Each submodule wraps recall's search/save API for a specific framework:
- anthropic: Memory Tool backend for the Anthropic SDK
- langchain: BaseChatMessageHistory adapter (langchain-synapt on PyPI)
- openai_agents: Session adapter for the OpenAI Agents SDK
- google_adk: MemoryService + SessionService for Google ADK
"""
