"""Shared agent utilities — ported/adapted from TestAgent's app/agents/_utils/.

Ported near-verbatim (no TestAgent-specific dependencies): workspace.py, hitl.py.
Adapted for a Postgres-less, TestAgent-independent greenfield project:
checkpointer.py (InMemorySaver instead of AsyncPostgresSaver), model_factory.py
(talks to the llm_bridge HTTP proxy instead of importing TestAgent's in-process
browser LLM module — see model_factory.py docstring for why).
"""
