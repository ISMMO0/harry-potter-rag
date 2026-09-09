"""Opt-in live test: RUN_LOCAL_AI_INTEGRATION=1 python -m pytest -m local_ai."""
import os
import pytest

from rag import DenseIndex, demo_chunks, generate_answer


@pytest.mark.local_ai
@pytest.mark.skipif(os.getenv("RUN_LOCAL_AI_INTEGRATION") != "1", reason="requires downloaded MiniLM and running Ollama")
def test_real_embeddings_and_llm_together():
    question = "What does the Sorting Hat do?"
    hits = DenseIndex(demo_chunks()).search(question, k=3)
    assert hits and hits[0][0].source == "hogwarts.md"
    answer = generate_answer(question, hits, os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"))
    assert "[" in answer
