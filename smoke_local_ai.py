"""Real-model integration check. This never substitutes a mock for a failed run."""
import argparse
import json
import os
from pathlib import Path
import platform
import time
from rag import DEFAULT_LLM_MODEL, EMBEDDING_MODEL, DenseIndex, demo_chunks, generate_answer

def main(output=None):
    question = 'What does the Sorting Hat do, and which house is Harry in?'
    started = time.perf_counter()
    index = DenseIndex(demo_chunks())
    hits = index.search(question, k=3)
    if not hits or hits[0][0].source != 'hogwarts.md':
        raise RuntimeError('Semantic retrieval did not return the expected source.')
    llm_model = os.getenv('OLLAMA_MODEL', DEFAULT_LLM_MODEL)
    answer = generate_answer(question, hits, llm_model)
    result = {
        'status': 'completed-needs-human-support-review',
        'environment': {'python': platform.python_version(), 'platform': platform.platform()},
        'embedding_model': EMBEDDING_MODEL, 'llm_model': llm_model,
        'question': question,
        'retrieved': [{'id': i, 'source': chunk.source, 'page': chunk.page, 'score': score, 'text': chunk.text}
                      for i, (chunk, score) in enumerate(hits, 1)],
        'answer': answer,
        'citation_ids_valid': True,
        'evidence_supports_answer': None,
        'limitation': 'Set evidence_supports_answer manually after reviewing every factual claim; valid citation IDs do not prove support.',
        'elapsed_seconds': time.perf_counter() - started,
    }
    rendered = json.dumps(result, indent=2)
    if output:
        Path(output).write_text(rendered + '\n')
    print(rendered)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    args = parser.parse_args()
    main(args.output)
