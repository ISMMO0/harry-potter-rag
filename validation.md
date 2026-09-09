# Validation record — 2026-09-08

## Implemented and tested

`./.venv/bin/python -m pytest -q` → **20 passed in 0.34s** on Python 3.9.6 (Apple Silicon macOS).

Coverage includes UTF-8 text/Markdown ingestion, malformed/empty/unsupported/oversized inputs, a generated two-page PDF with physical-page provenance, chunk overlap and IDs, BM25 ranking/top-k/no-match behavior, the dense-index boundary, Ollama request construction, prompt/evidence separation for an embedded instruction, abstention, citation validation, connection/malformed-response failures, evaluation-dataset coverage, and Streamlit's main and empty/unsupported question flows. Model HTTP behavior is mocked in unit tests; that is not counted as live AI verification.

The desktop browser flow successfully submitted “What does the Sorting Hat do?”, displayed a clearly labeled source excerpt, and exposed `hogwarts.md`, page 1, in an expandable source. At 390×844, measured page width and scroll width were both 390 pixels. This was a local in-app browser check, not a cross-browser/device lab.

## Real embedding evaluation

`./.venv/bin/python evaluate.py --retriever semantic --split heldout --output evaluation-semantic-results.json` completed using real `sentence-transformers/all-MiniLM-L6-v2` inference.

| Held-out metric | Semantic MiniLM | BM25 baseline |
| --- | ---: | ---: |
| Mean source recall@3, 16 answerable questions | 0.9375 | 1.0000 |
| Top-1 source accuracy, 14 single-source questions | 0.9286 | 1.0000 |
| No retrieval match, 4 unsupported questions | 0.5000 | 0.5000 |
| Index construction | 19.19 s | 0.0008 s |
| Mean query latency | 114.44 ms | 0.010 ms |

Environment: Python 3.9.6, macOS arm64, five tiny original notes. The first semantic run included model loading; per-query timings followed index construction. Exact cases are in `evaluation-semantic-results.json` and `evaluation-results.json`.

BM25's perfect answerable retrieval here is specific to this tiny lexical dataset. Both retrievers return passages for familiar-entity questions whose requested facts are absent, demonstrating that retrieval scores are rankings—not confidence or answerability probabilities.

## Real local generation integration

The real Ollama generation path is implemented, has contract/error tests, and is reproducible with:

```bash
ollama pull qwen2.5:1.5b
ENABLE_LOCAL_AI=1 .venv/bin/streamlit run app.py
.venv/bin/python smoke_local_ai.py --output local-ai-results.json
```

Ollama 0.33.3 and `qwen2.5:1.5b` were installed and the command completed in 6.89 seconds. MiniLM retrieved `hogwarts.md` for “What does the Sorting Hat do, and which house is Harry in?” Qwen answered that the Hat assigns students based on qualities/preferences and that Harry is in Gryffindor, citing `[1]`. Both claims were manually verified against the retrieved passage. The full record is in `local-ai-results.json`. This verifies one end-to-end example only; it is not an answer-accuracy, citation-support-rate, or abstention-rate evaluation.

## Account-dependent / not performed

No Git repository existed in the supplied folder, no remote was identified, and no GitHub push or Streamlit deployment was authorized. No live URL exists.

## Interpretation limits

The 10 development and 20 held-out prompts were hand-authored against five tiny notes. The split is useful for regression checking but is not an independent benchmark. Retrieval metrics are separate from generation. Citation ID validity does not prove factual entailment, and the prompt-injection boundary reduces risk without guaranteeing protection.
