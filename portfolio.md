# Portfolio copy and interview notes

## GitHub description

Inspectable local RAG app for PDF/TXT/Markdown: MiniLM semantic retrieval, Ollama-generated cited answers, Streamlit UI, tests, and retrieval evaluation—with an honest no-key source-search demo mode.

## CV bullets

- Built a Python/Streamlit document QA application with in-memory PDF/TXT/Markdown ingestion, physical-page provenance, overlapping chunks, BM25 and MiniLM retrieval, and inspectable source citations.
- Implemented a local Ollama generation path with evidence-only prompting, explicit abstention, citation-ID validation, prompt/document separation, and user-visible failure fallback to retrieved evidence.
- Created 22 passing automated tests and a 30-question development/held-out retrieval set. Historical five-note results are retained but are not presented as metrics for the current seven-summary corpus.

## LinkedIn launch draft

I built **Potter RAG — Ask the Archive**, a small document question-answering project designed to be inspectable rather than over-engineered.

It ingests PDF, TXT, and Markdown files, preserves page/source metadata, compares BM25 with MiniLM semantic retrieval, and can generate cited answers through a local Ollama model. Every retrieved passage remains visible, and the public-safe mode is clearly labeled as source search—not generative AI.

I also added 22 passing automated tests and a 30-question retrieval evaluation with paraphrases, multi-source questions, unsupported familiar entities, and a document-instruction attack case. The checked-in metrics are historical results from the earlier five-note corpus, not scores for the current seven-summary corpus or a claim about answer accuracy.

The project reinforced an important RAG lesson: retrieving a related passage does not prove the question is answerable, and a valid citation number does not prove the answer is supported.

Repository: https://github.com/ISMMO0/harry-potter-rag
Demo: [add URL after deploying the source-search mode]

## Pipeline in plain English

The app reads each authorized document in memory. PDFs are extracted by physical page; text files use page 1. Markdown is separated by heading and split into approximately 600-character passages with 100-character overlap. BM25 ranks shared informative words, while MiniLM embeds questions and passages and ranks cosine similarity. A hybrid score combines both signals. The top five passages are shown directly or placed in an evidence block for Ollama. The model must use only that evidence, abstain when insufficient, and cite passage numbers. Code checks that citation IDs exist, then shows passages for human verification.

## Why these choices

- **MiniLM:** small 384-dimensional embeddings and practical laptop inference, with better paraphrase potential than keyword matching.
- **~600 characters / 100 overlap:** keeps passages focused, prefers sentence boundaries, and preserves nearby context for a small local model.
- **Exact cosine search:** transparent and adequate for thousands, not millions, of chunks.
- **Hybrid MiniLM + BM25:** balances semantic similarity with exact names and short factual questions; BM25 remains the no-model hosted fallback.
- **Qwen 2.5 1.5B/Ollama:** free local execution with a modest footprint; real end-to-end questions were verified on this machine.
- **Recall@3/top-1/no-match:** isolates whether retrieval surfaces evidence; correctness, citation support and abstention need separate reviewed generation labels.

## Five likely interview questions

### 1. Why avoid a vector database?

The limit is 5,000 chunks. A normalized NumPy matrix and exact cosine similarity are inspectable and adequate. A vector service would add lifecycle and deployment complexity without solving a measured problem.

### 2. How do you reduce hallucination?

The model receives only retrieved evidence, must cite each factual sentence or abstain, runs at temperature zero, and has citation IDs validated. Sources remain visible. These controls reduce risk but do not guarantee factuality or entailment.

### 3. What about prompt injection in uploads?

Documents are evidence-delimited and explicitly labeled untrusted; a test checks that separation. This is a basic defense, not complete protection. Production work would add adversarial evaluation and stronger input/output policy controls.

### 4. Why can an unsupported question retrieve a passage?

Similarity asks which passage is closest, not whether it contains the requested fact. Hermione's birth date can retrieve Hermione text with no date. Retrieval no-match and generation abstention are therefore separate metrics.

### 5. What comes next?

Run and manually label the Ollama integration set for correctness, citation support and abstention; add a larger authorized corpus; tune only on development cases; consider hybrid fusion/reranking if measured failures justify it; and preserve an untouched test set.
