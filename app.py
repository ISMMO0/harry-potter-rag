import os
import time

import streamlit as st

from rag import (
    DenseIndex,
    GenerationError,
    IngestionError,
    KeywordIndex,
    demo_chunks,
    evidence_answer,
    generate_answer,
    ingest,
)


st.set_page_config(page_title="Harry Potter Q&A · RAG Project", page_icon="✦", layout="centered")
st.markdown(
    f"""
    <style>
      .stApp {{
        background:
          radial-gradient(circle at 76% 8%, rgba(34, 94, 174, .25), transparent 34%),
          radial-gradient(circle at 20% 88%, rgba(22, 64, 125, .18), transparent 30%),
          linear-gradient(160deg, #020713 0%, #07152c 52%, #020713 100%);
      }}
      [data-testid="stHeader"] {{background: transparent;}}
      .block-container {{
        max-width: 900px;
        margin-top: 2rem;
        padding: 2.5rem 3rem 3.5rem;
        border: 1px solid rgba(124, 166, 224, .18);
        border-radius: 22px;
        background: rgba(3, 10, 24, .48);
        box-shadow: 0 24px 80px rgba(0, 0, 0, .45);
        backdrop-filter: blur(10px);
      }}
      .hero {{
        min-height: 250px;
        margin: -1.5rem -2rem 2rem;
        border: 1px solid rgba(132, 177, 237, .28);
        border-radius: 18px;
        background:
          radial-gradient(circle at 50% -20%, rgba(99, 159, 239, .38), transparent 58%),
          linear-gradient(135deg, rgba(5, 18, 43, .95), rgba(1, 7, 18, .96));
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1.5rem;
        box-shadow: 0 18px 46px rgba(0, 0, 0, .36);
      }}
      .hero-mark {{color:#dceaff; font-size:.78rem; font-weight:800; letter-spacing:.28em; text-transform:uppercase;}}
      .hero-name {{color:#f4f8ff; font-size:clamp(2.2rem,7vw,4.5rem); font-weight:800; line-height:1; letter-spacing:-.055em; margin:.8rem 0 .45rem;}}
      .hero-name span {{color:#78aef3;}}
      .hero-copy {{color:#9eb7d8; font-size:.95rem;}}
      .project-title {{font-size: 3.15rem; line-height: 1; letter-spacing: -0.045em; margin: .5rem 0 .7rem; color: #f5f8ff;}}
      .project-title span {{font-size: 1.35em; color: #78aef3; text-shadow: 0 0 28px rgba(65, 132, 221, .3);}}
      h2, h3 {{letter-spacing: -0.02em;}}
      [data-testid="stForm"] {{border: 1px solid rgba(93, 146, 218, .4); background: rgba(7, 19, 42, .82); border-radius: 16px; padding: 1.2rem;}}
      [data-testid="stSidebar"] {{background: rgba(1, 6, 16, .97); border-right: 1px solid rgba(111,156,219,.18);}}
      [data-testid="stAlert"] {{border-radius: 12px;}}
      .stButton button, [data-testid="stFormSubmitButton"] button {{border-radius: 10px; font-weight: 700; background: #1d4f91; border-color: #3977c3; color: white;}}
      .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {{background: #2863ad; border-color: #6aa0e6;}}
      .eyebrow {{color: #78a9ea; font-size: .82rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;}}
      .subtitle {{color: #b4c3d8; font-size: 1.05rem; margin: 0 0 2rem 0;}}
      .status {{display:inline-block; padding:.3rem .65rem; border:1px solid rgba(100,151,218,.4); background:rgba(11,31,62,.72); border-radius:999px; color:#cbd9ec; font-size:.82rem; margin-bottom:1rem;}}
      footer {{visibility: hidden;}}
      @media (max-width: 640px) {{
        .block-container {{margin-top: 0; padding: 1.5rem 1rem 2.5rem; border-radius: 0;}}
        .hero {{height: 220px; margin: -1rem 0 1.5rem;}}
        .project-title {{font-size: 2.15rem;}}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

local_ai = os.getenv("ENABLE_LOCAL_AI") == "1"

with st.sidebar:
    st.header("Documents")
    collection = st.radio("Collection", ["Sample notes", "Upload files"], label_visibility="collapsed")
    uploads = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        disabled=collection == "Sample notes",
    )
    if collection == "Upload files":
        st.caption("Up to 10 files, 10 MB each. Files stay in this session.")

    with st.expander("Settings"):
        method = st.selectbox(
            "Retrieval",
            ["Semantic (MiniLM)", "Keyword (BM25)"] if local_ai else ["Keyword (BM25)"],
        )
        mode = st.selectbox(
            "Response",
            ["AI answer", "Source excerpts"] if local_ai else ["Source excerpts"],
        )
        k = st.slider("Sources", 1, 7, 5)

st.markdown(
    '<div class="hero"><div class="hero-mark">The Wizarding Archive</div>'
    '<div class="hero-name">Potter <span>RAG</span></div>'
    '<div class="hero-copy">Local retrieval · grounded answers · visible evidence</div></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">Question Answering Project</div>', unsafe_allow_html=True)
st.markdown('<h1 class="project-title">Harry Potter Q&amp;A · <span>RAG</span></h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ask the archive and get an answer grounded in the retrieved sources.</p>',
    unsafe_allow_html=True,
)

chunks = []
if collection == "Sample notes":
    chunks = demo_chunks()
else:
    if len(uploads or []) > 10:
        st.error("Upload at most 10 files.")
        st.stop()
    names = [file.name for file in uploads or []]
    if len(names) != len(set(names)):
        st.error("Each uploaded file needs a unique filename.")
        st.stop()
    for file in uploads or []:
        try:
            chunks.extend(ingest(file.name, file.getvalue()))
        except IngestionError as exc:
            st.error(f"{file.name}: {exc}")
    if not chunks:
        st.info("Upload a document to begin.")
        st.stop()

if len(chunks) > 5_000:
    st.error("This demo supports up to 5,000 passages. Use fewer documents.")
    st.stop()

source_count = len({chunk.source for chunk in chunks})
response_label = "AI answers on" if mode == "AI answer" else "Source search"
st.markdown(
    f'<div class="status">{response_label} &nbsp;·&nbsp; {source_count} documents</div>',
    unsafe_allow_html=True,
)

with st.form("question_form"):
    question = st.text_input(
        "Your question",
        placeholder="What does the Sorting Hat do?",
        max_chars=1_000,
    )
    submit = st.form_submit_button("Ask", type="primary", use_container_width=True)

st.caption("Try: What is a Horcrux?  ·  Which position does Harry play?  ·  What does the Patronus Charm do?")

if submit:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        start = time.perf_counter()
        hits = []
        try:
            with st.spinner("Finding the best passages…"):
                # Version the cached index so retrieval implementation changes
                # rebuild it in existing Streamlit sessions after hot reload.
                key = ("index-v3", method, tuple(chunks))
                if st.session_state.get("index_key") != key:
                    st.session_state["index"] = (
                        DenseIndex(chunks) if method.startswith("Semantic") else KeywordIndex(chunks)
                    )
                    st.session_state["index_key"] = key
                hits = st.session_state["index"].search(question, k)

            if mode == "AI answer":
                with st.spinner("Writing an answer…"):
                    answer = generate_answer(question, hits, os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"))
                st.subheader("Answer")
                with st.container(border=True):
                    st.markdown(answer)
            else:
                st.subheader("Source matches")
                st.caption("Search results only—not an AI-written answer.")
                st.text(evidence_answer(hits))

            st.subheader("Sources")
            if not hits:
                st.caption("No supporting passages found.")
            for number, (chunk, score) in enumerate(hits, 1):
                page = f" · page {chunk.page}" if chunk.page else ""
                section = f" · {chunk.section}" if chunk.section else ""
                with st.expander(f"[{number}] {chunk.source}{section}{page}"):
                    st.write(chunk.text)
                    st.caption(f"Retrieval score: {score:.3f} · ranking signal, not confidence")
            st.caption(f"{len(hits)} passages · {time.perf_counter() - start:.2f}s")

        except GenerationError as exc:
            st.error(str(exc))
            if hits:
                st.subheader("Retrieved sources")
                for number, (chunk, _) in enumerate(hits, 1):
                    with st.expander(f"[{number}] {chunk.source} · page {chunk.page}"):
                        st.write(chunk.text)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
