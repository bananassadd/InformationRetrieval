import streamlit as st
import pandas as pd
import os
import time
import html
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker
import torch
import search_utils as su

# Page Configuration
st.set_page_config(
    page_title="ArXiv Search",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load external CSS (Ensure you have saved your CSS inside style.css in the same folder)
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass # App will still run if style.css is missing, just without the custom styling

# Data Loading (Updated for cleaned dataset and correct column)
@st.cache_data
def load_data():
    df_raw = pd.read_csv("cleaned_arxiv_corpus.csv").dropna(subset=['content']).reset_index(drop=True)
    return df_raw

df = load_data()
documents = df['content'].tolist()
titles = df['title'].tolist()
links = df['url'].tolist() if 'url' in df.columns else df['arxiv_id'].tolist()
sub_topics = df['sub_topic'].tolist()

def dense_text(i):
    """Text the dense model reads: title + the start of the abstract (MiniLM only sees ~256 tokens anyway)."""
    t = documents[i]
    a = t.lower().find("abstract", 0, 4000)
    body = t[a:a + 1500] if a != -1 else t[:1500]
    return titles[i] + ". " + " ".join(body.split())

# Model Loading
@st.cache_resource
def load_models():
    bm25 = BM25Okapi([su.tokenize(doc) for doc in documents])
    docs_norm = [su.normalize(doc) for doc in documents]
    vocab = su.build_vocab(documents)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    cache = f"embeddings_cache_{len(documents)}.npy"
    if os.path.exists(cache):
        emb = torch.tensor(np.load(cache))
    else:
        emb = model.encode([dense_text(i) for i in range(len(documents))],
                           convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
        np.save(cache, emb.cpu().numpy())
    return bm25, model, emb, docs_norm, vocab

bm25, dense_model, doc_embeddings, docs_norm, vocab = load_models()

@st.cache_resource
def load_spellchecker():
    sp = SpellChecker()
    # teach it the corpus vocabulary so technical terms (LSTM, BLEU, ROUGE...) are never "corrected"
    sp.word_frequency.load_words([w for w, c in vocab.items() if c >= su.MIN_COUNT])
    return sp

spell = load_spellchecker()

# Header Section
st.markdown("""
<div class="search-hero">
    <div class="search-title">arXiv Search</div>
    <div class="search-subtitle">Computer Science & Computational Linguistics Papers</div>
</div>
""", unsafe_allow_html=True)

# Session state query management
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

def set_query(text):
    st.session_state.search_query = text

# Train Station Ticker for the 5 Suggested Topics
SUGGESTED_TOPICS = [
    ("Machine Translation", "Machine Translation"),
    ("Question Answering", "Question Answering"),
    ("Named Entity Recognition", "Named Entity Recognition"),
    ("Attention Mechanisms", "Attention Mechanisms"),
    ("Text Summarization", "Text Summarization"),
]

if not st.session_state.search_query:
    st.markdown('<div class="popular-topics-label">Popular Topics</div>', unsafe_allow_html=True)

    # 2 sets of topics for continuous seamless automatic marquee loop (0% to -50%)
    ticker_topics = SUGGESTED_TOPICS + SUGGESTED_TOPICS

    with st.container(key="train_station"):
        t_cols = st.columns(len(ticker_topics))
        for idx, (t_label, t_q) in enumerate(ticker_topics):
            with t_cols[idx]:
                if st.button(t_label, key=f"topic_btn_{idx}"):
                    set_query(t_q)
                    st.rerun()

# Search Input Bar
query_input = st.text_input(
    label="Search",
    value=st.session_state.search_query,
    placeholder="Search research papers, topics, or questions...",
    label_visibility="collapsed"
)

# Model Selection
MODES = ["Hybrid (RRF)", "Hybrid (Weighted)", "BM25 (Keywords)", "Dense (Semantic)"]
mode = st.pills(
    label="Model",
    options=MODES,
    default="Hybrid (RRF)",
    label_visibility="collapsed"
) or "Hybrid (RRF)"          # clicking the active pill deselects it -> fall back to the default

# Fixed top_k = 10
top_k = 10

# Search Execution
if query_input:
    t_start = time.time()
    tokens, shown_q, corrections, expansions = su.clean_query(query_input, spell, vocab)
    phrases = su.extract_phrases(query_input)

    # BM25 scores
    bm25_scores = list(bm25.get_scores(tokens))

    # Dense scores (cosine similarity on normalised embeddings)
    query_emb = dense_model.encode(" ".join(tokens), convert_to_tensor=True, normalize_embeddings=True)
    dense_scores = util.dot_score(query_emb, doc_embeddings)[0].cpu().tolist()

    # Fusion
    rrf_scores = su.rrf(bm25_scores, dense_scores, k=60)
    weighted_scores = su.weighted(bm25_scores, dense_scores, alpha=0.5)

    if mode.startswith("Hybrid (RRF)"):
        main = rrf_scores
    elif mode.startswith("Hybrid (Weighted)"):
        main = weighted_scores
    elif mode.startswith("BM25"):
        main = bm25_scores
    else:
        main = dense_scores

    # Phrase matching
    candidates = list(range(len(documents)))
    phrase_note = ""
    if phrases:
        kept = [i for i in candidates if all(su.contains_phrase(docs_norm[i], p) for p in phrases)]
        if kept:
            candidates = kept
            phrase_note = f" - exact phrase filter: {len(kept)} matching papers"
        else:
            phrase_note = " - no paper contains that exact phrase, showing normal results"
    else:
        whole = su.normalize(shown_q)
        if len(whole.split()) > 1 and not mode.startswith("Dense"):
            if mode.startswith("Hybrid (RRF)"):
                bonus = 1 / 60
            elif mode.startswith("Hybrid (Weighted)"):
                bonus = 0.15
            else:  # BM25
                bonus = 0.25 * max(main) if max(main) > 0 else 0
            main = [m + (bonus if su.contains_phrase(docs_norm[i], whole) else 0) for i, m in enumerate(main)]

    ranked = sorted(candidates, key=lambda i: main[i], reverse=True)[:top_k]
    elapsed = time.time() - t_start

    # Result summary (shows what the query handling did)
    notes = []
    if expansions:
        notes.append("expanded " + ", ".join(f"{a.upper()} → {b}" for a, b in expansions.items()))
    if corrections:
        notes.append("corrected " + ", ".join(f"{a} → {b}" for a, b in corrections.items()))
    note_html = f" ({'; '.join(notes)})" if notes else ""
    st.markdown(f"""
    <div class="results-meta">
        Showing top {len(ranked)} results in {elapsed:.2f}s for <b>"{html.escape(query_input)}"</b>{html.escape(note_html)}{html.escape(phrase_note)}
    </div>
    """, unsafe_allow_html=True)

    # Render result cards
    for rank, doc_idx in enumerate(ranked, 1):
        title = html.escape(titles[doc_idx])
        link = links[doc_idx]
        topic = html.escape(sub_topics[doc_idx])
        snippet = su.highlight(su.make_snippet(documents[doc_idx], tokens + [w for p in phrases for w in p.split()]), tokens)
        full_text = documents[doc_idx].strip()

        st.markdown(f"""
        <div class="result-card">
            <div class="result-link-preview">{link}</div>
            <div class="result-title">
                <a href="{link}" target="_blank">{rank}. {title}</a>
            </div>
            <div class="result-snippet">{snippet}</div>
            <div>
                <span class="badge-topic">{topic}</span>
                <span class="badge-score">Score: {main[doc_idx]:.4f}</span>
                <span class="badge-score">RRF: {rrf_scores[doc_idx]:.4f}</span>
                <span class="badge-score">BM25: {bm25_scores[doc_idx]:.2f}</span>
                <span class="badge-score">Dense: {dense_scores[doc_idx]:.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Read Full Document Text"):
            escaped_text = html.escape(full_text)
            st.markdown(f'<div class="paper-full-text">{escaped_text}</div>', unsafe_allow_html=True)