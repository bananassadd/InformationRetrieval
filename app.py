import streamlit as st
import pandas as pd
import os
import re
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

# Soft Bloom Rose & Blush Theme (sampled from Soft Bloom editorial palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Global Background Gradient & Typography */
    .stApp {
        background: linear-gradient(145deg, #FFF0F5 0%, #F8D7DF 30%, #EAC0CB 70%, #E0A6B8 100%) !important;
        background-attachment: fixed !important;
        color: #261B20 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .block-container {
        max-width: 760px;
        padding-top: 3.5rem;
        padding-bottom: 5rem;
    }

    /* Hide default Streamlit headers */
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* Minimalist Editorial Header */
    .search-hero {
        text-align: center;
        margin-bottom: 2.85rem;
    }
    .search-title {
        font-family: 'Verdana', Geneva, Tahoma, sans-serif !important;
        font-size: 3.25rem !important;
        font-weight: 400 !important;
        letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
        color: #111111 !important;
        margin-bottom: 0.45rem !important;
        line-height: 1.15 !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }
    .search-subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #7A5361 !important;
        font-size: 1.05rem !important;
        font-weight: 400 !important;
        letter-spacing: 0.02em !important;
    }

    /* Popular Topics Section Heading */
    .popular-topics-label,
    .popular-questions-label {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #BD5575 !important;
        letter-spacing: 0.02em !important;
        text-align: center !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.65rem !important;
    }

    /* Search Bar Capsule styled after the browser pill (www.soft bloom.com) */
    div[data-testid="stTextInputRootElement"],
    div[data-testid="stTextInput"] > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border-radius: 9999px !important;
    }

    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 2px solid #DD91A8 !important;
        border-radius: 9999px !important;
        box-shadow: 0 4px 20px rgba(168, 79, 107, 0.22) !important;
        padding-left: 0.8rem !important;
        padding-right: 1.3rem !important;
        padding-top: 0.45rem !important;
        padding-bottom: 0.45rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #BD5575 !important;
        box-shadow: 0 6px 26px rgba(189, 85, 117, 0.35) !important;
    }

    /* Magnifying glass lens on the left */
    div[data-testid="stTextInput"] [data-testid="stIconMaterial"],
    div[data-testid="stTextInput"] svg,
    div[data-testid="stTextInputRootElement"] [data-testid="stIconMaterial"],
    div[data-baseweb="input"] [data-testid="stIconMaterial"],
    div[data-baseweb="input"] svg {
        color: #111111 !important;
        fill: #111111 !important;
        font-size: 1.6rem !important;
        margin-left: 0.35rem !important;
        margin-right: 0.45rem !important;
    }

    /* Black Input Text with comfortable breathing room from the lens */
    div[data-baseweb="input"] input,
    div[data-testid="stTextInput"] input,
    input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.15rem !important;
        font-weight: 500 !important;
        background: transparent !important;
        padding-left: 0.95rem !important;
    }
    div[data-baseweb="input"] input::placeholder,
    div[data-testid="stTextInput"] input::placeholder,
    input::placeholder {
        color: #8E6D77 !important;
        -webkit-text-fill-color: #8E6D77 !important;
        font-weight: 400 !important;
    }

    /* Train Station Ticker Container */
    div.st-key-train_station {
        overflow: hidden !important;
        border-radius: 9999px !important;
        padding: 0.2rem 0 !important;
        margin-bottom: 0.95rem !important;
        width: 100% !important;
        mask-image: linear-gradient(to right, transparent, black 4%, black 96%, transparent) !important;
        -webkit-mask-image: linear-gradient(to right, transparent, black 4%, black 96%, transparent) !important;
    }

    /* Seamless continuous automatic moving track of pills */
    div.st-key-train_station div[data-testid="stHorizontalBlock"],
    div.st-key-train_station [data-testid="stHorizontalBlock"] {
        display: inline-flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        width: max-content !important;
        min-width: max-content !important;
        max-width: none !important;
        gap: 0 !important;
        animation: trainStationSeamless 22s linear infinite !important;
        will-change: transform !important;
    }

    /* Pause animation only when directly hovering on a pill button to click it */
    div.st-key-train_station div[data-testid="stHorizontalBlock"]:has(button:hover) {
        animation-play-state: paused !important;
    }

    /* Seamless infinite automatic loop: smoothly slides through all topics continuously */
    @keyframes trainStationSeamless {
        0% {
            transform: translate3d(0, 0, 0);
        }
        100% {
            transform: translate3d(-50%, 0, 0);
        }
    }

    /* Prevent stacking: enforce separate max-content columns with margins */
    div.st-key-train_station [data-testid="column"],
    div.st-key-train_station [data-testid="stColumn"],
    div.st-key-train_station div.stColumn {
        width: max-content !important;
        min-width: max-content !important;
        max-width: none !important;
        flex: 0 0 auto !important;
        flex-grow: 0 !important;
        flex-shrink: 0 !important;
        flex-basis: auto !important;
        display: inline-flex !important;
        align-items: center !important;
        margin: 0 10px !important;
        padding: 0 !important;
    }

    div.st-key-train_station [data-testid="stVerticalBlock"],
    div.st-key-train_station [data-testid="stVerticalBlockBorderWrapper"] {
        width: max-content !important;
        min-width: max-content !important;
        max-width: none !important;
        flex: 0 0 auto !important;
        display: inline-flex !important;
        gap: 0 !important;
        padding: 0 !important;
    }

    div.st-key-train_station div.stButton,
    div.st-key-train_station [data-testid="stButton"] {
        width: max-content !important;
        min-width: max-content !important;
        display: inline-flex !important;
    }

    /* Topic Pill Buttons with full names, completely separated */
    div.st-key-train_station div.stButton > button,
    div.st-key-train_station button[data-testid="baseButton-secondary"],
    div.st-key-train_station button[data-testid="stBaseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #BD5575 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        border: 1.5px solid #DD91A8 !important;
        border-radius: 9999px !important;
        box-shadow: 0 2px 8px rgba(189, 85, 117, 0.18) !important;
        padding: 0.35rem 1.15rem !important;
        min-height: 2.15rem !important;
        height: 2.2rem !important;
        width: max-content !important;
        min-width: max-content !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.2 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.15s ease !important;
    }
    div.st-key-train_station div.stButton > button * {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 0.85rem !important;
        max-width: none !important;
        display: inline !important;
    }
    div.st-key-train_station div.stButton > button:hover {
        background-color: #DD91A8 !important;
        border-color: #BD5575 !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
    }

    /* Results Header */
    .results-meta {
        font-size: 0.9rem;
        color: #7A5361;
        font-weight: 600;
        margin-top: 1.6rem;
        margin-bottom: 1rem;
    }

    /* Modern Result Card */
    .result-card {
        background-color: #FFFFFF;
        border: 1.5px solid #E2A4B6;
        border-radius: 18px;
        box-shadow: 0 6px 22px rgba(168, 79, 107, 0.16);
        padding: 1.35rem 1.6rem;
        margin-bottom: 1.2rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 26px rgba(189, 85, 117, 0.25);
        border-color: #BD5575;
    }
    .result-link-preview {
        font-size: 0.8rem;
        color: #8A5F6E;
        margin-bottom: 0.35rem;
        word-break: break-all;
    }
    .result-title a {
        color: #BD5575 !important;
        font-size: 1.22rem !important;
        font-weight: 600 !important;
        line-height: 1.35;
        text-decoration: none !important;
    }
    .result-title a:hover {
        color: #9C3857 !important;
        text-decoration: underline !important;
    }
    .result-snippet {
        color: #2D2025;
        font-size: 0.93rem;
        line-height: 1.55;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }

    /* Highlighted query terms */
    mark {
        background-color: #FFD9E3;
        color: #8C2D4A;
        font-weight: 600;
        padding: 0 2px;
        border-radius: 3px;
    }

    /* Metadata Badges */
    .badge-topic {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        background-color: #FCEEF2;
        border: 1px solid #DD91A8;
        color: #BD5575;
        font-weight: 600;
        font-size: 0.75rem;
        border-radius: 9999px;
        margin-right: 0.4rem;
    }
    .badge-score {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        background-color: #FFFFFF;
        border: 1px solid #E2A4B6;
        color: #7A5361;
        font-size: 0.75rem;
        font-weight: 500;
        border-radius: 9999px;
        margin-right: 0.35rem;
    }

    /* Expander Container */
    div[data-testid="stExpander"] {
        background-color: #FDF6F8 !important;
        border: 1.5px solid #E2A4B6 !important;
        border-radius: 12px !important;
        margin-top: 0.4rem !important;
    }
    div[data-testid="stExpander"] summary {
        color: #BD5575 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stExpander"] h1,
    div[data-testid="stExpander"] h2,
    div[data-testid="stExpander"] h3,
    div[data-testid="stExpander"] h4 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        line-height: 1.4 !important;
        margin: 0.4rem 0 !important;
        color: #2D2025 !important;
    }

    /* Scrollable Clean Document Viewer inside Expander */
    .paper-full-text {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 400 !important;
        line-height: 1.65 !important;
        color: #2D2025 !important;
        white-space: pre-wrap !important;
        word-break: break-word !important;
        max-height: 460px !important;
        overflow-y: auto !important;
        padding: 1rem 1.15rem !important;
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        border: 1px solid #E2A4B6 !important;
    }

    /* Model Selection Pills (Styled after Soft Bloom 'Learn more>' buttons) */
    div[data-testid="stButtonGroup"],
    div.stButtonGroup,
    div[data-testid="stPills"] {
        margin-top: 0.35rem !important;
        margin-bottom: 0.5rem !important;
    }

    div[data-testid="stButtonGroup"] button,
    div.stButtonGroup button,
    div[data-testid="stButtonGroup"] [data-variant="pills"],
    div[data-testid="stPills"] button {
        border-radius: 9999px !important;
        border: 1.5px solid #DD91A8 !important;
        background-color: #FFFFFF !important;
        color: #7A2F45 !important;
        -webkit-text-fill-color: #7A2F45 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(189, 85, 117, 0.12) !important;
        transition: all 0.15s ease !important;
    }

    div[data-testid="stButtonGroup"] button *,
    div.stButtonGroup button * {
        color: #7A2F45 !important;
        -webkit-text-fill-color: #7A2F45 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stButtonGroup"] button:hover,
    div.stButtonGroup button:hover {
        border-color: #BD5575 !important;
        background-color: #FCEEF2 !important;
    }

    /* Selected / Active State: Deep rich dark berry with crisp pure white text for maximum clarity */
    div[data-testid="stButtonGroup"] button[data-selected],
    div[data-testid="stButtonGroup"] button[data-selected="true"],
    div[data-testid="stButtonGroup"] button[aria-selected="true"],
    div[data-testid="stButtonGroup"] button[aria-checked="true"],
    div[data-testid="stButtonGroup"] [data-selected],
    div[data-testid="stButtonGroup"] [data-selected="true"],
    div[data-testid="stButtonGroup"] [aria-selected="true"],
    div[data-testid="stButtonGroup"] [aria-checked="true"],
    div.stButtonGroup button[data-selected],
    div.stButtonGroup button[aria-selected="true"],
    div.stButtonGroup button[aria-checked="true"],
    div.stButtonGroup [data-selected],
    div[data-testid="stPills"] button[aria-selected="true"],
    div[data-testid="stPills"] button[aria-checked="true"] {
        background-color: #8C2D4A !important;
        border-color: #6B1B33 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(140, 45, 74, 0.4) !important;
    }

    div[data-testid="stButtonGroup"] button[data-selected] *,
    div[data-testid="stButtonGroup"] button[aria-selected="true"] *,
    div[data-testid="stButtonGroup"] button[aria-checked="true"] *,
    div[data-testid="stButtonGroup"] [data-selected] *,
    div[data-testid="stButtonGroup"] [aria-selected="true"] *,
    div[data-testid="stButtonGroup"] [aria-checked="true"] *,
    div.stButtonGroup button[data-selected] *,
    div.stButtonGroup button[aria-selected="true"] *,
    div.stButtonGroup button[aria-checked="true"] *,
    div.stButtonGroup [data-selected] *,
    div[data-testid="stPills"] button[aria-selected="true"] *,
    div[data-testid="stPills"] button[aria-checked="true"] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Data Loading
@st.cache_data
def load_data():
    df_raw = pd.read_csv("our_arxiv_corpus.csv").dropna(subset=['text']).reset_index(drop=True)
    return df_raw

df = load_data()
documents = df['text'].tolist()
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
    label_visibility="collapsed",
    icon=":material/search:"
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
    #  - "quoted text" must appear exactly in the document (hard filter)
    #  - an unquoted multi-word query gets a bonus when the whole phrase appears exactly
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
