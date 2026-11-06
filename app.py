import streamlit as st
import pandas as pd
import time
import html
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker

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
        font-weight: 700 !important;
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

    /* Popular Questions Section Heading */
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

    /* Train Station Ticker Row Container */
    div[data-testid="stHorizontalBlock"]:has(.st-key-train_station) {
        align-items: center !important;
        margin-bottom: 0.9rem !important;
    }

    /* Train Station Ticker Container */
    div.st-key-train_station {
        overflow: hidden !important;
        border-radius: 9999px !important;
        padding: 0.2rem 0 !important;
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

    /* Hidden Navigation Arrows (Only Visible on Hover) */
    div.st-key-btn_prev_topic button,
    div.st-key-btn_next_topic button {
        opacity: 0 !important;
        border-radius: 9999px !important;
        width: 32px !important;
        min-width: 32px !important;
        height: 32px !important;
        min-height: 32px !important;
        padding: 0 !important;
        background-color: #FFFFFF !important;
        color: #BD5575 !important;
        border: 1.5px solid #DD91A8 !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        box-shadow: 0 2px 8px rgba(189, 85, 117, 0.22) !important;
        transition: opacity 0.25s ease, background-color 0.15s ease !important;
        cursor: pointer !important;
    }

    /* Reveal arrows when hovering anywhere on the ticker row */
    div[data-testid="stHorizontalBlock"]:has(.st-key-train_station):hover div.st-key-btn_prev_topic button,
    div[data-testid="stHorizontalBlock"]:has(.st-key-train_station):hover div.st-key-btn_next_topic button,
    div.st-key-btn_prev_topic button:hover,
    div.st-key-btn_next_topic button:hover {
        opacity: 1 !important;
    }

    div.st-key-btn_prev_topic button:hover,
    div.st-key-btn_next_topic button:hover {
        background-color: #DD91A8 !important;
        color: #FFFFFF !important;
        border-color: #BD5575 !important;
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
    df_raw = pd.read_csv("our_arxiv_corpus.csv").dropna(subset=['text'])
    docs = df_raw['text'].tolist()
    tts = df_raw['title'].tolist()
    lks = df_raw['arxiv_id'].tolist()
    subs = df_raw['sub_topic'].tolist() if 'sub_topic' in df_raw.columns else ['CS.CL'] * len(df_raw)
    return df_raw, docs, tts, lks, subs

df, documents, titles, links, sub_topics = load_data()

# Model Loading
@st.cache_resource
def load_models():
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=False)
    return bm25, model, doc_embeddings

bm25, dense_model, doc_embeddings = load_models()

@st.cache_resource
def load_spellchecker():
    return SpellChecker()

spell = load_spellchecker()

abbreviations = {
    "nlp": "natural language processing", 
    "ml": "machine learning",
    "mt": "machine translation",
    "ner": "named entity recognition",
    "qa": "question answering"
}

def clean_query(q):
    q = q.lower() 
    q = q.replace('"', '') 
    
    words = q.split()
    processed_words = []
    for w in words:
        if w in abbreviations:
            processed_words.extend(abbreviations[w].split()) 
        else:
            corrected = spell.correction(w) 
            processed_words.append(corrected if corrected else w)
            
    return " ".join(processed_words)

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

if "topic_start" not in st.session_state:
    st.session_state.topic_start = 0

if not st.session_state.search_query:
    st.markdown('<div class="popular-questions-label">Popular Questions</div>', unsafe_allow_html=True)
    col_prev, col_train, col_next = st.columns([0.06, 0.88, 0.06], vertical_alignment="center")
    
    with col_prev:
        if st.button("‹", key="btn_prev_topic"):
            st.session_state.topic_start = (st.session_state.topic_start - 1) % len(SUGGESTED_TOPICS)
            st.rerun()
            
    with col_next:
        if st.button("›", key="btn_next_topic"):
            st.session_state.topic_start = (st.session_state.topic_start + 1) % len(SUGGESTED_TOPICS)
            st.rerun()

    # 5 suggested topics with full names
    current_5 = [SUGGESTED_TOPICS[(st.session_state.topic_start + i) % len(SUGGESTED_TOPICS)] for i in range(len(SUGGESTED_TOPICS))]
    # Duplicate for immediate seamless infinite marquee loop (0% to -50%)
    ticker_topics = current_5 + current_5

    with col_train:
        with st.container(key="train_station"):
            t_cols = st.columns(len(ticker_topics))
            for idx, (t_label, t_q) in enumerate(ticker_topics):
                with t_cols[idx]:
                    if st.button(t_label, key=f"topic_btn_{idx}_{st.session_state.topic_start}"):
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
mode = st.pills(
    label="Model",
    options=["Hybrid Fusion", "BM25 (Keywords)", "Dense (Semantic)"],
    default="Hybrid Fusion",
    label_visibility="collapsed"
)

# Fixed top_k = 10
top_k = 10

# Search Execution
if query_input:
    t_start = time.time()
    processed_q = clean_query(query_input)
    
    # BM25 Scores
    bm25_scores = bm25.get_scores(processed_q.split())
    
    # Dense Scores
    query_emb = dense_model.encode(processed_q, convert_to_tensor=True)
    dense_scores = util.dot_score(query_emb, doc_embeddings)[0].cpu().tolist()

    # Reciprocal Rank Fusion (RRF)
    bm25_ranks = {idx: rank for rank, idx in enumerate(sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True))}
    dense_ranks = {idx: rank for rank, idx in enumerate(sorted(range(len(dense_scores)), key=lambda i: dense_scores[i], reverse=True))}
    
    final_results = []
    for i in range(len(documents)):
        rrf_score = (1.0 / (60 + bm25_ranks[i])) + (1.0 / (60 + dense_ranks[i]))
        final_results.append((i, rrf_score, bm25_scores[i], dense_scores[i]))
        
    if "Fusion" in mode:
        results = sorted(final_results, key=lambda x: x[1], reverse=True)[:top_k]
    elif "BM25" in mode:
        results = sorted(final_results, key=lambda x: x[2], reverse=True)[:top_k]
    else:
        results = sorted(final_results, key=lambda x: x[3], reverse=True)[:top_k]

    elapsed = time.time() - t_start

    # Result Summary
    expansion_note = f" (expanded to <i>{processed_q}</i>)" if processed_q != query_input.lower() else ""
    st.markdown(f"""
    <div class="results-meta">
        Showing top {len(results)} results in {elapsed:.2f}s for <b>"{query_input}"</b>{expansion_note}
    </div>
    """, unsafe_allow_html=True)

    # Render Modern Cards
    for rank, (doc_idx, rrf, b_score, d_score) in enumerate(results, 1):
        title = titles[doc_idx]
        link = links[doc_idx]
        topic = sub_topics[doc_idx]
        snippet = documents[doc_idx][:220].replace('\n', ' ').strip() + "..."
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
                <span class="badge-score">Score: {rrf:.4f}</span>
                <span class="badge-score">BM25: {b_score:.2f}</span>
                <span class="badge-score">Dense: {d_score:.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Read Full Document Text"):
            escaped_text = html.escape(full_text)
            st.markdown(f'<div class="paper-full-text">{escaped_text}</div>', unsafe_allow_html=True)
