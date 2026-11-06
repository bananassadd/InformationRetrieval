import streamlit as st
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker

# Load the corpus you downloaded
@st.cache_data
def load_data():
    return pd.read_csv("my_arxiv_corpus.csv").dropna(subset=['text'])

df = load_data()
documents = df['text'].tolist()
titles = df['title'].tolist()
links = df['arxiv_id'].tolist()

# Load the search models (cached so it only loads once)
@st.cache_resource
def load_models():
    # 1. BM25 (Sparse) setup
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 2. Dense Embedding setup
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=True)
    return bm25, model, doc_embeddings

bm25, dense_model, doc_embeddings = load_models()
spell = SpellChecker()

# Custom dictionary for abbreviation expansion
abbreviations = {
    "nlp": "natural language processing", 
    "ml": "machine learning",
    "mt": "machine translation",
    "ner": "named entity recognition",
    "qa": "question answering"
}

def clean_query(q):
    q = q.lower() # Case insensitivity
    q = q.replace('"', '') # Handle phrase match quotes
    
    words = q.split()
    processed_words = []
    for w in words:
        if w in abbreviations:
            processed_words.extend(abbreviations[w].split()) # Abbreviation expansion
        else:
            corrected = spell.correction(w) # Spelling correction
            processed_words.append(corrected if corrected else w)
            
    return " ".join(processed_words)

# --- GUI SETUP ---
st.title("ArXiv CS.CL Hybrid Search Engine")
query = st.text_input("Enter your search query:")
mode = st.selectbox("Select Retrieval Model", ["Hybrid Fusion (BM25 + Dense)", "BM25 (Sparse) Only", "Embeddings (Dense) Only"])

if query:
    processed_q = clean_query(query)
    st.caption(f"Processed query: {processed_q}")
    
    # Get BM25 Scores
    bm25_scores = bm25.get_scores(processed_q.split())
    
    # Get Dense Scores
    query_emb = dense_model.encode(processed_q, convert_to_tensor=True)
    dense_scores = util.dot_score(query_emb, doc_embeddings)[0].cpu().tolist()
    
    # Combine scores using Reciprocal Rank Fusion (RRF)
    bm25_ranks = {idx: rank for rank, idx in enumerate(sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True))}
    dense_ranks = {idx: rank for rank, idx in enumerate(sorted(range(len(dense_scores)), key=lambda i: dense_scores[i], reverse=True))}
    
    final_results = []
    for i in range(len(documents)):
        # RRF Math Formula
        rrf_score = (1.0 / (60 + bm25_ranks[i])) + (1.0 / (60 + dense_ranks[i]))
        final_results.append((i, rrf_score, bm25_scores[i], dense_scores[i]))
        
    # Sort based on the user's GUI selection
    if "Fusion" in mode:
        top_10 = sorted(final_results, key=lambda x: x[1], reverse=True)[:10]
    elif "BM25" in mode:
        top_10 = sorted(final_results, key=lambda x: x[2], reverse=True)[:10]
    else:
        top_10 = sorted(final_results, key=lambda x: x[3], reverse=True)[:10]
        
    # Display the Top 10 Results
    for rank, (doc_idx, rrf, b_score, d_score) in enumerate(top_10, 1):
        st.subheader(f"{rank}. {titles[doc_idx]}")
        st.write(f"**Link:** {links[doc_idx]} | **Score:** {rrf:.4f}")
        # Show ~200 character snippet
        snippet = documents[doc_idx][:200].replace('\n', ' ') + "..."
        st.write(snippet)
        st.divider()