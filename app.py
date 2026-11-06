import streamlit as st
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker

# Load the corpus 
@st.cache_data
def load_data():
    return pd.read_csv("our_arxiv_corpus.csv").dropna(subset=['text'])

df = load_data()
documents = df['text'].tolist()
titles = df['title'].tolist()
links = df['arxiv_id'].tolist()

# Load the search models 
@st.cache_resource
def load_models():
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=True)
    return bm25, model, doc_embeddings

bm25, dense_model, doc_embeddings = load_models()
spell = SpellChecker()

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

st.title("ArXiv CS.CL Hybrid Search Engine")
query = st.text_input("Enter your search query:")
mode = st.selectbox("Select Retrieval Model", ["Hybrid Fusion (BM25 + Dense)", "BM25 (Sparse) Only", "Embeddings (Dense) Only"])

if query:
    processed_q = clean_query(query)
    st.caption(f"Processed query: {processed_q}")
    
    bm25_scores = bm25.get_scores(processed_q.split())
    
    query_emb = dense_model.encode(processed_q, convert_to_tensor=True)
    dense_scores = util.dot_score(query_emb, doc_embeddings)[0].cpu().tolist()

    bm25_ranks = {idx: rank for rank, idx in enumerate(sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True))}
    dense_ranks = {idx: rank for rank, idx in enumerate(sorted(range(len(dense_scores)), key=lambda i: dense_scores[i], reverse=True))}
    
    final_results = []
    for i in range(len(documents)):
        # RRF Math Formula
        rrf_score = (1.0 / (60 + bm25_ranks[i])) + (1.0 / (60 + dense_ranks[i]))
        final_results.append((i, rrf_score, bm25_scores[i], dense_scores[i]))
        
    if "Fusion" in mode:
        top_10 = sorted(final_results, key=lambda x: x[1], reverse=True)[:10]
    elif "BM25" in mode:
        top_10 = sorted(final_results, key=lambda x: x[2], reverse=True)[:10]
    else:
        top_10 = sorted(final_results, key=lambda x: x[3], reverse=True)[:10]

    for rank, (doc_idx, rrf, b_score, d_score) in enumerate(top_10, 1):
        st.subheader(f"{rank}. {titles[doc_idx]}")
        st.write(f"**Link:** {links[doc_idx]} | **Score:** {rrf:.4f}")
        # Show ~200 character snippet
        snippet = documents[doc_idx][:200].replace('\n', ' ') + "..."
        st.write(snippet)
        st.divider()