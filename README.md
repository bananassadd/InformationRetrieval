# InformationRetrieval - arXiv cs.CL Hybrid Search Engine (Phase 1)

A small literature-search engine over **493 arXiv cs.CL papers** (5 sub-topics: machine translation, summarization,
sentiment analysis, named entity recognition, question answering). It combines **BM25** (sparse) and a
**sentence-embedding retriever** (dense, `all-MiniLM-L6-v2`), fuses them, and shows the results in a Streamlit GUI.

## Run it

```bash
git clone https://github.com/bananassadd/InformationRetrieval.git
cd InformationRetrieval
pip install -r requirements.txt
python -m streamlit run app.py
```

The first start downloads the embedding model (about 90 MB, internet needed) and embeds the corpus (about a minute on CPU);
embeddings are then cached in `embeddings_cache_493.npy`.

## What the engine does

| Feature | How |
|---|---|
| BM25 | `rank_bm25` over a regex tokenizer (punctuation stripped, lower-cased) |
| Dense retrieval | MiniLM embeddings of title + start of the abstract, cosine similarity |
| Fusion | **Reciprocal Rank Fusion** (k = 60) or **weighted score fusion** (0.5 / 0.5 on min-max normalised scores) |
| Model selector | Hybrid (RRF), Hybrid (Weighted), BM25, Dense |
| Case insensitivity | everything is lower-cased |
| Abbreviation expansion | e.g. `NER`, `NLP`, `NMT`, `QA`, `LLM`, `RAG` are expanded and the abbreviation is kept too |
| Spelling correction | only words the corpus has (almost) never seen are corrected, so `LSTM`, `BLEU`, `GPT` are left alone |
| Phrase matching | `"quoted text"` must appear exactly (filter); an unquoted multi-word query gets a ranking bonus for exact-phrase hits |
| GUI | search box, model selector, top-10 results with title, arXiv link, ~200-character snippet with highlighted terms, and scores |

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit GUI and search pipeline |
| `search_utils.py` | query handling, fusion, snippets, highlighting (unit-tested) |
| `test_search_utils.py` | tests for the query handling (`python test_search_utils.py`) |
| `download_corpus.py` | collects papers from the arXiv API and extracts PDF text |
| `clean_corpus.py` | cleans the raw download (see `docs/corpus_method.md`) |
| `cleaned_arxiv_corpus.csv` | the final corpus (493 papers) |
| `make_report.py`, `corpus_report.pdf` | corpus statistics report |
| `docs/corpus_method.md` | exact queries, tier rule, cleaning steps |
| `ai-logs/phase1/` | AI usage disclosure |
