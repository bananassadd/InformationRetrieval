# Corpus method (reproducibility notes)

**Category:** cs.CL (Computation and Language). All documents come from arXiv.

**Collection script:** `download_corpus.py` (arXiv Python client + PyMuPDF text extraction).
One query per sub-topic, `sort_by = Relevance`, `max_results = 100`, no date filter:

| Sub-topic | Query |
|---|---|
| Machine Translation | `cat:cs.CL AND "machine translation"` |
| Summarization | `cat:cs.CL AND "summarization"` |
| Sentiment Analysis | `cat:cs.CL AND "sentiment analysis"` |
| Named Entity Recognition | `cat:cs.CL AND "named entity recognition"` |
| Question Answering | `cat:cs.CL AND "question answering"` |

**Date range:** no filter was applied. The resulting papers span 1999-2026 (from the arXiv IDs). Downloaded in early November 2026.
Relevance ranking on the live arXiv index can change, so the committed `our_arxiv_corpus.csv` is the authoritative copy of the data.

**Length tiers** (PDF page count, references included): short <= 6 pages, medium 7-9, long > 9.
The assignment leaves exactly 6 pages undefined; we put 6-page papers in *short*.

**Cleaning** (`clean_corpus.py`): 500 raw rows -> removed 6 failed PDF extractions (< 3000 characters) and 1 duplicate ID
(1912.02998, matched by two topic queries; first label kept) -> **493 papers**. Added `arxiv_id` (no version), `url` and `year`.

**Columns:** `arxiv_id, url, title, sub_topic, length_category, page_count, year, text`.
