"""
Clean the raw arXiv corpus produced by download_corpus.py.

Usage:  python clean_corpus.py our_arxiv_corpus_raw.csv our_arxiv_corpus.csv

Steps (each one is counted and printed):
  1. drop papers whose extracted text is shorter than 3000 characters (PDF extraction failed)
  2. drop duplicate arXiv IDs (a paper matched by two topic queries) - first topic wins
  3. add a clean arXiv ID, a clean abs URL (no version suffix) and the submission year
"""
import re, sys, json
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "our_arxiv_corpus.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "our_arxiv_corpus.csv"

df = pd.read_csv(src)
log = {"raw_rows": len(df)}

short = df.text.fillna("").str.len() < 3000
log["dropped_failed_extraction"] = int(short.sum())
df = df[~short].copy()

df["arxiv_id"] = df.arxiv_id.str.extract(r"abs/(.+?)(?:v\d+)?$")[0]
dup = df.arxiv_id.duplicated(keep="first")
log["dropped_duplicate_ids"] = int(dup.sum())
df = df[~dup].copy()

df["url"] = "https://arxiv.org/abs/" + df.arxiv_id
# new-style IDs: 2401.01234 -> 2024 ; old-style IDs: cs/9906034 -> 1999
yy = df.arxiv_id.str.extract(r"^(?:[a-z\-\.]+/)?(\d{2})")[0].astype(int)
df["year"] = yy.where(yy < 91, yy + 1900).where(yy >= 91, yy + 2000)

cols = ["arxiv_id", "url", "title", "sub_topic", "length_category", "page_count", "year", "text"]
df = df[cols].reset_index(drop=True)
df.to_csv(dst, index=False)
log["clean_rows"] = len(df)
print(json.dumps(log, indent=2))
