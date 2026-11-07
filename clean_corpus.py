import pandas as pd
import re

def clean_pdf_text(text):
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'[\n\t]+', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading raw corpus...")
df = pd.read_csv("our_arxiv_corpus.csv")
original_count = len(df)

df.drop_duplicates(subset=['arxiv_id'], keep='first', inplace=True)

print("Cleaning text formatting...")
df['content'] = df['content'].apply(clean_pdf_text)
df['title'] = df['title'].apply(clean_pdf_text)

short = df['content'].str.len() < 3000
df = df[~short].copy()

cleaned_filename = "cleaned_arxiv_corpus.csv"
cols = ["arxiv_id", "title", "sub_topic", "length_category", "page_count", "content"]
df = df[cols].reset_index(drop=True)

df.to_csv(cleaned_filename, index=False, encoding="utf-8")

print(f"Done! Cleaned dataset saved to {cleaned_filename}.")
print(f"Rows before: {original_count} | Rows after: {len(df)}")