import arxiv 
import requests
import pymupdf as fitz
import sys
import time
import csv
import pandas as pd

# The 5 topics we decided on
topics = {
    "Machine Translation": 'cat:cs.CL AND "machine translation"',
    "Summarization": 'cat:cs.CL AND "summarization"',
    "Sentiment Analysis": 'cat:cs.CL AND "sentiment analysis"',
    "Named Entity Recognition": 'cat:cs.CL AND "named entity recognition"',
    "Question Answering": 'cat:cs.CL AND "question answering"'
}

corpus_data = []
target_per_topic = 100 #so 500 total 

print("Starting arXiv paper download...")

for topic_name, search_query in topics.items():
    print(f"\nSearching for: {topic_name}")
    
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3) 
    search = arxiv.Search(
        query = search_query,
        max_results = 100,
        sort_by = arxiv.SortCriterion.Relevance 
    )

    for paper in client.results(search):
        try:
            
            headers = {'User-Agent': 'Python-urllib/3.0'}
            response = requests.get(paper.pdf_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            pdf_document = fitz.open(stream=response.content, filetype="pdf")
            page_count = len(pdf_document)
            
            if page_count <= 6:
                length_category = "short"
            elif 7 <= page_count <= 9:
                length_category = "medium"
            else:
                length_category = "long"

            full_text = ""
            for page in pdf_document:
                full_text += page.get_text() + "\n"
                
            pdf_document.close()
    
            document_info = {
                "title": paper.title,
                "arxiv_id": paper.entry_id,
                "sub_topic": paper.primary_category,
                "content": full_text,
                "length_category": length_category,
                "page_count": page_count,
                "text": full_text
            }

            corpus_data.append(document_info)

            safe_text = full_text[:50].replace('\n', ' ')
            print(f"{paper.title[:50]}... | {paper.entry_id} | ({length_category}, {page_count} pages) \n")

            time.sleep(2)
        
        except KeyboardInterrupt:
            print("\nStopping the entire script...")
            sys.exit()

        except Exception as e:
            print(f"Error processing {paper.title}: {e}")

df = pd.DataFrame(corpus_data)
df.to_csv("our_arxiv_corpus.csv", index=False)
print("Done")