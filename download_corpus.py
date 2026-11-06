import arxiv 
import fitz  
import pandas as pd 
import os 
import time 
import requests 

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
        max_results = target_per_topic,
        sort_by = arxiv.SortCriterion.Relevance 
    )

    for paper in client.results(search):
        try:
            pdf_url = paper.pdf_url
            response = requests.get(pdf_url)
            
            temp_pdf_name = "temp_paper.pdf"
            with open(temp_pdf_name, 'wb') as f: 
                f.write(response.content)

            doc = fitz.open(temp_pdf_name)
            page_count = len(doc)
            
            if page_count <= 6:
                length_category = "short"
            elif 7 <= page_count <= 9:
                length_category = "medium"
            elif page_count > 9:
                length_category = "long"
        

            full_text = ""
            for page in doc:
                full_text += page.get_text()
            
            doc.close()
            os.remove(temp_pdf_name) 

            corpus_data.append({
                "title": paper.title,
                "arxiv_id": paper.entry_id,
                "sub_topic": topic_name,
                "length_category": length_category,
                "page_count": page_count,
                "text": full_text
            })
            
            print(f"Saved: {paper.title[:50]}... ({length_category}, {page_count} pages)")
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Skipping a paper due to error: {e}")

df = pd.DataFrame(corpus_data)
df.to_csv("our_arxiv_corpus.csv", index=False)
print("\nAll done! Saved everything to our_arxiv_corpus.csv")