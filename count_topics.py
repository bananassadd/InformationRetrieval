import arxiv
client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
candidates = {
    "machine_translation": 'abs:"machine translation"',
    "question_answering":  'abs:"question answering"',
    "summarization":       'abs:"summarization"',
    "dialogue_systems":    'abs:"dialogue system" OR abs:"conversational agent"',
    "sentiment_analysis":  'abs:"sentiment analysis"',
    "named_entity_recognition": 'abs:"named entity recognition"',
}
for name, q in candidates.items():
    full = f'cat:cs.CL AND ({q}) AND submittedDate:[202401010000 TO 202512312359]'
    n = sum(1 for _ in client.results(arxiv.Search(query=full, max_results=300)))
    print(name, n, flush=True)
