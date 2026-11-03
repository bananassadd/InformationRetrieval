import arxiv
client = arxiv.Client()
s = arxiv.Search(query='cat:cs.CL AND abs:"machine translation"', max_results=3)
for r in client.results(s):
    print(r.get_short_id(), r.title)
