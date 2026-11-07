"""Run with: python test_search_utils.py   (no model download needed)"""
import pandas as pd
from spellchecker import SpellChecker
from rank_bm25 import BM25Okapi
import search_utils as su

df = pd.read_csv("our_arxiv_corpus.csv")
docs = df.text.tolist()
vocab = su.build_vocab(docs)
spell = SpellChecker()
spell.word_frequency.load_words([w for w, c in vocab.items() if c >= su.MIN_COUNT])
bm25 = BM25Okapi([su.tokenize(d) for d in docs])
docs_norm = [su.normalize(d) for d in docs]

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); assert cond

# 1 case-insensitivity
a = su.clean_query("Machine TRANSLATION", spell, vocab)[0]
b = su.clean_query("machine translation", spell, vocab)[0]
check("case insensitive", a == b)
# 2 abbreviation expansion (and abbreviation kept)
t = su.clean_query("NER", spell, vocab)[0]
check("NER expands", "named" in t and "entity" in t and "recognition" in t and "ner" in t)
check("NLP expands", "natural" in su.clean_query("NLP", spell, vocab)[0])
# 3 spelling correction fixes typos...
t, shown, corr, _ = su.clean_query("trasnlation modle", spell, vocab)
check("typo corrected", "translation" in t and "model" in t); print("   ->", shown, corr)
# ...but does not mangle domain terms
for w in ["lstm", "bleu", "gpt", "chatgpt", "rouge", "bert", "seq2seq", "transformer"]:
    t = su.clean_query(w, spell, vocab)[0]
    ok = w in t or w in su.ABBREVIATIONS
    check(f"domain term kept: {w}", ok)
# 4 phrase matching
ph = su.extract_phrases('"machine translation" evaluation')
check("phrase extracted", ph == ["machine translation"])
hits = [i for i in range(len(docs)) if su.contains_phrase(docs_norm[i], ph[0])]
check("phrase hits found", len(hits) > 50); print("   phrase hits:", len(hits))
# 5 BM25 + RRF + weighted run end-to-end with a stand-in dense score
q = su.clean_query("named entity recognition", spell, vocab)[0]
bs = list(bm25.get_scores(q))
fake_dense = [s * 0.5 + (i % 7) * 0.01 for i, s in enumerate(bs)]
r = su.rrf(bs, fake_dense); w = su.weighted(bs, fake_dense)
top = sorted(range(len(r)), key=lambda i: r[i], reverse=True)[:10]
check("RRF top-10", len(top) == 10)
print("   RRF top-3 topics:", [df.sub_topic[i] for i in top[:3]])
check("BM25 finds NER papers", df.sub_topic[sorted(range(len(bs)), key=lambda i: bs[i], reverse=True)[0]] == "Named Entity Recognition")
# 6 snippet + highlight
s = su.make_snippet(docs[top[0]], q)
h = su.highlight(s, q)
check("snippet ~200 chars", 150 <= len(s) <= 230); print("   len", len(s))
check("highlight marks terms", "<mark>" in h)
check("html escaped", "<script>" not in su.highlight("a <script> named", ["named"]))
print("\nALL TESTS PASSED")
