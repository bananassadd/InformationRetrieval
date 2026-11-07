"""Query handling, fusion and display helpers (pure Python, no Streamlit/torch needed)."""
import re
import html
from collections import Counter

MIN_COUNT = 5   # a word seen fewer times than this in the corpus may be a typo
TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# abbreviation -> expansion. The expansion is ADDED to the query; the abbreviation is kept too,
# because many papers only ever write "NER" or "NMT".
ABBREVIATIONS = {
    "nlp": "natural language processing", "ml": "machine learning",
    "mt": "machine translation", "nmt": "neural machine translation",
    "ner": "named entity recognition", "qa": "question answering",
    "llm": "large language model", "llms": "large language models",
    "rag": "retrieval augmented generation", "nlg": "natural language generation",
    "nlu": "natural language understanding", "asr": "automatic speech recognition",
    "pos": "part of speech", "ir": "information retrieval",
    "sa": "sentiment analysis", "ie": "information extraction",
    "kb": "knowledge base", "kbqa": "knowledge base question answering",
}


def tokenize(text):
    return TOKEN.findall(text.lower())


def normalize(text):
    """Lower-cased tokens joined by single spaces - used for exact phrase matching."""
    return " ".join(tokenize(text))


def build_vocab(documents):
    return Counter(tok for d in documents for tok in tokenize(d))


def extract_phrases(raw_query):
    """Text inside double quotes is a phrase that must appear exactly."""
    return [normalize(p) for p in re.findall(r'"([^"]+)"', raw_query) if normalize(p)]


def clean_query(raw_query, spell, vocab):
    """Returns (tokens_for_search, display_string, corrections, expansions).

    - case-insensitive (everything lower-cased)
    - abbreviations are expanded (and the abbreviation itself is kept)
    - a token is only spell-corrected if the corpus has never seen it
    """
    tokens, shown, corrections, expansions = [], [], {}, {}
    for w in tokenize(raw_query):
        if w in ABBREVIATIONS:
            exp = ABBREVIATIONS[w]
            expansions[w] = exp
            tokens.append(w)
            tokens.extend(tokenize(exp))
            shown.append(w)
            continue
        if vocab.get(w, 0) < MIN_COUNT and not w.isdigit():
            fixed = spell.correction(w)
            if fixed and fixed != w:
                corrections[w] = fixed
                w = fixed
        tokens.append(w)
        shown.append(w)
    return tokens, " ".join(shown), corrections, expansions


def contains_phrase(doc_norm, phrase):
    return f" {phrase} " in f" {doc_norm} "


def rank_positions(scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def rrf(bm25_scores, dense_scores, k=60):
    """Reciprocal Rank Fusion. Docs with BM25 score 0 match no query term, so they get no BM25 vote."""
    b_ranks, d_ranks = rank_positions(bm25_scores), rank_positions(dense_scores)
    out = []
    for i in range(len(bm25_scores)):
        s = 1.0 / (k + d_ranks[i] + 1)
        if bm25_scores[i] > 0:
            s += 1.0 / (k + b_ranks[i] + 1)
        out.append(s)
    return out


def minmax(xs):
    lo, hi = min(xs), max(xs)
    return [0.0] * len(xs) if hi == lo else [(x - lo) / (hi - lo) for x in xs]


def weighted(bm25_scores, dense_scores, alpha=0.5):
    """Weighted score fusion: alpha * normalised BM25 + (1 - alpha) * normalised dense."""
    b, d = minmax(bm25_scores), minmax(dense_scores)
    return [alpha * x + (1 - alpha) * y for x, y in zip(b, d)]


def make_snippet(text, terms, width=200):
    """~200 chars around the first query-term hit, starting after the 'Abstract' heading when present."""
    low = text.lower()
    a = low.find("abstract", 0, 4000)
    start = a + len("abstract") if a != -1 else 0
    hits = [p for t in set(terms) if (p := low.find(t, start)) != -1]
    pos = min(hits) if hits else start
    s = max(start, pos - 60)
    snippet = " ".join(text[s:s + width].split())
    return ("... " if s > start else "") + snippet + " ..."


def highlight(snippet, terms):
    """HTML-escape the snippet, then wrap query terms in <mark>."""
    esc = html.escape(snippet)
    terms = sorted({t for t in terms if len(t) > 1}, key=len, reverse=True)
    if not terms:
        return esc
    pat = "|".join(re.escape(html.escape(t)) for t in terms)
    return re.sub(rf"(?<![A-Za-z0-9])({pat})(?![A-Za-z0-9])", r"<mark>\1</mark>", esc, flags=re.I)