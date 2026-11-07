"""Builds corpus_report.pdf from our_arxiv_corpus.csv.  Usage: python make_report.py  (needs reportlab, matplotlib)"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

df = pd.read_csv("our_arxiv_corpus.csv")
df["words"] = df.text.str.split().str.len()
tiers = ["short", "medium", "long"]
ct = df.groupby(["sub_topic", "length_category"]).size().unstack(fill_value=0)[tiers]
ct["total"] = ct.sum(axis=1)

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ct[tiers].plot(kind="barh", stacked=True, ax=ax, color=["#E8A5B8", "#BD5575", "#6B1B33"])
ax.set_xlabel("papers"); ax.set_ylabel("")
ax.legend(title="length tier", fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
plt.tight_layout(); plt.savefig("chart_tiers.png", dpi=170); plt.close()
fig, ax = plt.subplots(figsize=(6.4, 2.6))
yc = df.year.value_counts().sort_index()
ax.bar(yc.index, yc.values, color="#BD5575"); ax.set_xlabel("submission year (from arXiv ID)"); ax.set_ylabel("papers")
plt.tight_layout(); plt.savefig("chart_years.png", dpi=170); plt.close()

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10)
def tbl(data, widths=None, left_cols=(0,)):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#8C2D4A")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTSIZE", (0,0), (-1,-1), 8.5), ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#CCAAB5")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FCEEF2")]), ("ALIGN", (1,1), (-1,-1), "CENTER")]
        + [("ALIGN", (c,0), (c,-1), "LEFT") for c in left_cols]))
    return t

doc = SimpleDocTemplate("corpus_report.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm)
E = [Paragraph("Phase 1 - Corpus Summary Report", H1),
     Paragraph("Information Retrieval - arXiv cs.CL hybrid search engine. Princess Sumaya University for Technology.", B), Spacer(1, 6),
     Paragraph("1. Overview", H2),
     Paragraph(f"The corpus contains <b>{len(df)} papers</b> from arXiv category <b>cs.CL</b> (Computation and Language), "
               f"labelled with <b>5 sub-topics</b>: machine translation, summarization, sentiment analysis, named entity recognition and question answering. "
               f"Every document has a title, full extracted text, arXiv ID/link, sub-topic label, page count and length tier. All papers come from arXiv only.", B)]
rows = [["Sub-topic", "Short", "Medium", "Long", "Total"]] + [[i] + [int(v) for v in r[tiers + ["total"]]] for i, r in ct.iterrows()]
rows.append(["All", int(ct.short.sum()), int(ct.medium.sum()), int(ct.long.sum()), int(ct.total.sum())])
E += [Spacer(1, 6), Paragraph("Table 1 - papers per sub-topic and length tier", SM), tbl(rows, [6*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm]), Spacer(1, 4),
      Image("chart_tiers.png", width=14*cm, height=14*cm*3.2/6.4),
      Paragraph("2. How the corpus was collected (reproducibility)", H2),
      Paragraph("Script: <font face='Courier'>download_corpus.py</font> (arXiv Python client, then PyMuPDF to extract PDF text). For each sub-topic one query was run, "
                "sorted by <b>relevance</b>, with <b>max_results = 100</b>:", B)]
q = [["Sub-topic", "arXiv API query"],
     ["Machine Translation", 'cat:cs.CL AND "machine translation"'], ["Summarization", 'cat:cs.CL AND "summarization"'],
     ["Sentiment Analysis", 'cat:cs.CL AND "sentiment analysis"'], ["Named Entity Recognition", 'cat:cs.CL AND "named entity recognition"'],
     ["Question Answering", 'cat:cs.CL AND "question answering"']]
E += [Spacer(1, 4), tbl(q, [5.5*cm, 10.5*cm], left_cols=(0, 1)), Spacer(1, 4),
      Paragraph("<b>Date range:</b> no submission-date filter was applied, so the date range is whatever the relevance ranking returned. "
                f"The papers span <b>{int(df.year.min())} to {int(df.year.max())}</b> (derived from the arXiv IDs); the data was downloaded in early November 2026. "
                "Because relevance ranking on the live arXiv index can change over time, re-running the script later may return a slightly different set; "
                "the exact papers are therefore stored in the committed dataset <font face='Courier'>our_arxiv_corpus.csv</font>.", B),
      Paragraph("<b>Length tiers</b> use the PDF page count (references and appendices included): short = 6 pages or fewer, medium = 7-9 pages, long = more than 9 pages. "
                "The assignment leaves exactly 6 pages undefined; we placed 6-page papers in <i>short</i>.", B),
      Paragraph("3. Cleaning applied to the raw download", H2),
      Paragraph("The raw download had 500 rows. We removed <b>6</b> papers whose PDF text extraction failed (fewer than 3,000 characters; five had a page count of 1) "
                "and <b>1</b> duplicate (arXiv 1912.02998 matched both the machine-translation and question-answering queries; the first label was kept). "
                "We also stored a clean arXiv ID and URL without the version suffix and added the submission year. Script: <font face='Courier'>clean_corpus.py</font>. "
                f"Final size: <b>{len(df)} papers</b>.", B),
      PageBreak(), Paragraph("4. Corpus statistics", H2)]
ws = df.groupby("length_category")[["page_count", "words"]].agg(["min", "median", "max"]).reindex(tiers)
r2 = [["Tier", "Papers", "Pages min/median/max", "Words min/median/max"]]
for t in tiers:
    p, w = ws.loc[t, "page_count"], ws.loc[t, "words"]
    r2.append([t, int((df.length_category == t).sum()), f"{int(p['min'])} / {int(p['median'])} / {int(p['max'])}", f"{int(w['min']):,} / {int(w['median']):,} / {int(w['max']):,}"])
E += [Paragraph("Table 2 - length statistics per tier", SM), tbl(r2, [3*cm, 2.5*cm, 5*cm, 5.5*cm]), Spacer(1, 6)]
r3 = [["Metric", "Value"], ["Total papers", len(df)], ["Total words (extracted text)", f"{int(df.words.sum()):,}"],
      ["Mean words per paper", f"{df.words.mean():,.0f}"], ["Median pages", f"{df.page_count.median():.0f}"],
      ["Papers from 2020 or later", int((df.year >= 2020).sum())], ["Papers before 2015", int((df.year < 2015).sum())]]
mx = df.loc[df.page_count.idxmax()]
E += [Paragraph("Table 3 - overall statistics", SM), tbl(r3, [7*cm, 4*cm]), Spacer(1, 6),
      Paragraph("Figure 2 - publication years", SM), Image("chart_years.png", width=14*cm, height=14*cm*2.6/6.4),
      Paragraph("5. Limitations", H2),
      Paragraph(f"- <b>Sub-topics are balanced (about 100 per topic) but length tiers are not:</b> long papers are {100*(df.length_category=='long').mean():.0f}% of the corpus, and "
                f"machine translation ({int(ct.loc['Machine Translation','short'])}) and summarization ({int(ct.loc['Summarization','short'])}) have few short papers. All three tiers are present in every topic.", B),
      Paragraph(f"- <b>No date filter.</b> The corpus mixes papers from {int(df.year.min())} to {int(df.year.max())}; most are 2017 or later.", B),
      Paragraph(f"- <b>One outlier:</b> arXiv {mx.arxiv_id} has {int(mx.page_count)} pages (a thesis-length document); it is kept and affects only the maximum page count.", B),
      Paragraph("- <b>Labels come from the query</b> (a paper gets the topic whose query returned it) and were not hand-verified; a paper on two topics keeps one label.", B),
      Paragraph("- <b>Page count includes references and appendices</b>, and text extracted from PDFs contains headers, formulas and reference lists.", B)]
doc.build(E)
print("report ok")
