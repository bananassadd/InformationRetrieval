# AI usage disclosure - Phase 1

| Tool (version) | Used by | What it covers | Transcript file |
|---|---|---|---|
| Claude (Sonnet 5.5), claude.ai | Banan | choosing the sub-topics, arXiv collection script and timeout handling, Git/PowerShell setup help, review of `app.py` and the corpus, cleaning script, query-handling code in `search_utils.py` and the matching changes to `app.py`, corpus report | `claude_chat_transcript.md` |
| TODO: teammate - tool name and version | Saredigi | TODO: e.g. original `download_corpus.py`, original `app.py` (Streamlit GUI, BM25 + dense + RRF), README | TODO: add the exported chat or share-link file here |

If the teammate used no AI tool for their part, replace the TODO row with: "No AI tool was used for <files>."

Notes for the discussion: sub-topic choice and the "6 pages go to short" rule were our own decisions; the review found
that the original query handling lacked phrase matching and highlighting and that spelling correction was changing
technical terms (for example `lstm` -> `let`), which is why those parts were rewritten.
