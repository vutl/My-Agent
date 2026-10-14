# 09 — Sealed multi-format evaluation (plan, 2026-10-15)

Goal: measure Aya on documents it was never tuned on, in formats a real user
uploads, once, with an independent author and judge. The 21 SER papers and
every suite built on them are regression sets from now on.

## Roles

| Role | Who | Writes | Must not |
|---|---|---|---|
| Head agent | Claude (main session) | plan, ingest/run/report scripts, final report | read the questions before the run |
| A. Corpus builder | Claude sub-agent | `data/retrieval_eval/sealed/v1/raw/*`, `manifest.json` | read Aya code |
| B. Question author | Claude sub-agent (fresh context) | `sealed/v1/questions.jsonl` + notes | read Aya code, run Aya |
| C. Embedding bake-off | Claude sub-agent | retrieval-only report | change production config |
| D. Judge | Claude sub-agent (fresh context) | per-turn verdicts | see which model answered |

## Quota policy

- 9router has only the two `cx` accounts (gh/cu/ag pools are unusable).
  `cx` quota is spent on **Aya answering with `cx/gpt-5.6-sol`** (the user's
  choice) and on ingestion steps that call a model. Authoring, judging,
  corpus building and analysis run as Claude sub-agents (no 9router quota).
- Keep headroom for the user's own Aya use: one sealed run, no reruns of the
  whole set; reruns only of failed turns after fixes.

## Corpus (target ~20 documents, A)

- Native PDF papers outside speech emotion (NLP, vision, medicine, economics).
- Scanned PDF (image-only; made by rasterising a native PDF or a real scan).
- DOCX with tables; Markdown/TXT; slide-deck PDF.
- Vietnamese documents (report / regulation / article), at least 3.
- Public, redistributable sources only; manifest records URL, license,
  format, language, domain, pages.

## Questions (~80 turns, B)

Single and multi-turn (switch documents and return, vague references),
numbers from tables, figure/caption questions, formulas, cross-document
comparison, library questions, typos/no-diacritics, false premises, and ~15%
unanswerable. Each turn: reference answer, evidence location, and
`must_mention` / `must_not_mention`. Schema = `evals/aya_hard_v1.jsonl`.

## Run (head)

1. Fresh isolated runtime `sealed/v1/runtime`, backend on :7791, Aya's own
   upload/index pipeline (tests parsing: OCR, DOCX tables, Vietnamese).
2. Parse audit: per document chunk/table/figure counts, empty-text pages.
3. One end-to-end run with `cx/gpt-5.6-sol`, answers saved.
4. Judge D; head spot-checks ≥20% of verdicts by hand.
5. Report: accuracy by category/format/language with intervals, fabricated
   numbers on unanswerable turns, false abstentions, latency, quota used.

## Gates (from plan 08 / 00)

Correct ≥ 85%; 0 fabricated numbers on unanswerable turns; false abstention
≤ 10%. Failures are classified (parse / retrieval / routing / generation)
and fixed generically; the set is then regression-only.

## Embedding question (C)

Retrieval-only bake-off on the sealed corpus: `embeddinggemma:300m`
(current) vs `qwen3-embedding:0.6b`, `bge-m3` (and `qwen3-embedding:4b` if
time): evidence recall@5/10, MRR, Vietnamese vs English, index time. Switch
only if a model wins clearly on both languages at acceptable latency.
