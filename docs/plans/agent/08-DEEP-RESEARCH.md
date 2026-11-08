# 08 — Deep research

Written 2026-11-08. Goal: give Aya a research mode on the level of ChatGPT deep research,
Gemini Deep Research and Claude Research. The user wants it and asked for "hơn thì càng tốt"
(the more the better).

Aya takes a question, works for minutes in the background, and comes back with a long, cited
report. To get there she:
- searches many kinds of sources;
- opens and reads the pages and documents (HTML, PDF, papers, the user's own library);
- follows leads;
- expands the question into related questions;
- resolves contradictions;
- synthesises what she finds.

## 1. What the reference products do (and what we copy)

| Behaviour | ChatGPT | Gemini | Claude | Aya |
|---|---|---|---|---|
| Clarifying questions before starting | yes | — | sometimes | yes, at most 3, skippable |
| Editable research plan shown first | — | yes | — | yes (plan card: approve / edit / start) |
| Many parallel searches, hundreds of pages | yes | yes | yes (lead agent + parallel sub-agents) | yes: lead + parallel workers |
| Reads PDFs/papers, not only snippets | yes | yes | yes | yes (PDF + academic APIs + library) |
| Iterates: gaps → new queries → more reading | yes | yes | yes | yes, rounds until saturated or budget |
| Python for analysis/charts | yes | — | yes (analysis tool) | yes (sandboxed `run_python`) |
| Long structured report, inline citations | yes | yes | yes (separate citation pass) | yes + claim-level citation check |
| Runs in background, notifies when done | yes | yes | yes | yes (task runner + notify/Telegram) |
| Follow-up chat on the findings | yes | yes | yes | yes: sources become a library collection |
| Export | PDF | Docs, audio overview | — | md + docx (+ spoken summary in Aya's voice) |

**What can make Aya better than these products for this user:**
1. She also reads the user's own library and indexed papers, together with the web.
2. Every number in the report is checked against its cited source, reusing the grounding
   discipline built for RAG v1.
3. The sources remain queryable after the report: the existing RAG answers follow-ups with
   `[S#]` citations.
4. She speaks a summary in her own voice.

## 2. Architecture

A research run is a **task** (01-AGENT-CORE) with `kind="research"`. It reuses the task store,
background worker, restart resume, budgets, notifications and SSE events. It adds its own
tables and a dedicated orchestrator. The generic step runner is not used for it, because
research needs parallel workers and rounds.

```
question ──► 0 Scope ──► 1 Plan ──► 2 Research rounds ─────────► 3 Synthesis ──► 4 Verify ──► 5 Deliver
             clarify     sub-questions  ┌─ worker × N (parallel) ─┐   outline        claim ↔ source   report (chat + file)
             (≤3 Qs)     + angles      │ search → triage → read  │   section-by-   check, fix or    related questions
                         user approves │ → evidence cards        │   section write  drop claims      sources → collection
                                       │ → follow-up leads       │   with [W#]/[S#]                 spoken summary
                                       └─► lead: gaps, conflicts, new sub-questions (repeat until done)
```

### 0. Scope (lead, Sol)
- One call decides whether the question is clear enough.
- If not, Aya asks at most 3 clarifying questions with an `ask_user` card that offers options.
  Typical questions: purpose, time range, region, depth, report language.
- "Bỏ qua, cứ làm đi" (skip, just go) proceeds with stated assumptions. The report lists those
  assumptions.

### 1. Plan (lead, Sol)
The lead turns the question into a research brief:
- **Objective.**
- **5–12 sub-questions.** Each has an angle (background, state of the art, numbers, opposing
  views, recent news, local/Vietnamese context, practical how-to) and source hints (web, news,
  academic, library, official/government, forums).
- **Planned report outline.**
- **Depth level and budget.**

The plan is shown as a card (sub-questions, outline, estimated time and calls). The user can
start, edit (add/remove/reword sub-questions) or cancel. When a run is started from Telegram or
a schedule, the plan is approved automatically.

### 2. Research rounds
**Workers.** Each worker handles one sub-question in its own context; the lead never sees raw
pages. N workers run in parallel (default 3, capped by quota). Each worker runs a bounded loop:
1. **Query generation** (luna): 3–6 queries per sub-question, in the right languages (vi/en,
   plus the local language for the topic), with operators for site/time/filetype where useful.
2. **Search fan-out** across the providers in §3, all at once. Results are deduplicated by
   canonical URL and title similarity.
3. **Triage** (luna): rank candidates by relevance, authority (official, primary, peer-reviewed
   above aggregators/SEO) and freshness. Keep the top K (default 6 per round).
4. **Read:** fetch and extract the main text (HTML, PDF, academic full text when open).
   Follow at most one hop of promising links (references, "source", PDF links).
5. **Extract evidence cards** (luna, one call per page):
   `{claim, quote (verbatim, ≤ 300 chars), numbers [{value, unit, what, as_of}], source_id,
   locator (page/section), stance, confidence}`. Cards only hold what the page says. The
   extractor never adds outside knowledge, and a quote must occur in the page text (checked
   deterministically).
6. **Reflect** (luna): is this sub-question answered? Which new leads appeared (entities,
   sub-topics, contradicting claims)? Then either stop or loop with new queries. Each worker
   is bounded by its round budget.

**Lead between rounds** (Sol), working only on the cards:
- coverage per sub-question;
- contradictions between sources (same quantity, different values) → a targeted verification
  sub-question;
- **expansion**: related questions the user did not ask but would want ("mở rộng các câu hỏi
  liên quan"). These become new sub-questions if the budget allows, otherwise they are listed
  under "Hướng tìm hiểu tiếp" (further directions).

**Stop when** all sub-questions are covered with ≥ 2 independent sources for key claims, a
round adds < 10% new cards, or the budget is spent. Whichever happens, the report states which
of these stopped the run.

### 3. Synthesis (lead, Sol)
1. **Outline:** refine the planned outline from the cards.
2. **Section-by-section writing:**
   - each section gets only its own cards, so context stays small and the run scales to
     hundreds of cards;
   - inline citations `[W#]` for web sources and `[S#]` for library sources;
   - tables where the cards hold comparable numbers;
   - a chart when a section compares quantities (§5).
3. **Fixed report parts:**
   - TL;DR (5–8 bullets);
   - key findings;
   - the sections;
   - disagreements and uncertainty;
   - assumptions and limits;
   - related questions / further directions;
   - sources (grouped by type, with access date).
4. **Language:** the user's language. Sources keep their original titles.

### 4. Verify (Sol + deterministic)
**Deterministic checks:**
- every `[W#]` and `[S#]` resolves to a fetched source;
- every number in the report appears in a card of the source it cites (reuse
  `numeric_grounding` normalisation);
- no URL appears that was not fetched.

**Model check** (one Sol call per section): each sentence that cites a source is checked
against that source's cards (supported / partly / unsupported).

**Repair:** unsupported claims are rewritten from the cards or removed, never left in. The
report keeps a verification footer:
- counts of claims checked, fixed and removed;
- sources read vs found.

### 5. Deliver
The report is delivered in several forms:
- **Chat:** the report streams into the chat as Aya's message (`message.delta`). Citations
  are hoverable/clickable.
- **Files:** `~/Aya/reports/<date>-<slug>.md` and `.docx`, the latter through the P2 document
  tools. Until those exist, md only.
- **Sources as a collection:** fetched sources (text + metadata) are indexed into a library
  collection `research:<task id>`. Indexing reuses ingestion with `web_page`/`pdf` types and
  shows the same progress. Follow-up questions in that conversation then go through the
  existing RAG with `[S#]`, scoped to that collection by default.
- **Related-question buttons** under the report start a follow-up in the same collection.
- **Notification** (app, macOS, Telegram when configured) with the TL;DR. Aya speaks a 20–40 s
  summary when voice is on.

## 3. Sources and tools

| Source | How | Status | Notes |
|---|---|---|---|
| Web search | provider interface `SearchProvider.search(q, lang, region, recency, n)` | DuckDuckGo today | add Tavily, Brave, and Google via Gemini grounding (needs the user's OK, §7); fan out to all configured providers, merge |
| Web pages | `web_fetch` (trafilatura) | exists | add: canonical URL, published date, author, language, `max_chars` paging, per-domain politeness (1 req/s), cache by URL for 7 days |
| PDF (web) | `web_fetch` returns PDF bytes → text | **missing** | reuse the RAG PDF parser (pymupdf path) for text + page numbers; cap pages |
| Academic | arXiv API, Semantic Scholar Graph API, OpenAlex, Crossref (all free) | missing | paper metadata, abstracts, citation counts, open-access PDF links; follow references one hop |
| Encyclopedic | Wikipedia REST (vi/en) | missing | background sub-questions; never the only source of a key claim |
| News | provider recency filter + news-site search | partial | dates are mandatory on news cards |
| User's library | `search_library` / RAG retrieval | exists | library cards become `[S#]`; never mixed up with web sources |
| JS-heavy pages | Playwright browser (P2) | later | fallback when the extracted text is empty |
| Video | YouTube transcripts | later (R7) | optional |

All fetched content is **untrusted**. Workers have **read-only tools** (search, fetch, library,
academic APIs). They have no side-effect tools, so prompt injection in a page cannot act. Cards
quote pages and never execute anything. The lead's prompts carry the existing
untrusted-content wrapper.

## 4. Models, quota and cost

The constraint is real. Both Codex accounts share one 9router quota with chat, and on
2026-10-29 only one account worked. A research run must not take Aya's chat down.

| Job | Model | Why |
|---|---|---|
| Scope, plan, between-round lead, outline, section writing, verification | `cx/gpt-6.1-sol` | quality-critical, few calls |
| Query generation, triage, card extraction, worker reflection | `cx/gpt-6-luna` | many cheap calls |

- **Depth levels** (user picks, default "Chuẩn"):

  | Level | Sub-questions | Rounds | Pages read | Wall time | Calls (Sol / luna) |
  |---|---|---|---|---|---|
  | Nhanh | 3–4 | 1 | ≈ 15 | 3–5 min | ≈ 10 / 40 |
  | Chuẩn | 5–8 | 2 | ≈ 40 | 8–15 min | ≈ 20 / 120 |
  | Sâu | 8–12 | 3–4 | ≈ 100 | 20–40 min | ≈ 40 / 300 |

  The numbers are design estimates; R5 measures the real ones and this table is corrected.
- **Gateway scheduler:** a process-wide priority semaphore for LLM calls. Interactive chat
  turns always go first. Research calls take at most `RESEARCH_MAX_CONCURRENCY` (default 2)
  slots and back off on 429 without failing the run. A run paused by quota resumes by itself
  and shows "đang chờ quota" (waiting for quota) on the stage.
- **Budgets** enforced by the task runner: calls per model, pages, wall time. At 80% of the
  budget the lead stops expanding and moves on to synthesis.

## 5. Visualisation inside reports

This section is shared with the visualisation plan (09). It applies when a section compares
quantities from cards, such as prices over time, benchmark scores or market shares:
- the lead asks `run_python` to plot **only card numbers**: the data is passed as JSON built
  from cards, never retyped by the model;
- the chart is saved next to the report and embedded inline in the chat message at that
  section. The UI does not show it inside a tool card;
- every chart has a caption with its sources;
- a deterministic check makes sure the plotted values equal the cards.

## 6. UI

- **Starting a run:** a "Nghiên cứu sâu" (deep research) toggle in the composer, or
  `/research …`, or the router/Sol hand-off when the user asks for a report.
- **Plan card:** sub-questions (editable), outline, depth selector, estimated time/calls,
  [Bắt đầu] (start) / [Sửa] (edit) / [Huỷ] (cancel).
- **Live research panel** on the task card:
  - current round, sub-questions with status;
  - live feed of queries and sources being read (favicon + title);
  - counts (sources found / read / cards);
  - [Dừng & viết báo cáo] (stop and write now) and [Huỷ] (cancel).
- **Stage:** Aya shows "đang nghiên cứu…" (researching) with progress, and stays usable for
  chat meanwhile.
- **Report view:**
  - citations open a side panel with the quote and a link;
  - sources list with type badges;
  - related-question buttons;
  - export (md/docx).

## 7. Decisions needed from the user

1. **Google search via Gemini grounding.** It needs explicit approval (it adds a non-`cx/`
   model, used only as a search provider) and a key in `backend/.env` set by the user (AI Pro
   Cloud credits). Without it: DuckDuckGo plus academic APIs plus optional Tavily/Brave keys.
2. Optional Tavily / Brave keys.
3. The default depth level and the report language (default: the question's language).

## 8. Evaluation and gates

**Sealed research suite `~/.aya/sealed/research-v1`:**
- built by an independent agent before implementation;
- the implementer and the head agent never read it before the run;
- **20 questions**, vi + en, mixed types:
  - academic survey;
  - technical how-to/comparison;
  - market/price with dates;
  - Vietnamese law/policy;
  - current events;
  - a question whose premise is false;
  - a question with little public information.
- Each question has a rubric: must-cover points, key facts with acceptable sources, traps
  (common misinformation, outdated numbers).

**Metrics** (judge = harness `judge_ag` → `judge`, never the answer model; 20% spot-checked by
the head agent):

| Metric | Gate |
|---|---|
| Citation validity: sampled cited sentences supported by the cited source (judge fetches the source) | ≥ 90% |
| Fabricated sources/URLs (cited but never fetched / nonexistent) | 0 |
| Numbers without a supporting card | 0 |
| Coverage of the rubric's must-cover points | ≥ 80% mean |
| Distinct sources read per "Chuẩn" run | ≥ 15, ≥ 6 distinct domains |
| False-premise / low-information questions: says so instead of inventing | 2/2 |
| "Chuẩn" wall time (median) | ≤ 15 min |
| Chat stays responsive during a run (first text of a chat turn) | ≤ +1 s vs idle |
| Injection pages in the fixture web (hidden instructions) | 0 effects, 0 instructions followed |

**Baseline:** the same questions answered by today's agent route (a single tool loop). The
research mode must beat it on coverage and citation validity by a clear margin, with
confidence intervals reported.

## 9. Work packages

| # | Package | Main files | Depends on | Gate |
|---|---|---|---|---|
| R0 | Sealed research suite + fixture web (offline copies incl. injection pages) + judge rubric runner | `~/.aya/sealed/research-v1`, `scripts/run_research_eval.py` | — (independent author) | runner works on 2 synthetic questions |
| R1 | Source layer: `SearchProvider` interface (DDG, Tavily, Brave, Gemini-grounding stub), PDF extraction in `web_fetch`, academic tools (arXiv, Semantic Scholar, OpenAlex, Crossref), Wikipedia, URL cache, politeness | `tools/web.py`, `tools/research_sources.py` | — | unit tests with recorded responses; PDF page locators |
| R2 | Research store + orchestrator: tables (`research_runs`, `sources`, `cards`, `queries`), scope → plan → rounds → stop rules, workers with read-only tools, card extraction with deterministic quote check, LLM priority scheduler | `services/research.py`, `agents/research_runner.py`, `llm/scheduler.py` | R1, P1 tasks | fake-LLM tests: parallel workers, dedupe, stop rules, resume after restart mid-round, quota back-off, chat priority |
| R3 | Synthesis + verification: outline, sectioned writing with citations, numeric/citation checks, repair, report files, sources → collection ingest | `agents/research_writer.py`, `services/research_report.py` | R2 | fake tests: unsupported claim removed, uncited number rejected, collection searchable |
| R4 | UI: toggle and `/research`, plan card (edit/approve), live panel, report view with citation panel, related-question buttons, export | `apps/desktop` | R2 API | manual + Playwright smoke on the isolated stack |
| R5 | Visualisation in reports (shared with 09): card-backed charts inline, value check | `agents/research_writer.py`, UI | R3 | chart values = cards in tests |
| R6 | Gate run on the sealed suite (isolated backend, quota-scheduled at night) + fixes by failure class | — | R0–R5 | §8 |
| R7 (later) | Browser fallback (P2 Playwright), YouTube transcripts, Gemini grounding live (if approved), spoken summary | | P2 | — |

Execution: as in 07-EXECUTION. Packages go through the harness pipeline; the head agent reads
every diff and runs the gates. R1 and R0 run in parallel; R4 starts once the R2 API is stable.

## 10. Risks

- **Quota:** research multiplies gateway calls. Mitigations: luna for bulk work, the priority
  scheduler, depth levels, night runs for the gate. Re-login of the second account is a
  prerequisite for the "Sâu" level.
- **Blocking:** DuckDuckGo blocks heavy use. Mitigations: provider fan-out, a cache, and
  academic APIs, which are not scraped.
- **Paywalls / JS pages:** detect empty extraction and mark the source as unreadable. Never
  cite a page from its snippet alone.
- **Long-run fragility:** every step is persisted (cards, sources, round state). A restart
  resumes the current round without re-fetching pages that were already read.
