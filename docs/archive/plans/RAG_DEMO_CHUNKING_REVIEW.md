# RAG review: demo_chunking branch ReAct vs current app

Scope: only RAG-related logic in `demo_chunking` on branch `ReAct`, compared with the current project RAG. This is a static code review, not a runtime benchmark.

## Short verdict

`demo_chunking` has a stronger multimodal/document-ingestion design than our current RAG. It extracts text, tables, images, summarizes images through a vLLM-compatible vision model, stores image assets, indexes `text/table/image` chunks into Qdrant, and exposes a ReAct agent with separate `retrieve_text` and `retrieve_images` tools.

The best part to borrow is its agent streaming shape: `ChatService._stream_chat()` streams from `agent.astream_events(...)` and sends token deltas while LangGraph/ReAct is running, then sends final metadata/images. That is closer to a real agent stream than our current "retrieve first, then stream final Ollama answer" flow.

Important caveat: some advertised retrieval pieces are not wired into the main chat path. `RRF_retrieval.py`, `hybrid_retrieval.py`, and `RerankerService` exist, but `app/services/retrieval/retriever_service.py` currently uses only embedding + Qdrant vector search. So for the live chat API, it is dense vector retrieval with chunk-type filtering, not true BM25+dense+rerank.

## End-to-end RAG flow in demo_chunking

### Ingestion

1. User uploads `.pdf`, `.docx`, or `.txt` through `app/api/routes/document.py`.
2. `DocumentProcessingService.process_file()` copies the file into `data/uploads/{workspace_id}/{document_id}/source.ext`.
3. `DocumentProcessingService.process_document()` runs phases:
   - `inspect`: choose parser through `UnifiedDocumentProcessor`.
   - `parse`: create `ParsedDocument` with `ParsedChunk`, `ParsedImage`, `ParsedTable`.
   - `dedup`: remove near/exact duplicate chunks.
   - `persist`: save sanitized `parsed.json`.
   - `ensure_collection`: create Qdrant collection for workspace.
   - `index`: embed chunks and upsert them to Qdrant.
4. SQLite metadata stores document status, counts, parser version, timings, and asset metadata.
5. Qdrant stores vector points with payload fields: document id/name, workspace id, chunk type, title, content, page, image/table refs.

### Chat/retrieval

1. `app/api/routes/chat.py` receives `POST /chat`.
2. `ChatService.chat()` resolves workspace/document scope and builds an `AgentToolkit`.
3. `create_rag_agent()` creates a LangGraph ReAct agent with OpenAI-compatible `ChatOpenAI` pointed at vLLM.
4. Agent can call:
   - `retrieve_text()`: search `TEXT` and `TABLE` chunks.
   - `retrieve_images()`: search `IMAGE` chunks and return image display ids.
   - `list_workspaces()`.
   - `list_documents()`.
5. `RetrieverService.retrieve()` embeds query with `EmbeddingService`, queries Qdrant, resolves image/table references through `AssetService`, then returns `RetrievalResult`.
6. `ContextBuilder` merges results, selects chunks by score/char budget, formats context, and maps image ids to `IMG_1`, `IMG_2`, etc.
7. Agent answer includes `[DISPLAY_IMAGES: IMG_1, IMG_2]` or `[DISPLAY_IMAGES: NONE]`.
8. `parse_image_display_commands()` removes that command and maps selected ids to base64 images collected by `AgentToolkit`.

## Streaming logic

### Files

`app/api/routes/chat.py`

- Has `POST /chat` with `stream: bool = False` query parameter.
- If `stream=True`, wraps async generator in `StreamingResponse`.
- Emits Server-Sent Events as raw `data: {...}\n\n`.
- It does not use named SSE events like `event: message.delta`.

`app/services/retrieval/chat_service.py`

- `chat(..., stream=False)` runs the whole agent with `agent.ainvoke(...)`, then extracts final answer and images.
- `chat(..., stream=True)` returns `_stream_chat(...)`.
- `_stream_chat()`:
  - Calls `agent.astream_events({"messages": messages}, version="v2", config={"recursion_limit": 8})`.
  - For every `on_chat_model_stream` event, extracts `event["data"]["chunk"].content`.
  - Yields `{"delta": content}` immediately.
  - Builds `full_answer` from streamed chunks.
  - After the stream ends, parses `[DISPLAY_IMAGES: ...]`.
  - Yields final payload:
    `{"done": True, "images": [...], "source_collections": [...], "source_document_ids": [...], "workspace_id": ...}`.

### Why it is good

- It streams at LangGraph/LangChain agent level, not only at final LLM response level.
- It can stream while the ReAct graph is running.
- It sends a final metadata packet after tokens, including images and source documents.

### Issues to fix if we borrow it

- It yields every `on_chat_model_stream` event. In a ReAct graph, that can include tool-call/planning LLM calls, not only the final answer. Usually tool-call chunks have empty content, but the implementation does not explicitly filter by node/name/tag.
- It streams `[DISPLAY_IMAGES: ...]` as normal deltas, then strips it only after completion. Unless frontend hides that command, users may briefly see internal display syntax.
- SSE envelope has only `data:` and no named event. Our current app's `sse_event("message.delta", ...)` style is cleaner for frontend state machines.
- Tool start/end events are not surfaced to frontend. It streams model tokens, but not structured `tool.started`, `tool.completed`, `retrieval.completed` events.

## Visual RAG status in demo_chunking

It does have visual RAG logic, and it is much more complete than ours.

What exists:

- `ChunkType.IMAGE` in domain model.
- `ParsedImage` with image id, file path, page number, caption, dimensions, mime type, metadata.
- PDF image extraction in `app/ingestion/pdf/image_extractor.py`.
- DOCX image extraction in `app/ingestion/docx_pipeline.py`.
- Docling PDF scan pipeline in `app/ingestion/pdf/docling_pipeline.py` with picture extraction.
- vLLM vision summaries for extracted images.
- Image chunks are embedded and indexed into Qdrant like normal chunks.
- `retrieve_images()` searches only `ChunkType.IMAGE`.
- `AssetService.resolve_image_base64()` reads the saved image file and returns base64 for UI display.
- Agent prompt explicitly tells the model to output `[DISPLAY_IMAGES: IMG_1, IMG_2]`.

Limitations:

- It is "image-summary RAG", not true image-vector retrieval. Retrieval searches the text summary/caption embedding, not CLIP-style visual embeddings.
- It relies on a vLLM/OpenAI-compatible vision model. If that service is unavailable, image chunks still exist but summaries degrade to fallback text.
- No clear bounding boxes/crops exposed to the UI beyond saved extracted image file paths. Page screenshots and layout overlays are not a complete UI feature.
- PDF image extraction has a small code smell: in `extract_images_from_page()`, code after an early `return temp_images_data, images_positions` is unreachable.
- Legacy hybrid/RRF image methods expect base64 in chunk metadata, but the main new mapping strips base64 and uses `AssetService` from file path instead. That reinforces that RRF/hybrid code is legacy or not mainline.

## Main RAG files in demo_chunking

### API layer

`app/api/routes/chat.py`

- Request/response endpoint for chat.
- Validates that user passed a workspace/document/collection scope.
- Switches between normal JSON response and SSE stream.
- Delegates all RAG/agent work to `ChatService`.

`app/api/routes/document.py`

- Uploads and indexes files.
- Supports `.pdf`, `.docx`, `.txt`.
- Resolves or creates workspace.
- Writes upload to temp file, then calls `DocumentProcessingService.process_file()`.
- Returns document metadata, indexed chunk count, timings, warnings.

`app/api/routes/workspace.py`

- CRUD-ish workspace endpoints.
- A workspace maps to a Qdrant collection.
- Deleting a workspace deletes its documents and vector collection through processing service.

`app/api/dependencies.py` and `app/core/providers.py`

- Dependency injection factories.
- Important finding: `get_retriever_service()` builds `RetrieverService` with `EmbeddingService`, `QdrantVectorService`, `AssetService`.
- It does not inject `RerankerService`, `HybridRetriever`, or `RRFRetriever` into the chat path.

`app/api/schemas.py`

- Pydantic response/request DTOs for workspace/document/chat.
- Mostly shape definitions.

### Domain model

`app/domain/types.py`

- Central typed model for the RAG system.
- `DocumentStatus`: upload/parse/index lifecycle.
- `ChunkType`: `text`, `table`, `image`.
- `WorkspaceRecord`, `DocumentRecord`: SQLite metadata objects.
- `ParsedChunk`: normalized parser chunk with page, heading, image refs, table refs.
- `ParsedDocument`: parsed file output with chunks/images/tables.
- `ParsedDocument.from_legacy_result()`: adapts older parser output into the new domain model.
- `ParsedImage`, `ParsedTable`: visual/table artifact records.
- `RetrievedChunk`, `RetrievedImage`, `RetrievalResult`: query-time objects.
- Caveat: `ParsedChunk.from_legacy_chunk()` removes `base64` from metadata, which is good for SQLite/vector payload size, but means old retrieval classes expecting metadata base64 are stale.

### Agent and chat

`app/bot/react_agent.py`

- Builds the LangGraph ReAct agent around `ChatOpenAI`.
- Uses `settings.VLLM_API_URL`, `settings.VLLM_API_KEY`, `settings.VLLM_MODEL_NAME`.
- `build_react_prompt()` fills workspace and collection scope into `react_prompt.txt`.
- `extract_agent_answer_text()` finds the final non-tool AI message.
- `parse_image_display_commands()` strips `[DISPLAY_IMAGES: ...]` and returns selected image ids.
- `_sanitize_model_output()` removes model artifact tokens.

`app/bot/prompts/react_prompt.txt`

- Vietnamese ReAct instruction.
- Tells agent when to use `retrieve_text` vs `retrieve_images`.
- Forces final image command syntax.
- Good idea: makes visual display an explicit contract between model and UI.

`app/bot/tools.py`

- Defines `AgentToolkit`, a per-request stateful tool bundle.
- `retrieve_text()`:
  - Resolves workspace/collection/document scope.
  - Queries `ChunkType.TEXT` and `ChunkType.TABLE`.
  - Merges results and builds context with tables included, images excluded.
  - Tracks source documents and collections.
- `retrieve_images()`:
  - Queries `ChunkType.IMAGE`.
  - Builds image context.
  - Stores selected image base64 in `image_base64_map`.
  - Rewrites internal image ids to display ids like `IMG_1`.
- `list_workspaces()` and `list_documents()` are support tools.

`app/services/retrieval/chat_service.py`

- Orchestrates system prompt, retrieval scope, toolkit, agent run, streaming/non-stream response.
- This is the core file for the streaming logic.

### Retrieval and context

`app/services/retrieval/retriever_service.py`

- Main live retrieval service used by chat tools.
- Embeds query through `EmbeddingService`.
- Calls Qdrant vector search.
- Applies chunk type and document filters.
- Converts chunks into citations, image refs, and table refs.
- No BM25, RRF, or reranker in this live path.

`app/retrieval/context_builder.py`

- Merges multiple `RetrievalResult`s.
- Deduplicates chunks, citations, images, tables.
- Selects top chunks by score with char budget.
- Formats text/table chunks with source citation and type.
- Formats image chunks as `[ẢNH n - ID: IMG_n]` plus title/source/summary.
- Creates `image_display_map` for model-facing ids.

`app/retrieval/RRF_retrieval.py`

- Legacy/alternative retriever.
- Builds BM25 index with Vietnamese tokenization using PyVi.
- Runs vector search and BM25 search.
- Combines ranks by Reciprocal Rank Fusion.
- Has text and image formatting helpers.
- Not wired into `ChatService` or dependency providers.

`app/retrieval/hybrid_retrieval.py`

- Alternative score-fusion retriever.
- Normalizes vector and BM25 scores and combines by alpha.
- Also not wired into main chat path.

`app/services/ml/reranker_service.py`

- Lazy-loads `SentenceTransformer`.
- Reranks by cosine similarity between question embedding and text embeddings.
- Has idle cleanup thread.
- Not wired into main chat path.
- Despite README mentioning CrossEncoder reranking, this code is embedding cosine reranking, not CrossEncoder pairwise scoring.

### Vector store

`app/services/vector/qdrant_vector_service.py`

- Qdrant adapter.
- Ensures workspace collection exists.
- Upserts parsed chunks into Qdrant.
- Queries workspace with optional chunk type/document filters.
- Deletes documents/collections.
- Lists chunks/collections.
- Migrates legacy per-document collections to workspace collections.

`app/services/vector/vector_mappers.py`

- Converts `ParsedChunk` to Qdrant `PointStruct`.
- Builds stable point id from `document_id:chunk_ref`.
- Builds Qdrant filters for chunk type and document ids.
- Converts Qdrant points back into `RetrievedChunk`.
- Sanitizes heavy metadata, removing base64 and file-specific fields.

`app/services/vector/vector_interfaces.py`

- Interface/protocol boundary for vector service methods.

### Model services

`app/services/ml/embedding_service.py`

- Lazy-loads local SentenceTransformer embedding model.
- Supports async encoding through `asyncio.to_thread`.
- Has idle cleanup thread and CUDA memory cleanup.
- Used in live retrieval and indexing.

`app/services/ml/reranker_service.py`

- Covered above. Exists but not used in live chat/retrieval provider graph.

### Ingestion orchestration

`app/services/document/document_processing_service.py`

- Main ingestion service.
- Creates document id.
- Copies upload to managed storage.
- Registers document in SQLite.
- Runs inspect/parse/dedup/persist/index phases.
- Embeds chunks in batches and upserts to Qdrant.
- Updates document counts, parser version, timings, asset metadata.
- Deletes vector points and metadata on document delete.
- Caveat: calls `embedding_service.cleanup()` in `finally`, so repeated indexing may reload model often.

`app/services/document/document_processing_support.py`

- Storage/path helpers.
- JSON persistence.
- `chunk_text()` decides what text gets embedded: title + content.
- Serializes image/table assets into document metadata.
- Phase timing helpers.

`app/services/document/chunk_dedup_service.py`

- Deduplicates parsed chunks before indexing.
- Useful for PDFs with repeated headers/images/tables.

`app/services/document/document_service.py`

- CRUD wrapper around document repository.
- Status updates and document listing.

`app/services/document/asset_service.py`

- Resolves image asset metadata from document metadata.
- Resolves relative image path under `asset_dir`.
- Reads image bytes and returns base64.
- This is the bridge from visual retrieval to UI display.

### Parser selection

`app/ingestion/document_processor.py`

- `UnifiedDocumentProcessor` chooses parser through `DocumentInspector`.
- Parser registry:
  - `.pdf` -> `PDFParser`
  - `.pdf_scan` -> `PdfScanParser`
  - `.docx` -> `DocxParser`
  - `.txt` -> `TxtParser`
- Provides CLI batch/single processing too.

`app/ingestion/inspector.py`

- Inspects file extension and PDF text layer/profile.
- Decides whether a PDF should use regular parser or scan/Docling parser.

`app/ingestion/parsers/base.py`

- Parser interface.

`app/ingestion/parsers/pdf_parser.py`

- Uses `create_chunks_from_pdf()` from `legacy_pipeline.py`.
- For text-layer PDFs, still extracts tables, images with vLLM summaries, and heading-based text chunks.

`app/ingestion/parsers/pdf_scan_parser.py`

- Uses Docling pipeline for scan/image-heavy PDFs.
- Calls a synchronous Docling function inside async parser wrapper.

`app/ingestion/parsers/docx_parser.py`

- Uses `create_chunks_from_docx()`.

`app/ingestion/parsers/txt_parser.py`

- Text parser. Simpler text chunking path.

### PDF ingestion internals

`app/ingestion/pdf/legacy_pipeline.py`

- Main pipeline for text PDFs:
  - Extract tables first.
  - Extract images asynchronously from all pages.
  - Summarize each image with vLLM vision.
  - Extract text by headings while skipping header/footer/table regions.
  - Split long chunks by numbered items.
- Produces legacy chunk dicts with `type=text/table/image`.

`app/ingestion/pdf/table_extractor.py`

- Uses Camelot lattice extraction and PyMuPDF table areas.
- Cleans DataFrames.
- Converts tables to Markdown.
- Detects table titles from nearby text.
- Merges continued tables across pages.
- Skips spurious one-column/caption-like tables.

`app/ingestion/pdf/image_extractor.py`

- Uses PyMuPDF image extraction.
- Deduplicates by MD5 hash.
- Saves images to `images_{pdf_filename}/`.
- Calls vLLM vision asynchronously with concurrency.
- Returns image metadata including filename, path, page, size, base64, summary.
- Also has helper functions to skip table/header/footer text and align captions with images.

`app/ingestion/pdf/docling_pipeline.py`

- Uses Docling if installed.
- Detects whether PDF has text layer.
- Configures OCR/table/picture extraction.
- Iterates Docling document items:
  - Title/section headers update heading path.
  - Text items fill text buffer.
  - Table items become Markdown table chunks.
  - Picture items are saved as images and summarized by vLLM.
- Produces page-aware text/table/image chunks.

`app/ingestion/pdf/analyzer.py`

- Header/footer and heading detection helpers for PDF text extraction.

### DOCX ingestion internals

`app/ingestion/docx_pipeline.py`

- Extracts tables and converts to Markdown.
- Splits large tables by section.
- Extracts images from relationships/runs.
- Finds image captions.
- Saves images and summarizes them through vLLM vision.
- Extracts text chunks by headings.
- Splits long text chunks by numbered sections.

`app/ingestion/docx_tables.py`

- Table row parsing/splitting helpers.

`app/ingestion/heuristics/table_titles.py`

- Heuristics to infer table title from previous paragraphs/elements.

### Metadata and workspace

`app/repositories/sqlite_metadata.py`

- SQLite persistence for workspace/document metadata.
- Stores status, counts, parser info, asset metadata.

`app/repositories/interfaces.py`

- Repository interface definitions.

`app/services/workspace/workspace_service.py`

- Workspace CRUD.
- Creates collection names.

`app/services/workspace/workspace_index_migration_service.py`

- Migrates old document-level vector collections to workspace-level collections.

## Comparison with our current RAG

| Area | demo_chunking ReAct | Current app |
| --- | --- | --- |
| Runtime target | FastAPI + Qdrant + vLLM + local model files | FastAPI + SQLite + LanceDB + Ollama, Mac/local-first |
| Core storage | SQLite metadata + Qdrant vectors | SQLite source of truth + FTS5 + LanceDB vectors |
| File model | Workspace -> Qdrant collection -> documents | Approved folders/files -> collections -> documents |
| Chunk types | `text`, `table`, `image` as first-class vector chunks | Mainly `text`; tables/figures stored as metadata tables, not first-class retrieval chunks |
| PDF parsing | PyMuPDF/Camelot/Docling, table extraction, image extraction, vLLM summaries | pypdf text extraction, page-aware chunks, caption regex for tables/figures |
| DOCX parsing | Tables and images extracted; image summaries | Paragraph text only, no real DOCX table/image extraction |
| Visual RAG | Yes: extracted image files + VLM summary chunks + `retrieve_images` + base64 display | Partial: figure captions/metadata only, no image crop extraction or VLM summary |
| Table RAG | Stronger: tables are chunks, Markdown indexed | Partial: Markdown table extraction for `.md`, regex captions for PDF/text; tables not vector-indexed separately |
| Retrieval live path | Dense Qdrant vector search with chunk filters | Hybrid LanceDB text chunks + SQLite FTS5, RRF merge, lexical boost |
| BM25/RRF | Code exists but not main chat path | Actively used in `RagService.search_hybrid()` via FTS + vector RRF |
| Reranker | Service exists but not wired; cosine embedding rerank | Lightweight lexical rerank boost wired into hybrid merge |
| Streaming | LangGraph `agent.astream_events()` token stream + final metadata | Named SSE events; agent retrieves first, then streams final Ollama answer |
| Tool events | Not emitted to frontend, only model deltas and final done | Emits `run.started`, `retrieval.started/completed`, `planner.started/completed`, `message.delta`, `run.completed` |
| Local robustness | Heavier external services and hard-coded defaults | More local-first and easier to run on user's Mac |
| Memory/model cleanup | Lazy embedding/reranker load, idle cleanup | Ollama manages LLM/embedding runtime externally |

## What demo_chunking does better

1. Visual RAG architecture is much better.
   It treats image chunks as retrievable objects and has a UI-display path through `AssetService`.

2. Parser quality is much richer.
   PDF table extraction, image extraction, Docling fallback, DOCX table/image parsing, heading-aware chunks, and dedup are all more advanced than ours.

3. Context building is cleaner.
   `ContextBuilder` returns a structured `ContextBuildResult` with chunks, citations, images, tables, and `image_display_map`.

4. Agent tools are cleanly separated.
   `retrieve_text` and `retrieve_images` let the agent route intent naturally.

5. Streaming is closer to true agent streaming.
   `agent.astream_events()` is the right layer if we want token streaming from LangGraph execution rather than only final answer streaming.

## What our current RAG does better

1. Hybrid retrieval is actually wired.
   Our `RagService.search_hybrid()` really merges LanceDB vector chunks and SQLite FTS chunks with RRF.

2. Desktop/local-first assumptions are stronger.
   Our file catalog, approved folders, selected file indexing, SQLite source of truth, LanceDB local store, and Ollama runtime fit the Mac app better than Qdrant + vLLM + hard-coded network endpoints.

3. SSE event protocol is cleaner.
   Our stream has named events. Frontend can respond to lifecycle events without parsing generic JSON blobs.

4. Page-aware chunks are already active.
   Our parser/chunker stores `page_number` on chunks and includes page in composed context.

5. Current codebase has less external service weight.
   Demo's visual pipeline is stronger but depends on vLLM vision availability and heavier libraries.

## Best parts to borrow into our RAG

Priority 1: structured visual model

- Add first-class `chunk_type` retrieval for `text`, `table`, `figure/image`.
- Add `ParsedImage`/`RetrievedImage` equivalent.
- Add local `AssetService` to resolve image path/base64.
- Store image artifacts under a managed app data folder.

Priority 2: visual ingestion

- PDF: extract actual images/crops with PyMuPDF.
- DOCX: extract embedded images and tables.
- Generate `visual_summary` through Ollama vision or a later local VLM.
- Index figure/image summaries into LanceDB and SQLite FTS.

Priority 3: context builder upgrade

- Replace ad hoc context composition with a richer result object:
  - selected chunks
  - citations
  - tables
  - figures/images
  - display id map
  - context stats

Priority 4: better agent streaming

- Keep our named SSE envelope.
- Borrow the `agent.astream_events()` concept.
- Emit structured events:
  - `tool.started`
  - `tool.completed`
  - `retrieval.completed`
  - `message.delta`
  - `message.completed`
- Filter final-answer model stream carefully so tool-call/planner internals do not leak.
- Do not stream internal `[DISPLAY_IMAGES: ...]` syntax to the user; either buffer the tail or use structured tool result selection.

Priority 5: table extraction

- Borrow table extraction ideas, but avoid pulling in Camelot/Docling blindly until we decide dependency weight.
- Start with PyMuPDF table extraction where available, then optional advanced parser.

## What not to copy directly

- Do not replace LanceDB with Qdrant unless we decide to move away from local-first desktop storage.
- Do not copy hard-coded `QDRANT_URL`, `VLLM_API_URL`, model names, or network assumptions.
- Do not assume `RRF_retrieval.py` is production-ready; it is not integrated into the live chat path.
- Do not copy the raw `data:` SSE format. Keep named SSE events.
- Do not copy the exact `[DISPLAY_IMAGES: ...]` user-visible command behavior; make it a hidden structured result.

## Practical conclusion

For our Phase 3.5/v1.5 RAG, the most valuable delta is not its Qdrant setup. It is the multimodal object model and ingestion path:

- text/table/image as first-class chunks
- image extraction into asset storage
- VLM-generated visual summaries
- `retrieve_images` as a separate tool
- final response metadata carrying images/sources

Our current retrieval core is more appropriate for the Mac app because it already has local SQLite + FTS5 + LanceDB hybrid search. The right move is to graft the visual/table/context/streaming ideas onto our local stack, not to port the whole demo architecture.
