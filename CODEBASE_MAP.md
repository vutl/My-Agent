# CODEBASE_MAP.md

> Muc dich: nho lai nhanh file nao lam gi, luong chay cua app, va cac ham/class quan trong.
> Doc file nay truoc khi sua code.

## 0. Mo hinh tong quat

App co 2 nua chinh:

```text
apps/desktop/  React/Vite/Tauri UI
backend/       FastAPI + LangGraph-ish agent + SQLite + RAG + LanceDB
data/          SQLite DB, LanceDB index, du lieu runtime local
pdf/           tap tai lieu seed de test RAG
scripts/       script chay backend/dev/check Ollama
commands.txt   cac lenh van hanh/reindex/query mau
```

Luong chat co ban:

```text
React ChatPage
-> apps/desktop/src/lib/sse.ts
-> POST /chat/stream
-> backend/app/api/chat.py
-> ChatService
-> OllamaClient
-> stream token ve UI bang SSE
-> ChatHistory luu SQLite
```

Luong RAG/hybrid:

```text
/files/index-folder hoac /rag/index-selected-files
-> IndexingService
-> parse_document
-> chunk_parsed_document
-> SQLite documents/chunks/document_cards/tables/figures
-> /rag/vector/index-all
-> VectorIndexService
-> OllamaEmbeddingProvider
-> LanceDBRetrievalStore
-> /rag/search-hybrid
-> document card vector search
-> chunk vector search + SQLite FTS5
-> RRF merge + lightweight rerank
-> compose_retrieval_context
```

Luong agent file-QA:

```text
React AgentPage
-> POST /agent/run/stream
-> api/agent.py
-> optional hybrid retrieval
-> AgentService/build_agent_graph
-> deterministic source-grounded plan neu co retrieved docs
-> Ollama final answer
-> AgentRunStore luu trace/tool call
```

## 1. Root files

### `README.md`
Gioi thieu du an, cach cai va chay tong quat.

### `commands.txt`
Runbook hien tai. Chua cac lenh:
- start backend/UI
- check Ollama
- index folder `pdf/`
- build LanceDB vectors
- query RAG/hybrid
- inspect tables/figures
- ghi chu Phase 3/3.5 status

### `RAG_LOGIC.md`
Ban do logic RAG: storage, retrieval, hybrid flow, agent flow.

### `CODEBASE_MAP.md`
File nay. Dung de tra cuu nhanh codebase.

### `.env.example`
Mau env var cho backend/frontend.

### `.gitignore`
Bo qua cache/build/runtime data.

## 2. Generated/local folders nen bo qua

Khong can doc tru khi debug build/cache:

```text
apps/desktop/node_modules/
apps/desktop/dist/
apps/desktop/src-tauri/target/
backend/.venv/
backend/.pytest_cache/
backend/app/**/__pycache__/
data/
```

## 3. Backend entrypoint/core

### `backend/app/main.py`
- `create_app()`: tao FastAPI app, init SQLite DB, gan CORS, include routers: health, chat, agent, files, rag, catalog.
- `app = create_app()`: instance uvicorn dung khi chay server.

### `backend/app/core/config.py`
- `PROJECT_ROOT`: root project tinh tu file backend.
- `Settings`: config app, Ollama host/model, data dir, CORS.
- `Settings.sqlite_db_path`: duong dan `data/sqlite/app.db`.
- `Settings.lancedb_path`: duong dan `data/lancedb`.
- `_split_csv(value)`: parse env var dang comma-separated.
- `get_settings()`: doc env var va cache Settings bang `lru_cache`.

### `backend/app/core/events.py`
- `sse_event(event, data)`: format Server-Sent Event: `event: ...\ndata: ...\n\n`.

### `backend/app/db/sqlite.py`
- `SCHEMA`: toan bo schema SQLite: chat, messages, files, documents, chunks, document cards, FTS5, collections, agent runs, tool calls, document tables, document figures.
- `init_db(db_path)`: tao folder DB, chay schema, migrate cot thieu.
- `_migrate_existing_schema(connection)`: them cot moi vao DB cu.
- `_add_missing_columns(connection, table_name, columns)`: helper ALTER TABLE neu cot chua co.
- `connect(db_path)`: context manager mo SQLite connection, set row_factory, commit/close tu dong.

## 4. Backend LLM

### `backend/app/llm/ollama_client.py`
- `OllamaError`: exception khi Ollama loi.
- `ChatCompletion`: dataclass ket qua chat non-stream.
- `OllamaClient.__init__(host, timeout_seconds)`: luu base URL va timeout.
- `OllamaClient.health()`: goi `/api/tags`, tra model list/reachable.
- `OllamaClient.chat(...)`: goi Ollama chat non-stream, tra full message.
- `OllamaClient.stream_chat(...)`: goi Ollama stream, yield token delta.

## 5. Backend API routers

### `backend/app/api/health.py`
- `health()`: endpoint `GET /health`; check config va Ollama models.

### `backend/app/api/chat.py`
Pydantic models:
- `ChatRequest`: body cho chat.
- `ChatResponse`: response non-stream.
- `ConversationResponse`: item conversation.
- `StoredMessageResponse`: item message da luu.

Dependencies:
- `get_chat_service(settings)`: tao ChatService.
- `get_chat_history(settings)`: tao ChatHistory.

Endpoints:
- `chat(request, service, history)`: `POST /chat`; chat non-stream va luu messages.
- `chat_stream(request, service, history)`: `POST /chat/stream`; stream SSE token va luu ket qua.
- nested `event_stream()`: generator SSE ben trong `chat_stream`.
- `list_conversations(history)`: `GET /chat/conversations`.
- `list_messages(conversation_id, history)`: `GET /chat/conversations/{id}/messages`.

### `backend/app/api/agent.py`
Models/dependencies:
- `AgentRunRequest`: body chay agent.
- `get_agent_service(settings)`: tao AgentService.
- `get_chat_history(settings)`: tao ChatHistory.
- `get_rag_service(settings)`: tao RagService.
- `get_agent_run_store(settings)`: tao AgentRunStore.

Endpoints/helpers:
- `run_agent_stream(...)`: `POST /agent/run/stream`; stream lifecycle events, retrieval events, plan, final answer.
- nested `event_stream()`: generator SSE, la noi orchestration chinh cua agent endpoint.
- `_retrieve_for_agent(...)`: neu request can local docs, goi hybrid retrieval va compose context.
- `get_agent_run(run_id, store)`: `GET /agent/runs/{run_id}`; xem run/trace da luu.

### `backend/app/api/files.py`
Models:
- `IndexFolderRequest`: folder path, recursive, file_types.
- `ResolveFileRequest`: query filename/path.
- `ReadFileRequest`: direct read transient.

Dependencies:
- `get_indexing_service(settings)`: tao IndexingService.
- `get_catalog_service(settings)`: tao CatalogService.

Endpoints:
- `index_folder(request, service)`: `POST /files/index-folder`; deep index folder.
- `indexed_folders(service)`: `GET /files/indexed-folders`.
- `resolve_file(request, service)`: `POST /files/resolve`; tim file candidate.
- `read_file(request, service)`: `POST /files/read`; doc file truc tiep, khong can index.

### `backend/app/api/catalog.py`
Models:
- `ScanFolderRequest`: shallow scan folder.
- `CatalogSearchRequest`: search filename + document cards.

Endpoints:
- `scan_folder(request, service)`: `POST /catalog/scan-folder`.
- `search_catalog(request, service)`: `POST /catalog/search`.
- `collections(service)`: `GET /catalog/collections`.

### `backend/app/api/rag.py`
Models:
- `RagSearchRequest`: query + top_k.
- `IndexFileRequest`: index one file.
- `IndexSelectedFilesRequest`: index selected files into collection.
- `SearchInCollectionRequest`: search inside collection.
- `HybridSearchRequest`: LanceDB + FTS5 hybrid search.
- `VectorIndexDocumentRequest`: vector index one document.
- `VectorIndexAllRequest`: vector index all documents.

Dependencies:
- `get_rag_service(settings)`: tao RagService.
- `get_indexing_service(settings)`: tao IndexingService.
- `get_embedding_provider(settings)`: Ollama embedding provider.
- `get_lancedb_store(settings)`: LanceDB store.
- `get_vector_index_service(...)`: VectorIndexService.

Endpoints:
- `index_file(...)`: `POST /rag/index-file`.
- `index_selected_files(...)`: `POST /rag/index-selected-files`.
- `search(...)`: `POST /rag/search`; SQLite FTS5 chunk search.
- `search_debug(...)`: `POST /rag/search-debug`; tra ca FTS query.
- `search_in_collection(...)`: `POST /rag/search-in-collection`.
- `search_hybrid(...)`: `POST /rag/search-hybrid`; vector + FTS5 + context composer.
- `vector_index_document(...)`: `POST /rag/vector/index-document`.
- `vector_index_all(...)`: `POST /rag/vector/index-all`.
- `documents(...)`: `GET /rag/documents`.
- `document(document_id, service)`: `GET /rag/documents/{document_id}`.
- `document_chunks(...)`: `GET /rag/documents/{document_id}/chunks`.
- `document_tables(...)`: `GET /rag/documents/{document_id}/tables`.
- `document_figures(...)`: `GET /rag/documents/{document_id}/figures`.
- `delete_document(...)`: `DELETE /rag/documents/{document_id}`.

## 6. Backend services

### `backend/app/services/chat_history.py`
- `utc_now()`: timestamp UTC ISO.
- `ConversationSummary`: dataclass conversation list item.
- `StoredMessage`: dataclass message row.
- `ChatHistory.ensure_conversation(conversation_id, first_message)`: reuse conversation neu co, neu khong tao conversation moi.
- `ChatHistory.save_message(...)`: luu user/assistant message.
- `ChatHistory.list_conversations(limit)`: list conversation moi nhat.
- `ChatHistory.list_messages(conversation_id)`: list messages trong conversation.
- `ChatHistory._title_from_message(message)`: tao title ngan tu message dau.

### `backend/app/services/chat_service.py`
- `ChatService._messages(message, system_prompt)`: tao messages list cho Ollama.
- `ChatService.complete(...)`: non-stream chat qua OllamaClient.
- `ChatService.stream(...)`: stream token qua OllamaClient.

### `backend/app/services/agent_service.py`
- `AgentGraphResult`: ket qua agent graph gom plan/final/retrieved docs.
- `AgentService.run_graph(...)`: chay graph mot lan, tra ket qua day du.
- `AgentService.stream_final_answer(...)`: stream final answer bang Ollama dua tren prompt da compose.

### `backend/app/services/agent_run_store.py`
- `utc_now()`: timestamp UTC.
- `_json(data)`: JSON dumps stable.
- `AgentRunRecord`: dataclass run.
- `ToolCallRecord`: dataclass tool call.
- `AgentRunStore.create_run(...)`: tao run row.
- `AgentRunStore.update_plan(run_id, plan)`: luu plan.
- `AgentRunStore.complete_run(run_id, final_answer)`: mark completed.
- `AgentRunStore.fail_run(run_id, error_message)`: mark failed.
- `AgentRunStore.record_tool_call(...)`: luu tool call/retrieval call.
- `AgentRunStore.get_run(run_id)`: doc run + tool calls.
- `_decode_tool_call(row)`: decode JSON fields cua tool call.

### `backend/app/services/catalog_service.py`
Public:
- `utc_now()`: timestamp UTC.
- `CatalogService.resolve_file(...)`: nhan filename/query/path, tra candidate file.
- `CatalogService.read_file_direct(...)`: doc file transient, enforce approved folder.
- `CatalogService.scan_folder(...)`: shallow scan folder vao files + FTS, khong deep index.
- `CatalogService.search(...)`: search files + document cards bang FTS/RRF.
- `CatalogService.list_collections()`: list logical collections.

Private:
- `_can_read_path(path)`: kiem tra file co nam trong approved folder read/read_index.
- `_upsert_approved_folder(root, recursive, now)`: tao/cap nhat approved_folders.
- `_upsert_file_row(connection, path, file_type, now)`: tao/cap nhat files row + FTS.
- `_search_files(...)`: search FTS files.
- `_registered_file_candidates(...)`: lay candidates tu DB files.
- `_search_document_cards(...)`: search FTS document cards.
- `_merge_catalog_results(file_rows, card_rows, top_k)`: merge file/card result bang RRF.
- `_rrf(rank, k)`: reciprocal rank fusion score.
- `_decode_metadata(row)`: parse metadata_json.
- `_is_relative_to(path, root)`: safe path containment check.
- `_folder_candidates(...)`: scan filesystem candidates theo query.
- `_candidate_from_path(...)`: bien Path thanh candidate.
- `_candidate_from_row(...)`: bien DB row thanh candidate.
- `_dedupe_candidates(candidates)`: remove duplicate path.
- `_name_similarity(query, filename)`: fuzzy score don gian.

### `backend/app/services/indexing_service.py`
Dataclasses:
- `IndexedDocument`: document da index.
- `IndexFolderResult`: ket qua index folder.

Public:
- `utc_now()`: timestamp UTC.
- `IndexingService.index_file(...)`: index mot file, optional collection.
- `IndexingService.index_selected_files(...)`: index list file vao logical collection.
- `IndexingService.index_folder(...)`: deep index folder.
- `IndexingService.list_folders()`: list indexed_folders.
- `IndexingService.list_documents(limit)`: list documents.
- `IndexingService.get_document(document_id)`: doc document detail.
- `IndexingService.list_chunks(document_id)`: list chunks cua document.
- `IndexingService.delete_document(document_id)`: xoa document + child rows.
- `IndexingService.list_collections()`: list collections.
- `IndexingService.list_document_cards(limit)`: list document cards.

Private:
- `_upsert_folder(root, recursive, file_types, now)`: tao/cap nhat indexed_folders.
- `_index_file(...)`: ham chinh cua ingestion; parse, hash, skip/reindex, chunk, insert documents/chunks/tables/figures/cards/FTS.
- `_delete_document(connection, document_id)`: xoa document va tat ca child rows.
- `_needs_reindex_for_page_metadata(connection, document_id)`: reindex doc cu neu thieu page_number hoac artifact version.
- `_get_indexed_document_by_path(source_path)`: lay IndexedDocument theo path.
- `_upsert_approved_folder(...)`: cap quyen read_index cho folder.
- `_upsert_file(path, file_type, now)`: tao/cap nhat files row, hash file, sync FTS.
- `_upsert_collection(...)`: tao/cap nhat collection.
- `_link_document_to_collection(...)`: link document vao collection.
- `_upsert_document_card(...)`: tao/cap nhat card + FTS card.
- `_sync_file_fts(connection, file_id, path, file_type)`: sync fts_files.
- `_heading_path_for_chunk(file_type, chunk)`: lay heading hierarchy cho markdown chunk.
- `_hash_file(path)`: SHA256 file bytes.
- `_decode_metadata(row)`: parse metadata_json.
- `_decode_card(row)`: parse JSON fields cua document card.

### `backend/app/services/rag_service.py`
Public:
- `RagService.search(query, top_k, document_ids)`: SQLite FTS5 chunk search.
- `RagService.search_in_collection(collection_id, query, top_k)`: FTS search gioi han collection.
- `RagService.search_debug(query, top_k)`: tra fts_query + results.
- `RagService.search_hybrid(...)`: query embedding, search document cards, search chunks LanceDB + FTS5, merge/rerank.
- `RagService.get_document(document_id)`: doc document detail.
- `RagService.list_document_chunks(document_id)`: list chunks.
- `RagService.list_document_tables(document_id)`: list extracted table records.
- `RagService.list_document_figures(document_id)`: list extracted figure records.
- `RagService.delete_document(document_id)`: xoa document va rows lien quan.
- `RagService._collection_document_ids(collection_id)`: lay doc ids trong collection.

Private helpers:
- `_document_ids_from_vector_results(results, max_documents)`: lay unique document_id tu card vector results.
- `_filter_card_results_for_query(query, results)`: loc card theo topic constraints.
- `_query_topic_constraints(query)`: suy topic constraints nhu audio_visual/multimodal.
- `_constraint_text_match(tag, text)`: fallback check text.
- `_merge_hybrid_chunks(query, vector_chunks, fts_chunks, top_k)`: RRF merge + lexical rerank.
- `_rrf(rank, k)`: reciprocal rank fusion.
- `_decode_artifact(row)`: decode table/figure artifact row.
- `_lexical_rerank_boost(query, result)`: boost theo token overlap + dual channel.
- `_rerank_tokens(text)`: tokenize cho reranker.

### `backend/app/services/vector_index_service.py`
- `VectorIndexService.index_document(document_id)`: embed document card + chunks vao LanceDB.
- `VectorIndexService.index_all_documents(limit)`: prune stale vectors, index all current docs.
- `_document_ids(limit)`: list current document ids.
- `_document_card(document_id)`: fetch card row.
- `_document_chunks(document_id)`: fetch chunks.
- `_card_record(card)`: convert card -> VectorRecord.
- `_chunk_records(chunks)`: convert chunks -> VectorRecord list.
- `create_lancedb_vector_index_service(...)`: factory tao VectorIndexService.

## 7. Backend RAG modules

### `backend/app/rag/parsers.py`
Dataclasses:
- `ParsedPage`: page_number + text.
- `ParsedTable`: table metadata/caption/markdown.
- `ParsedFigure`: figure metadata/caption/image path placeholder.
- `ParsedDocument`: parsed text + pages + tables + figures.

Functions:
- `supported_file_type(path)`: check extension supported.
- `parse_text_file(path)`: compatibility helper tra text.
- `parse_document(path)`: router parse txt/md/pdf/docx.
- `_parse_pdf(path)`: pypdf extract text per page, detect table/figure captions.
- `_parse_docx(path)`: python-docx paragraphs.
- `_extract_markdown_tables(text)`: detect markdown pipe tables.
- `_looks_like_markdown_table_start(lines, index)`: table detector.
- `_split_markdown_row(line)`: split cells.
- `_previous_caption(lines, start_index, kind)`: tim caption ngay truoc table.
- `_caption_tables(text, page_number, start_index)`: detect `Table N...`.
- `_caption_figures(text, page_number, start_index)`: detect `Figure/Fig. N...`.
- `_caption_lines(text, kind)`: shared caption line extractor.
- `_caption_re(kind)`: regex caption.

### `backend/app/rag/chunking.py`
- `TextChunk`: chunk text + optional page_number.
- `chunk_text(text, chunk_size, overlap)`: split text thanh chunks co overlap.
- `chunk_parsed_document(parsed, chunk_size, overlap)`: chunk theo page neu parsed.pages co san; giu page_number.

### `backend/app/rag/retriever.py`
- `RetrievedChunk`: result FTS chunk.
- `build_fts_query(query)`: tokenize query thanh FTS5 OR query.
- `search_chunks(connection, query, limit, document_ids)`: search `rag_chunks_fts`, join chunks, tra RetrievedChunk.

### `backend/app/rag/context.py`
- `ComposedContext`: sources + context_text + stats.
- `compose_retrieval_context(results, max_sources, ...)`: dedupe chunks, cap chunks/document, tao source blocks cho prompt.
- `_format_source_block(source_id, result, content)`: tao text block co file/path/page/retrieval/content.
- `_rank_text(result)`: format vector/fts ranks.
- `_clean_content(content)`: normalize whitespace.
- `_truncate(content, max_chars)`: cat source content theo boundary.

### `backend/app/rag/embeddings.py`
- `EmbeddingError`: exception embedding.
- `EmbeddingProvider`: Protocol chung.
- `OllamaEmbeddingProvider.embed_texts(texts)`: embed batch qua Ollama.
- `OllamaEmbeddingProvider.embed_query(text)`: embed query.
- `OllamaEmbeddingProvider._embed_batch(texts)`: goi endpoint embed moi.
- `OllamaEmbeddingProvider._embed_legacy(text)`: fallback endpoint legacy.
- `HashEmbeddingProvider.embed_texts(texts)`: deterministic fake embeddings cho tests.
- `HashEmbeddingProvider.embed_query(text)`: fake query embedding.
- `_normalize_embedding(embedding)`: cast list -> floats.

## 8. Backend retrieval store

### `backend/app/retrieval_store/base.py`
- `VectorRecord`: id/text/vector/metadata.
- `RetrievalFilter`: optional document_ids/file_ids/folder_path.
- `RetrievalResult`: vector search result.
- `RetrievalStore`: Protocol interface: add/search/delete records.

### `backend/app/retrieval_store/lancedb_store.py`
- `LanceDBUnavailable`: LanceDB optional dependency missing.
- `LanceDBRetrievalStore.__init__(db_dir)`: connect LanceDB.
- `add_document_cards(records)`: add to `document_cards`.
- `add_text_chunks(records)`: add to `text_chunks`.
- `add_table_chunks(records)`: reserved for table vectors.
- `add_figure_chunks(records)`: reserved for figure vectors.
- `add_memory_chunks(records)`: reserved for memory vectors.
- `search_document_cards(...)`: vector search cards.
- `search_text_chunks(...)`: vector search chunks.
- `delete_document(document_id)`: delete all vectors for one doc.
- `prune_documents(document_ids)`: delete stale vector rows not in current SQLite docs.
- `_add(table_name, records)`: create/open table, delete existing ids, add rows.
- `_search(table_name, query_embedding, filters, top_k, source)`: LanceDB search.
- `_open_table(table_name)`: safe open table if exists.
- `_delete_existing_ids(table, ids)`: delete duplicate ids before add.
- `_record_to_row(record)`: VectorRecord -> LanceDB row.
- `_where(filters)`: build LanceDB where clause.
- `_in(column, values)`: SQL-ish IN clause.
- `_quote(value)`: quote string safely.
- `_row_metadata(row)`: decode metadata_json and defaults.
- `_table_names(db)`: compatibility helper.

## 9. Backend agent modules

### `backend/app/agents/state.py`
- `AgentState`: TypedDict state passed through agent graph: task, mode, plan, final_prompt, retrieved_docs, etc.

### `backend/app/agents/graph.py`
- `build_agent_graph(client, temperature)`: tao graph runnable voi router/planner/final_prompt nodes.
- nested `router_node(state)`: route mac dinh `file_qa` neu co retrieved docs, nguoc lai `general`.
- nested `planner_node(state)`: deterministic plan khi co retrieved docs; neu khong goi LLM planner.
- nested `final_prompt_node(state)`: build prompt cuoi, bat buoc dung sources neu co.
- `_normalize_plan(raw)`: strip bullets, limit step count.
- `_planner_input(state)`: tao planner prompt.
- `_format_retrieved_docs(docs, max_chars)`: format retrieved docs cho planner/final.

## 10. Backend catalog/document cards

### `backend/app/catalog/document_card.py`
- `STOPWORDS`: token bo qua khi keyword extract.
- `TAG_RULES`: regex tag rules: rag, lancedb, sqlite, agent, visual_rag, SER, etc.
- `DocumentCardDraft`: title/doc_type/language/summary/tags/keywords.
- `build_document_card(path, text)`: tao card tu document text.
- `_normalize_text(text)`: collapse whitespace.
- `_guess_title(path, normalized)`: doan title tu line dau hoac filename.
- `_summary(normalized, max_chars)`: tom tat ngan bang prefix.
- `_keywords(normalized, limit)`: keyword frequency.
- `_topic_tags(filename, normalized)`: gan topic tags theo regex.
- `_project_tags(filename, normalized)`: gan local_ai_agent/ser_research.
- `_doc_type(filename, normalized, tags)`: project_plan/technical_plan/research_paper/...
- `_language(normalized)`: detect vi/en don gian.

## 11. Frontend entry/layout

### `apps/desktop/src/main.tsx`
Mount React app vao DOM.

### `apps/desktop/src/App.tsx`
- `AppRoute`: route union `chat | agent | files | settings`.
- `App()`: giu state route/model, render Sidebar/Topbar va page tuong ung.

### `apps/desktop/src/styles/globals.css`
Toan bo style UI: shell layout, chat, files page, buttons, cards, source/result lists.

### `apps/desktop/src/components/layout/Sidebar.tsx`
- `Sidebar({ activeRoute, onRouteChange })`: nav ben trai, doi route.

### `apps/desktop/src/components/layout/Topbar.tsx`
- `Topbar({ route, model, onModelChange })`: title current route + input model.

## 12. Frontend chat/agent components

### `apps/desktop/src/components/chat/Composer.tsx`
- `Composer({ value, disabled, onChange, onSubmit })`: textarea + send button; submit chat.

### `apps/desktop/src/components/chat/ChatWindow.tsx`
- `ChatWindow({ messages })`: render list MessageBubble.

### `apps/desktop/src/components/chat/MessageBubble.tsx`
- `MessageBubble({ message })`: render one user/assistant message.

### `apps/desktop/src/components/agent/AgentTracePanel.tsx`
- `AgentTracePanel({ events, status })`: panel ben phai hien SSE events/status.

## 13. Frontend pages

### `apps/desktop/src/routes/ChatPage.tsx`
- `ChatPage({ model })`: chat screen, load health/latest conversation, stream chat.
- nested `loadInitialState()`: check health, load latest conversation/messages.
- `sendMessage()`: optimistic add user/assistant message, call `streamChat`, append deltas.

### `apps/desktop/src/routes/AgentPage.tsx`
- `AgentPage({ model })`: agent/research screen.
- `runAgent()`: call `streamAgent`, update run id, plan, sources, answer, trace events.

### `apps/desktop/src/routes/FilesPage.tsx`
Main RAG/catalog UI.
- `FilesPage()`: stateful file/RAG dashboard.
- `refresh()`: load folders/documents/collections.
- `handleIndex()`: deep index selected folder.
- `handleScan()`: shallow catalog scan.
- `handleSearch()`: SQLite RAG search.
- `handleCatalogSearch()`: catalog/card search.
- `handleResolve()`: resolve filename/path query.
- `handleRead(sourcePath)`: direct transient read.
- `handleIndexSelected()`: index listed files into collection.
- `handleVectorIndexAll()`: build LanceDB vectors.
- `handleHybridSearch()`: run hybrid retrieval and display context sources.

### `apps/desktop/src/routes/SettingsPage.tsx`
- `SettingsPage()`: placeholder/settings surface.

## 14. Frontend API/SSE/types

### `apps/desktop/src/lib/api.ts`
- `API_BASE_URL`: backend URL, env override `VITE_BACKEND_URL`, default `127.0.0.1:7777`.
- `readJson(path)`: GET helper.
- `writeJson(path, body)`: POST JSON helper.
- `getHealth()`: GET `/health`.
- `listConversations()`: GET chat conversations.
- `listMessages(conversationId)`: GET messages.
- `indexFolder(input)`: POST `/files/index-folder`.
- `scanCatalogFolder(input)`: POST `/catalog/scan-folder`.
- `listIndexedFolders()`: GET `/files/indexed-folders`.
- `listDocuments()`: GET `/rag/documents`.
- `listCollections()`: GET `/catalog/collections`.
- `searchRag(query, topK)`: POST `/rag/search`.
- `searchCatalog(query, folderPath, topK)`: POST `/catalog/search`.
- `resolveFile(input)`: POST `/files/resolve`.
- `readFileDirect(input)`: POST `/files/read`.
- `indexSelectedFiles(input)`: POST `/rag/index-selected-files`.
- `vectorIndexAll(limit)`: POST `/rag/vector/index-all`.
- `searchHybrid(input)`: POST `/rag/search-hybrid`.

### `apps/desktop/src/lib/sse.ts`
- `StreamChatInput`: input type for streaming chat.
- `streamChat(input)`: POST `/chat/stream`, yield ChatStreamEvent.
- `StreamAgentInput`: input type for streaming agent.
- `streamAgent(input)`: POST `/agent/run/stream`, yield AgentStreamEvent.
- `readEventStream(body)`: low-level SSE reader.
- `parseEvent(raw)`: parse one SSE event block.

### `apps/desktop/src/lib/types.ts`
TypeScript contracts mirroring backend:
- Chat/conversation: `ChatRole`, `ChatMessage`, `Conversation`, `StoredMessage`.
- Health: `HealthResponse`.
- Index/catalog: `IndexedFolder`, `IndexedDocument`, `CatalogCollection`, `CatalogFile`, `ScanFolderResult`, `CatalogSearchResult`, `CatalogSearchResponse`.
- Retrieval: `RetrievedDocument`, `IndexFolderResult`, `HybridSearchResponse`.
- Direct file: `FileCandidate`, `ResolveFileResponse`, `DirectReadResponse`.
- Vector/index selected: `IndexSelectedFilesResult`, `VectorIndexAllResult`.
- SSE: `ChatStreamEvent`, `AgentStreamEvent`.

## 15. Tauri/Rust shell

### `apps/desktop/src-tauri/src/main.rs`
Rust entrypoint Tauri. Hien tai chu yeu boot shell webview.

### `apps/desktop/src-tauri/build.rs`
Tauri build script.

### `apps/desktop/src-tauri/tauri.conf.json`
Tauri app config: bundle, window, security.

### `apps/desktop/src-tauri/capabilities/default.json`
Tauri permissions/capabilities.

## 16. Scripts

### `scripts/start_backend.sh`
Start FastAPI backend trong conda env.

### `scripts/dev_mac.sh`
Script dev cho Mac; dung de chay local workflow nhanh.

### `scripts/check_ollama.sh`
Check Ollama health/model availability.

## 17. Tests

### `backend/tests/test_rag.py`
- `test_chunk_text_splits_long_text`: chunker split text dai.
- `test_chunk_parsed_document_preserves_page_numbers`: page_number duoc giu.
- `test_markdown_table_and_figure_caption_are_indexed`: table/figure metadata extraction.
- `test_index_folder_and_search_markdown`: index folder + FTS search + document detail.
- `test_parse_docx`: docx parser works.

### `backend/tests/test_rag_quality.py`
- `test_document_card_tag_matching_avoids_short_substring_false_positives`: `rag` tag khong false positive trong `average`.
- `test_context_composer_deduplicates_and_caps_chunks_per_document`: context composer cap chunk/document.
- `test_lexical_reranker_boosts_query_matching_dual_channel_result`: reranker boost dung.

### `backend/tests/test_vector_retrieval.py`
- `test_lancedb_vector_index_and_hybrid_search`: LanceDB index + hybrid retrieval.
- `test_vector_index_all_skips_legacy_documents_without_cards`: skip legacy doc khong co card.
- `test_vector_index_all_prunes_stale_document_vectors`: prune stale vectors sau reindex.

### `backend/tests/test_catalog.py`
- `test_scan_folder_catalogs_files_without_deep_index`: shallow scan chi catalog.
- `test_read_file_direct_requires_approved_folder`: direct read bi gioi han approved folder.
- `test_index_selected_files_creates_collection_card_and_collection_search`: selected files + collection search.

### `backend/tests/test_chat_history.py`
- `test_chat_history_creates_conversation_and_stores_messages`: chat history CRUD co ban.

### `backend/tests/test_chat_service.py`
- `FakeClient`: fake Ollama client.
- `test_complete_uses_default_model_when_request_model_is_missing`: default model.
- `test_stream_uses_requested_model`: stream dung requested model.

### `backend/tests/test_agent_graph.py`
- `test_normalize_plan_strips_bullets_and_limits_steps`: normalize plan.
- `test_file_qa_uses_deterministic_source_grounded_plan`: file QA khong goi planner LLM khi da co docs.

### `backend/tests/test_agent_run_store.py`
- `test_agent_run_store_records_run_and_tool_call`: luu run/tool call.

### `backend/tests/test_events.py`
- `test_sse_event_formats_named_event_with_json_payload`: format SSE.

## 18. PDF/project plan files

### `pdf/PROJECT_PLAN_mac_ai_agent.md`
Plan goc: Mac-first local AI agent app.

### `pdf/PROJECT_PLAN_ADDENDUM_RAG_AGENT_MEMORY_VISUAL.md`
Addendum RAG/agent/memory/visual. Mo ta deep RAG, memory, visual docs, tables, rerank, quality eval.

### `pdf/PROJECT_PLAN_ADDENDUM_STORAGE_CATALOG_LANCEDB.md`
Addendum storage/catalog: SQLite + FTS5 + LanceDB, selected files, catalog, permission model.

### `pdf/*.pdf`
Seed research papers cho SER/multimodal/audio-visual retrieval tests.

## 19. Nen sua file nao khi lam task nao?

### Sua UI chat
```text
apps/desktop/src/routes/ChatPage.tsx
apps/desktop/src/components/chat/*
apps/desktop/src/lib/sse.ts
backend/app/api/chat.py
backend/app/services/chat_service.py
backend/app/services/chat_history.py
```

### Sua agent/research mode
```text
apps/desktop/src/routes/AgentPage.tsx
backend/app/api/agent.py
backend/app/services/agent_service.py
backend/app/agents/graph.py
backend/app/services/agent_run_store.py
```

### Sua file/RAG/indexing
```text
apps/desktop/src/routes/FilesPage.tsx
backend/app/api/files.py
backend/app/api/rag.py
backend/app/services/indexing_service.py
backend/app/services/rag_service.py
backend/app/rag/*
```

### Sua catalog/direct read
```text
backend/app/api/catalog.py
backend/app/api/files.py
backend/app/services/catalog_service.py
backend/app/db/sqlite.py
```

### Sua vector/LanceDB
```text
backend/app/services/vector_index_service.py
backend/app/retrieval_store/base.py
backend/app/retrieval_store/lancedb_store.py
backend/app/rag/embeddings.py
```

### Sua schema
```text
backend/app/db/sqlite.py
backend/app/services/* neu insert/select schema moi
backend/tests/* de cover migration/behavior
```

## 20. Trang thai hien tai

Da co:
- chat streaming voi Ollama
- conversation/message persistence
- agent stream co plan/sources/final answer
- SQLite catalog + FTS5
- direct file read co permission
- index folder/file/selected files
- document cards
- LanceDB vector index document_cards/text_chunks
- hybrid retrieval
- page-aware chunks
- lightweight reranker
- table/figure metadata extraction v0.1

Chua co:
- visual page screenshot cache
- figure crop/bbox extraction
- VLM visual QA
- model reranker that su nhu Qwen3-Reranker
- table calculation tools
- memory UI/long-term memory production flow
