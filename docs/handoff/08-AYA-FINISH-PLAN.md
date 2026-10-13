# 08 — Kế hoạch hoàn tất Aya (chốt 2026-10-13, Opus quyết, Haiku thực hiện)

Đọc file này trước khi làm. Mọi quyết định dưới đây **đã chốt**: không mở lại,
không hỏi lại user, không đổi hướng. Gặp điều không rõ thì chọn phương án đơn
giản hơn và ghi lại trong `00-STATUS.md`.

## Mục tiêu của user (nguyên văn yêu cầu)

Aya phải:
1. **Trả lời tốt mọi thứ từ paper.**
2. **Tự chèn hình, bảng, công thức** khi cần.
3. **Trả lời được câu ngoài paper**, rồi **quay lại đúng bài đó** khi được hỏi lại.
4. **Tự hiểu câu mơ hồ** ("cái đó", "cái này", "cái kia", "nó", "bảng đó",
   "hình kia") là đang nói về thứ gì trong bài.

## Trạng thái khi bắt đầu (đã đo, đã commit tới `ccec3c43`)

- Backend 1.376 test pass. Context 10 source / 16k ký tự.
- public-36 recall 0,653; local-60 0,97/0,90/1,00; dev-20 Sol **20/20**.
- `backend/app/services/answer_verifier.py`: kiểm "số có trong tài liệu" +
  bước sửa bằng model; hiện chỉ chạy **shadow** (`answer_number_shadow`).
- Validator cũ (`evidence_validator.py` + `numeric_grounding.py`) **vẫn chặn**
  câu trả lời có số: buffer cả câu, retry, có thể fallback "chưa đủ evidence".
- Hình: có (`MessageBubble.tsx` → `selectFigureAttachments`). Bảng: có
  (`_direct_canonical_table_answer`). **Công thức: không có** (parser không
  trích, `MarkdownBlock.tsx` không render LaTeX, không có KaTeX).
- Off-topic → quay lại: dev02-t3/t4 pass. Câu mơ hồ: `looks_like_followup`
  nhận ra "cái đó/cái này" là follow-up, nhưng **chưa gắn với hình/bảng/công
  thức vừa hiện** ở lượt trước.

## Quyết định đã chốt

- **D1 — Bỏ chặn, chuyển sang kiểm-rồi-sửa.** Câu trả lời luôn stream như
  chat thường, không buffer. Sau khi stream xong, chạy cả validator cũ lẫn
  `find_unsourced_numbers`, cả hai chỉ để *phát hiện*. Nếu có câu bị flag →
  gọi Sol **đúng 1 lần** (`build_verify_prompt` → `parse_corrections` →
  `apply_corrections`): giữ / sửa / xóa **đúng các câu bị flag**. Nếu nội dung
  đổi thì gửi `message.reset` kèm bản đã sửa. **Không** retry sinh lại cả câu,
  **không** fallback cả câu trả lời. Không bao giờ thêm số không có trong
  source (luật "không bịa số" vẫn giữ nhờ bước sửa này).
- **D2 — Bộ đo độc lập lưu vào repo ngay.** Bộ probe mù cũ đã mất vì nằm ở
  scratchpad. Mọi bộ đo mới phải nằm trong `evals/` và được commit ngay khi tạo.
- **D3 — Một bộ nghiệm thu chung cho 4 mục tiêu**:
  `evals/aya_acceptance_v1.jsonl`. Đây là thước đo "xong", thay cho dev-20.
- **D4 — Công thức:** render bằng KaTeX ở frontend; prompt cho phép viết lại
  công thức thành LaTeX **từ chính text trong evidence** và kèm nguồn; bật
  `do_formula_enrichment` của Docling sau flag, chỉ giữ nếu đo thấy tốt hơn.
- **D5 — Câu mơ hồ:** lưu "artifact vừa hiện" (hình / bảng / công thức + id +
  tài liệu) vào working state của hội thoại; câu mơ hồ ưu tiên gắn vào đó
  trước, rồi mới tới topic của paper.
- **D6 — Không làm:** không tune thêm regex validator; không đổi embedding,
  index production hay model; chỉ dùng `cx/gpt-5.6-sol` (trả lời) và
  `cx/gpt-5.6-terra` (judge).

## Các bước (làm đúng thứ tự, mỗi bước commit riêng)

### Bước A — Tạo bộ đo trước, chạy baseline (không sửa code)

1. Dùng một **agent độc lập** (Agent tool, `model: "haiku"`), dặn rõ: không
   đọc `backend/app/`, `backend/tests/`. Agent đọc nội dung các paper trong
   clone `data/retrieval_eval/internal/runtime/e2e-foundation-20261005`
   (bảng `chunks`, `document_tables`, `document_figures`) và viết
   `evals/aya_acceptance_v1.jsonl`: **12 hội thoại, mỗi hội thoại 4–6 lượt**,
   cùng format với `data/retrieval_eval/conversational-dev-v1.jsonl`. Phủ đủ:
   - **G1 paper QA:** task, kiến trúc, dataset, kết quả có số, so sánh 2 paper.
   - **G2 artifact:** "cho xem hình kiến trúc", "đưa bảng kết quả", "công
     thức loss là gì" (chọn paper có công thức trong text).
   - **G3 off-topic → quay lại:** 1–2 câu ngoài lề (thời tiết, nấu ăn, code
     Python) rồi "quay lại bài lúc nãy", "thế còn kết quả của nó?".
   - **G4 mơ hồ:** "cái đó là gì", "giải thích cái bảng này", "hình kia nói gì",
     "nó tốt hơn cái kia ở đâu", ngay sau một lượt đã hiện hình/bảng.
   - Một nửa câu tiếng Việt, một nửa tiếng Anh. Mỗi lượt ghi `expected`:
     `focus_document` (tên file), `must_attach` (`figure` / `table` /
     `formula` / không có), `must_mention` (2–4 ý chính rút từ tài liệu),
     `off_topic` (true/false).
2. Cũng bằng một agent độc lập, viết lại bộ probe số:
   `evals/numeric_probes_v3.jsonl` (khoảng 120 cặp đúng/sai, ngoài lĩnh vực
   SER, 40% tiếng Việt, đúng format cũ: `scenario`, `answer`, `expected`,
   `evidence`).
3. Viết `scripts/evaluate_aya_acceptance.py`: chạy hội thoại qua HTTP/SSE
   (dùng lại `scripts/evaluate_agent_conversations.py` — `run_turn`, checkpoint),
   rồi chấm mỗi lượt:
   - **cấu trúc (không cần LLM):** đúng `focus_document`; có hình / bảng /
     công thức khi `must_attach` yêu cầu (hình: `figure_ids` hoặc sự kiện ảnh;
     bảng: dòng Markdown `|---|`; công thức: `$...$` hoặc `$$...$$`); lượt
     off-topic **không** gọi `search_local_docs`; không có fallback "chưa đủ
     evidence".
   - **nội dung:** judge `cx/gpt-5.6-terra`, một lần gọi mỗi lượt, JSON
     `{"covers": [true/false theo từng must_mention], "fabricated_numbers": [...],
     "natural": 1-5}`.
   - Báo cáo theo nhóm G1–G4, kèm từng lượt fail.
4. Commit cả 3 file. Chạy **baseline** trên code hiện tại (backend cô lập
   :7790, lệnh trong `06-OPS.md`). Ghi kết quả theo nhóm vào `00-STATUS.md`.
   Từ đây, mọi sửa chỉ nhắm vào các lượt fail của baseline.

### Bước B — D1: bỏ chặn, kiểm-rồi-sửa

- File: `backend/app/api/agent.py`, nhánh `if validate_quantitative_claims and
  retrieved_docs:` (khoảng dòng 2810–3010) và nhánh `validated_blocks`
  (khoảng dòng 2688–2790).
- Thay bằng: stream từng chunk ra ngay (như nhánh chat thường) và gom
  `answer`. Sau khi xong: `legacy = validate_answer_claims(...)`,
  `sourcing = find_unsourced_numbers(answer, validation_documents)`. Câu bị
  flag = câu chứa claim trong `legacy.unsupported_claims` ∪ `sourcing.flagged`.
  Nếu có → `service.stream_final_answer` hoặc `client.chat` với
  `build_verify_prompt(...)`, `temperature=0`, timeout 30 s → `apply_corrections`.
  Nếu kết quả khác `answer` → `message.reset` rồi stream bản đã sửa. Ghi tool
  call `answer_verification` (flagged, corrections, changed).
- Lỗi hay timeout ở bước verify: giữ nguyên câu trả lời, ghi lỗi; không fallback.
- Giữ nguyên: direct table, visual guard, paper-section streaming.
- Sửa các test trong `test_answer_claim_guard.py` đang khẳng định hành vi
  retry/fallback: đổi kỳ vọng sang "sửa đúng câu bị flag". Không xóa test chặn số
  bịa; đổi chúng thành "số bịa bị xóa hoặc sửa sau verify".
- **Gate B:** suite xanh; dev-20 20/20 và **0 fallback**; median chữ đầu ≤ 3 s;
  trên `numeric_probes_v3`, tỷ lệ bỏ lọt số sai không cao hơn validator cũ đo
  trên cùng bộ (đo bằng `apply_corrections` với một model giả trả "remove"
  cho câu bị flag).

### Bước C — D4: công thức, và sửa artifact theo baseline

1. Frontend: thêm `katex` vào `apps/desktop/package.json`; trong
   `MarkdownBlock.tsx`, render `$$...$$` (block) và `$...$` (inline) bằng
   `katex.renderToString` (`throwOnError: false`). Không render bên trong
   code block. `npm run build` phải pass.
2. Prompt trả lời (tìm system prompt trong `backend/app/services/agent_service.py`
   / `agents/graph.py`): thêm luật "Khi được hỏi về công thức, viết lại công
   thức bằng LaTeX `$$...$$` dựa đúng trên text trong source và ghi nguồn
   [Sx]; nếu source không có công thức thì nói rõ, không tự suy ra."
3. Parser: trong `backend/app/rag/parsers.py` (chỗ `PdfPipelineOptions()`), đặt
   `pipeline_options.do_formula_enrichment = settings.formula_enrichment_enabled`
   (flag mới, mặc định tắt). Re-index **trên clone** 3 paper có công thức,
   chạy lại các lượt công thức trong G2. Bật mặc định chỉ khi G2-formula tăng
   và thời gian ingest không tăng quá 2 lần.
4. Sửa các lượt G2 fail của baseline: nếu thiếu hình → xem
   `_visual_attachment_candidates` và `selectFigureAttachments`; nếu thiếu
   bảng → xem `_direct_canonical_table_answer` / `_merge_focused_table_inventory`.
   Sửa theo cơ chế chung, kèm test; không hardcode tên paper.
- **Gate C:** nhóm G2 ≥ 90% lượt đạt `must_attach`; build desktop pass.

### Bước D — D5: câu mơ hồ, và quay lại paper

1. Working state: tìm chỗ lưu L1 (`ConversationWorkingState`,
   `backend/app/agents/state.py` và chỗ cập nhật trong `agent.py`). Thêm
   `last_artifacts: list[{kind, id, document_id, label}]`, cập nhật sau mỗi
   lượt có hình / bảng / công thức được hiện.
2. Query rewrite (`query_rewrite_service.py`): khi câu có từ chỉ định ("cái
   đó/này/kia", "nó", "bảng/hình/công thức + đó/này/kia", "this/that one",
   "it") và không nêu tên tài liệu → standalone query = câu gốc + nhãn
   artifact gần nhất + tên tài liệu đang focus; `focus_document_ids` = tài liệu
   của artifact đó. Nếu có cả hình và bảng ở lượt trước: "bảng" → bảng,
   "hình" → hình, còn lại → artifact hiện sau cùng.
3. Quay lại paper: lượt off-topic **không** xóa L1 focus (kiểm
   `_looks_like_topic_switch` / sticky focus). Câu "quay lại bài lúc nãy / bài
   trước / previous paper" → focus = paper gần nhất trước đoạn off-topic.
4. Test đơn vị cho từng quy tắc, dùng tên paper giả.
- **Gate D:** G3 ≥ 90% và G4 ≥ 85% lượt đúng `focus_document` và đúng artifact.

### Bước E — Chạy nghiệm thu một lần, chốt

- Chạy lại `evaluate_aya_acceptance.py` một lần trên code cuối. **Đạt khi:**
  G1 ≥ 85% đủ ý, G2 ≥ 90%, G3 ≥ 90%, G4 ≥ 85%, **0 số bịa**, 0 fallback,
  median `natural` ≥ 4.
- Không đạt: chỉ sửa đúng các lượt fail, chạy lại **tối đa 1 lần**, rồi ghi
  kết quả thật vào `00-STATUS.md` dù đạt hay không.
- Cập nhật `00-STATUS.md`, `04-DONE.md`, `05-TODO-AND-ISSUES.md`; chạy
  `~/.local/share/uv/tools/graphifyy/bin/graphify update .`.

## Lệnh hay dùng

- Test: `cd backend && .venv/bin/python -m pytest -q`
- Backend cô lập :7790 và frontend: xem `06-OPS.md`, mục "Current runtime".
- 9router :20128 round-robin 2 tài khoản; client đã tự retry 429 cùng model.
- Commit: kết thúc message bằng
  `Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>`.
