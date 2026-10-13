# Numeric grounding — tiếp nối Claude, 2026-10-10

## Latest continuation: canonical provenance and incomplete-answer detection

- `MetricValueClaim` optionally retains canonical `document_id`. Explicit pure
  Paper columns resolve shortened catalog identities uniquely; ambiguous and
  unknown aliases fail closed. Model columns remain independently constrained.
  Candidate narrowing, unlabeled inference and condition binding preserve that
  provenance even when the filename yields no lexical acronym. Legacy dicts
  omit the new field unless set.
- Declared Model/dataset composite cells accept spaced slash or dash separators.
  A bare Model column does not split slash-connected model names. Generic
  correct/wrong model, dataset and split tests cover both separators.
- Trace compaction now activates when the bounder clips a draft string before
  its hard-cap fallback, not only after total-cap failure. Successful repeated
  proof records are omitted first; secret redaction and caps remain mandatory.
- Full frozen Sol dev-20: **19/20**, final comparison answer fallback after two
  attempts (**45.241 s**), no tool/model/transport fallback. Median total/visible
  **9.397 / 3.075 s**; another turn retried. Full drafts retained unclipped.
- Diagnosis initially suspected code diagrams; removal of diagrams did not
  change validation. Root cause: paper identity from sentence one was applied
  to measurements in another sentence of the same paragraph. Claim binding now
  uses sentence-local canonical identity or a uniquely declared source model,
  never values; ordinary structured rows cannot override an explicit section.
  Correct/wrong synthetic controls, including the existing incidental-title
  scope rejection, pass. This is bounded binding, not general entailment.
- Sanitization previously only matched unsupported sentence text at the start
  of a physical line, leaving unsafe later sentences in labelled paragraphs.
  Now drops the affected prose paragraph conservatively, then revalidates.
  Removed reverse-prefix cell matching that cleared a correct table's model
  name merely because an unsafe sentence elsewhere began with that name.
  Both complete drafts of the failed full-run comparison pass offline after
  conservative removal; correct table content remains source-supported.
- Final-source affected replay **4/4 internal rubric**, no retries or answer/
  model/tool/transport fallback, median total/visible **11.515 / 2.994 s**.
  **Manual review FAILS full task completeness**: output selected
  `initial_sanitized`, removed later numeric rows and appended an insufficiency
  section. Its trace shows dotted scope/metric strings treated as model tokens,
  plus inherited condition/owner ambiguity. Do not report this as 4/4 quality.
- New completeness diagnostic v2 flags partial answers with retrieved artifacts
  and answer fallbacks even with clean Markdown/validator success. This is only
  a review trigger: artifact presence does not establish task answerability.
  Old structural-only reports stay preserved; v2 flags **1/4** affected and
  **1/20** full-run answers. No independent semantic scoring yet.
- Visible 172 probes unchanged **77/85 wrong rejected, 79/87 correct accepted,
  71/85 paired pass**, **8 false accepts / 8 false rejects**. No facts benefit,
  no public retrieval gain claim. Production and flags unchanged.
- Final backend **1,355 passed**, 7 existing warnings; desktop build/diff pass.
  Graphify **5,154 nodes / 15,919 edges / 185 communities**, vault regenerated
  AST-only, manual note preserved. Prior graph warnings retained. Owned isolated
  :7790 and eval process stopped; user 9router unchanged.

Reports (all under `data/retrieval_eval/internal/`):

- `numeric-provenance-sol-dev20-20261010.json` (failed full 19/20; retain).
- `numeric-sentence-sol-affected-20261010.json` (4/4 rubric; partial, not quality).
- `numeric-sentence-sol-completeness-v2-20261010.json` (1 answer needs review).
- `numeric-provenance-sol-completeness-v2-20261010.json` (1 full-run fallback).
- `numeric-sentence-regression-20261010.json` (visible numeric regression).

Next: repair condition-qualified metric cells from grammar/header schema with
wrong-model/wrong-split controls, without borrowing global values. Add task-facet
completeness independent of the system validator. Separately instrument public
candidate/context selection drops and audit the fixed zero-recall frontier;
its **0.40278** helper recall remains, broad arbitrary-format/visual gates open.

## Continuation: explicit prose ownership (2026-10-10)

- In source blocks with multiple explicit named subjects, each quantity gains
  the nearest preceding subject's owner label. A bounded subject/verb grammar
  reads names from the source; no County/paper/model dictionary. Counted noun
  heads add a separate measure axis: a hospitalization count cannot support a
  cases claim. Respectively pairing now applies to these prose owners as well.
- This is not general NLP/entailment: unsupported grammatical constructions
  retain existing checks. Noun heads do not prove arbitrary subtype, population
  or temporal qualifiers. Years, negated qualitative predicates, inherited
  comparison attributes and located arithmetic remain open.
- Added paired positive/negative tests with renamed entities, different noun
  descriptions and Vietnamese. Visible replay improves **74/85 → 77/85 wrong
  rejected**, preserves **79/87 correct accepted**, **71/85 pairs** both pass.
  Gains: `s15_borrow_w`, `s15_cond_w`, `s15_resp_w`; zero regressions in this
  visible suite. Still **8 false accepts / 8 false rejects**, not a blind gate.
- Final-source backend **1,334 passed**, 7 warnings; desktop build/diff check
  pass. Graphify **5,127 nodes / 15,844 edges / 172 communities**, Obsidian
  **5,233 notes**, manual `path_46.md` preserved. AST only, no graph model calls;
  existing skill-version/empty-JSON/stale-label warnings retained.
- Sol dev05 HTTP/SSE replay: **3/4 pass**, final comparison rejected after two
  attempts and emitted insufficiency answer fallback (**42.500 s** total,
  **42.329 s** first visible). No transport/tool/model fallback; do not conflate
  this with zero answer fallback. Group median total **11.059 s**, visible
  **4.032 s** includes direct canonical-table execution and failed last turn.
- Surviving tool-call attempt validations show metric/owner and caveat binding
  failures. All seven distinct rejected snippets produce identical offline
  validation with the new prose subject helper disabled; none of the five
  retrieved source blocks activates the new multi-subject binding. This does
  not replace replay of complete drafts, which were lost when debug payload
  exceeded the 64-KB cap. No speculative production fix from partial traces.
- Facts/dual-query stay off. No production DB/index/embedding/LightRAG changes.
  Owned isolated backend stopped; user 9router left running.

Reports:

- `data/retrieval_eval/internal/numeric-prose-regression-20261010.json`
- `data/retrieval_eval/internal/numeric-prose-sol-dev05-20261010.json` (**failed**, retained)

Next: preserve bounded full-draft diagnostics for the failed comparison, then
repair general local-owner/metric scoping with adversarial paired controls.
Do not rerun until a random passing answer replaces the recorded failure.

## Continuation: bounded trace and canonical local ownership — 2026-10-10

- Oversized traces now omit repeated successful proofs before giving up on
  the hard cap. Failed validations, full drafts and omission counts survive;
  redaction and byte bounds remain mandatory, in-memory validation unchanged.
  A live failed-comparison trace was retained at ~40 KB instead of a marker.
- Typed paper/model table cells may override an enclosing paper only when the
  row owner resolves to one canonical source document. Ordinary model columns,
  ambiguous owners and explicitly wrong paper labels retain strict scope.
- Occurrence-based metric rescue now counts repeated claims rather than using
  a set; identical correct measurements no longer hide a masked occurrence.
  Thresholds preceding packed metrics remain conditions, not unlabeled metrics.
- OCR-spaced decimal dots keep literal source offsets. Numbered document
  headings update canonical paper scope even with a copy ordinal/year, without
  exempting measured quantities printed alongside them. Generic paired tests
  reject wrong owner, dataset, split, threshold and cross-document borrowing.
- Captured correction replay no longer rejects its correct threshold. Caveat
  prose containing unlabeled values still fails ambiguity/ownership checks;
  do not bypass these checks based only on a global matching number.
- Visible 172-case suite is unchanged: **77/85 wrong rejected, 79/87 correct
  accepted, 71/85 pairs**; **8 false accepts / 8 false rejects remain**.
  Final backend **1,344 passed**, 7 existing warnings; desktop build passes.
- Full Sol dev-20 on the owner/heading fixes: **19/20**, final comparison
  answer fallback after two attempts, **48.943 s**. No transport/tool/model
  fallback. Median total/first visible **9.561 / 3.196 s**; structural audit
  has no findings, but the failed answer is still a failure.
- That trace exposed another generic parser defect: caption inference forced
  the preceding prose's F1 metric onto explicitly packed CCC/miF1/maF1 cells.
  Explicit per-cell measurement schema now wins. New paired regression is
  red→green; offline corrected draft retains 17 supported measurements after
  removing its unbound numeric caveat. Wrong metric/value controls still fail.
- Bounded affected-group replay: **3/4**, final comparison answer fallback
  after two attempts (**42.126 s**), no transport/tool/model fallback. This
  exposed separate Paper/Model columns dropping the later model identity;
  canonical paper + measured model now jointly constrain the row, with paired
  wrong-paper/wrong-model controls. Last source change is unit-verified only;
  no further Sol calls while resolving remaining contract failures.
- Remaining: abbreviated paper identity in owner cells, composite
  model/dataset/split cells and unlabelled cross-source caveats. Do not fix
  these by accepting globally matching numbers or erasing explicit owners.
- Trace hard-cap compaction retains failures; the generic redaction/bounder
  can still clip a long draft string to fit before reaching the hard-cap
  fallback. Latest affected trace has this marker and must not be described
  as a complete draft. Preserve drafts before proof/string-budget competition
  in future diagnosis while retaining the privacy/byte bounds.
- Do not relabel the earlier 19/20 full run as passing or as covering newer
  source. No clean final-source full gate yet.
  These repairs do not demonstrate better public retrieval or visual recall.

Reports: `internal/numeric-owner-final-regression-20261010.json`,
`internal/numeric-trace-sol-dev05-20261010.json` (diagnostic 4/4, two attempts,
sanitized final comparison; not a clean quality gate). Prior failed report
`internal/numeric-prose-sol-dev05-20261010.json` remains retained.
Full failure: `internal/numeric-owner-sol-final-dev20-20261010.json`;
clean structural-only diagnostic: `internal/numeric-owner-sol-final-structural-20261010.json`.
Latest replay: `internal/numeric-owner-schema-sol-affected-20261010.json`;
visible regression: `internal/numeric-owner-schema-regression-20261010.json`.
Latest source regression: `internal/numeric-owner-columns-regression-20261010.json`.

## Continuation: row binding, qualifiers and answer cleanup

- Numeric row cells now qualify other cells of the same source row. Literal
  header/value bindings (and structural Week/tuần aliases) reject a result
  borrowed from another week/dose/iteration. The previous Week xfail is fixed.
- Answer-table numeric cells also bind their peers. Numeric conditions are
  selected per clause, not unioned over the whole answer sentence. Two new
  adversarial tests were red before these repairs; both now pass.
- Missing table axes are checked against canonical siblings with different
  values, preserving explicitly named axes and equal-valued controls. This
  does not invent a paper/metric mapping or change retrieval/ranking.
- A bullet's explicit owner carries into its subsequent numbered clauses only
  when no new owner overrides it. Literal inclusion/exclusion is tied to the
  local quantity, not the entire sentence; object words come from the source.
  This is a bounded grammar check, not general semantic entailment.
- Sanitization drops a Markdown header/separator block when all its body rows
  have been removed. Safe remaining rows and document identities are retained.
  This fixes rendering, not the underlying reason those numeric rows failed.
- Visible regression: **74/85 wrong rejected; 79/87 correct accepted** (before:
  65/85 and 78/87); **68/85 paired controls** both pass. Ten case gains, zero
  regressions in this suite. **11 false accepts / 8 false rejects remain**.
  Old cached Sol facts still give the identical result: no demonstrated gain.
- First frozen Sol dev-20 run: 20/20 rubric, but manual review found an empty
  sanitized table. Subsequent source changes are not covered by that run.
  Final-source full replay is recorded separately; do not merge snapshots.
- Final backend **1,330 passed**, 7 existing warnings; no xfail. Desktop build
  and `git diff --check` pass. Lease-renewal fixture now observes renewal beyond
  the original expiry instead of relying on a fragile 120 ms scheduling window.
- Final-source Sol dev-20: **19/20 pass**, last turn failed with 9router
  `upstream connection lost` and no answer. The report remains incomplete/not
  a passing full-suite run. One bounded final-group replay then **4/4 pass**,
  including the failed case. No model/tool/answer fallback; the replay used one
  answer retry, with a **38.475 s** worst turn. Do not report "zero retries".
- Final run first-visible median **3.255 s** across its 19 visible answers;
  full report total median **9.570 s** (includes the failed turn). Replay median
  first-visible **4.462 s**, total **10.241 s**. No empty Markdown tables seen in
  either final report. Numeric/focus rubric pass is not semantic completeness.
- Graphify **5,120 nodes / 15,825 edges / 184 communities**, Obsidian **5,226
  notes**; manual `path_46.md` preserved. AST only, zero model calls. Existing
  version mismatch/14 empty-JSON/stale-label warnings remain; graph >5,000 uses
  aggregated HTML. Owned isolated backend stopped; user 9router left running.

Reports:

- `data/retrieval_eval/internal/numeric-binding-regression-rules-20261009.json`
- `data/retrieval_eval/internal/numeric-binding-regression-cached-facts-20261009.json`
- `data/retrieval_eval/internal/numeric-binding-sol-dev20-20261009.json` (pre-final fixes)
- `data/retrieval_eval/internal/numeric-binding-sol-final-dev20-20261009.json` (final replay)
- `data/retrieval_eval/internal/numeric-binding-sol-final-affected-replay-20261009.json` (bounded recovery replay)

Next: prose owner/attribute/year binding, inherited comparative predicates,
negated fees/free-shipping claims and located arithmetic. Independently
reviewed cross-domain evaluation and useful source-bound facts remain open.
Facts/dual-query stay disabled; production DB/index untouched.

## Kết luận hiện tại

Đã sửa các lỗi correctness và durability phát hiện khi review `check.txt`.
Chưa hoàn thành toàn bộ kế hoạch numeric grounding, chưa nghiệm thu RAG nền
tảng và chưa promote facts. `FACT_GROUNDING_ENABLED=false` và dual-query vẫn
tắt. Không sửa embedding, index production hoặc LightRAG.

## Những thay đổi đã kiểm chứng

- Số nhỏ, số giống năm và số được người dùng nêu trong câu hỏi không còn tự
  được xem là evidence. Thiếu nguồn là `unverifiable`, không phải pass.
- Mỗi số có offset/claim ID và assessment riêng; số trùng ở câu/ô khác không
  được dùng để hợp thức hóa claim sai. Số trong ô văn xuôi của Markdown table
  cũng được kiểm tra, không chỉ ô thuần số.
- Dùng Decimal và kiểm dấu, đơn vị, comparator. Không còn tự lấy trị tuyệt
  đối hoặc nhân/chia 100 cho mọi đại lượng. Đây chưa phải bộ chuyển đổi đơn
  vị hoặc arithmetic đầy đủ; phép chênh lệch cần các toán hạng có locator.
- Bộ kiểm metric nhiều tầng chỉ hỗ trợ occurrence tương ứng: mask chính vị
  trí số và kiểm measurement nào biến mất. Không override bằng global value
  set. Audit ghi được document và source measurement đã hỗ trợ.
- Header `model — benchmark/evaluation` được tách owner/scope; qualifier của
  group header flatten giữ parity với parser metric. Caveat đối chiếu nhiều
  nguồn không thừa kế paper cuối, nhưng vẫn kiểm số và owner được nêu trực tiếp.
- Fact phải có whole-number token, literal anchors, quote offset và đúng
  source/revision. Quote nhiều số hoặc lặp lại vẫn là proposal `unverified`.
  `source_bound` là kiểm tra literal cần thiết, **không** phải chứng minh đầy
  đủ semantic entailment. Facts chỉ thêm ràng buộc, không làm yếu canonical
  checks; không còn suppression toàn bộ occurrence có cùng một giá trị.
- Durable builds/batches có fingerprint source/model/prompt/validation policy,
  lease/heartbeat, backoff và checkpoint sau mỗi batch. Invocation tiếp theo
  phục hồi lease hết hạn/resume batch đúng fingerprint. Activation kiểm nguồn,
  active revision, lease và batch count trong transaction. Provider lỗi không
  fallback model, không làm mất projection facts trước đó.
- Reader kiểm lại quote và fingerprint của nguồn hiện tại. Fact legacy không
  tự được coi là validated. CLI backfill từ chối production/unmarked runtime.

## Kết quả và giới hạn

| Kiểm tra | Kết quả | Phạm vi |
|---|---|---|
| Backend suite | 1.311 pass, 1 xfail, 7 warnings | Không đồng nghĩa RAG acceptance |
| Focused guard/fact/lifecycle tests | 225 pass, 1 xfail | Correctness/crash fixtures |
| Sol HTTP/SSE smoke đầu | 3/3 | Isolated clone, qualitative first turns |
| Sol final affected replay | 4/4 | MSF-SER → table → correction → comparison |
| Median first visible delta / total của replay | 2,663 s / 8,482 s | 4 turns, có direct table |
| Replay 172 probe — rules | 65/85 wrong rejected; 78/87 correct accepted | Visible regression, không blind |
| Replay với Sol facts cache cũ | Cùng 65/85 và 78/87 | Không chứng minh facts có lợi |
| Sol durable build smoke | 2 batches, partial; 0 facts / 16 rejected | `9router - npm.pdf`, chỉ clone |

Final replay không transport error/model/tool fallback; không cần answer
retry/sanitization fallback. Số liệu đã kiểm trong answer cuối. Câu nhận xét
"bài Mamba giải bài toán khó hơn" vẫn là suy luận định tính của model; numeric
pass không chứng minh mọi nhận định định tính đều có evidence.

Lượt contextual 10-turn trước đó có 9 rubric pass/1 fail, và bị evaluator đánh
invalid vì source đã thay đổi trong lúc chạy. Giữ report để audit, **không**
dùng làm gate. Sau sửa mới chạy replay 4-turn trên source cố định; chưa chạy
lại toàn dev-20 hay một suite blind mới.

## Reports

- `data/retrieval_eval/internal/numeric-regression-rules-20261009.json`
- `data/retrieval_eval/internal/numeric-regression-cached-facts-20261009.json`
- `data/retrieval_eval/internal/numeric-correctness-sol-smoke-20261009.json`
- `data/retrieval_eval/internal/numeric-correctness-sol-contextual-20261009.json` (invalid)
- `data/retrieval_eval/internal/numeric-correctness-sol-final-replay-20261009.json`

Runtime: `data/retrieval_eval/internal/runtime/e2e-foundation-20261005`.
Production DB/index untouched. Graph/vault synchronized using the Graphify
skill (AST only, no model call); its version/empty-JSON warnings remain.

## Việc tiếp theo, theo thứ tự

1. Phân loại 20 false accepts và 9 false rejects còn lại. Chủ yếu là binding
   entity/attribute/condition trong prose, thiếu qualifier, negation và các
   phép tính chưa có toán hạng được định vị. Không bỏ case hay đổi expectation
   để làm đẹp điểm; expectation cần review thì ghi riêng rationale.
2. Thiết kế fact extraction có numeric/label/condition spans và quan hệ cục bộ,
   kiểm đáp án đúng lẫn sai theo cặp. Không coi literal quote là đủ; quote
   chứa nhiều số cần binding rõ trước khi dùng. Hiện policy quá bảo thủ cũng
   làm mất coverage; build smoke 0 facts là failure/partial, không thành tựu.
3. Xây bộ ≥120 balanced claims ở domain khác, annotate trước khi chạy và có
   review độc lập. Bộ 172 hiện tại chỉ dùng regression. Đo cả false accept và
   false reject, không dùng riêng tỷ lệ chặn số sai.
4. Sau local gate mới extraction bằng Sol có budget hữu hạn, so sánh flag
   off/on qua HTTP/SSE và replay full dev-20 trên source/index freeze. Không
   backfill toàn corpus hoặc promote nếu facts không có lợi rõ ràng.
5. Còn thiếu automatic coordinator/startup scheduling cho fact jobs, unit
   conversion/arithmetic tổng quát, `not_applicable` assessment explicit và
   corpus-wide semantic eval. Durable resume hiện diễn ra khi CLI/service
   được gọi lại, không được báo là đã có scheduler tự chạy toàn bộ.

Broader RAG foundation/public visual/artifact selection vẫn là hạng mục riêng
chưa đạt acceptance; commit này không chứng minh recall/embedding tăng.
