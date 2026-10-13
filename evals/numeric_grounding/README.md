# Numeric grounding regression

`regression-claude-v1.json` preserves Claude's second 172-probe set as portable
data: 16 scenarios, 87 correct claims and 85 wrong claims. Source documents are
shared by scenario. Expectations were authored before Claude's first run, but
the suite was subsequently used for debugging. It is **not blind or held-out**.

Run without LLM calls or database writes:

```bash
backend/.venv/bin/python scripts/evaluate_numeric_grounding.py \
  --output data/retrieval_eval/internal/numeric-regression-rules.json
```

`--cached-facts PATH` replays an existing document-ID-keyed fact cache. It does
not generate new facts; facts are revalidated against the current source.
Reports include false accepts, false rejects, paired control success, language,
category/layout slices, and suite/fact/code fingerprints. They are explicitly
invalid for blind promotion. Unit tests and this visible suite do not establish
retrieval recall, arbitrary-format readiness, or general semantic correctness.

On 2026-10-10, explicit source-subject/count-noun binding improved wrong-claim
rejection to **77/85**, with **79/87 correct accepted**, **71/85 paired pass**.
Three gains/no regressions versus Oct 9; **8 false accepts / 8 false rejects**
remain. This is bounded source grammar, not full semantic understanding.

On 2026-10-09, row/condition and literal inclusion binding improved rules from
65/85 to **74/85 wrong rejected**, and 78/87 to **79/87 correct accepted**.
Ten case gains, zero regressions; 68/85 paired controls both pass. The old Sol
fact cache gives the identical result, not a demonstrated benefit. Remaining
**11 false accepts and 8 false rejects** must stay visible. A newly authored, independently reviewed balanced
suite is still required; do not rename this set to a new blind benchmark.
