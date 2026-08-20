# Agent memory research bundle

Downloaded from the authors' arXiv records on **2026-07-19 (UTC+7)**. These
PDFs are reference material only; they are not part of Aya's indexed user
corpus unless explicitly indexed later.

| File | Paper | Design point used for Aya |
|---|---|---|
| `2310.08560-MemGPT.pdf` | [MemGPT](https://arxiv.org/abs/2310.08560) | Hierarchical context and paging between working/external memory |
| `2304.03442-Generative-Agents.pdf` | [Generative Agents](https://arxiv.org/abs/2304.03442) | Immutable experience log, retrieval, and higher-level reflection |
| `2309.02427-CoALA.pdf` | [CoALA](https://arxiv.org/abs/2309.02427) | Separate working, episodic, semantic, and procedural memory |
| `2410.10813-LongMemEval.pdf` | [LongMemEval](https://arxiv.org/abs/2410.10813) | Episode-granularity retrieval and tests for time/update/abstention |
| `2504.19413-Mem0.pdf` | [Mem0](https://arxiv.org/abs/2504.19413) | Extract/consolidate/retrieve salient long-term memories |
| `2502.12110-A-MEM.pdf` | [A-MEM](https://arxiv.org/abs/2502.12110) | Linked notes and memory evolution; retained as a later-stage option |
| `2501.13956-Zep-Graphiti.pdf` | [Zep/Graphiti](https://arxiv.org/abs/2501.13956) | Temporal validity, provenance, and superseding facts |
| `2605.12493-LongMemEval-V2.pdf` | [LongMemEval-V2](https://arxiv.org/abs/2605.12493) | Dynamic state/workflow/gotcha/premise-aware evaluation |
| `2601.01885-Agentic-Memory.pdf` | [Agentic Memory](https://arxiv.org/abs/2601.01885) | Agent-controlled memory operations; research frontier, not baseline |

## Implementation boundary

Aya adopts the low-risk common denominator first:

1. Full SQLite event/episode log remains the source of truth.
2. Working paper focus stays synchronous in L1.
3. L2 consolidation is durable, coalesced, retryable, and never required on
   the foreground response path.
4. Prompts contain the stable summary plus completed turns not yet folded, so
   an old summary is not equivalent to lost context.
5. L0 history can be retrieved as complete exchanges with timestamps.
6. L3 items are typed, versioned, provenance-bearing, and temporally closed
   when superseded.

Graph-wide autonomous memory mutation is intentionally not enabled merely
because the reference papers propose it. It needs dedicated evaluation before
it can become a trusted production behavior.
