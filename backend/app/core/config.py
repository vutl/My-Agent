from functools import lru_cache
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_9ROUTER_MODEL = "cx/gpt-5.6-sol"
APPROVED_9ROUTER_MODELS = frozenset(
    {"cx/gpt-5.6-sol", "cx/gpt-5.6-terra", "cx/gpt-5.6-luna"}
)


_LOOPBACK_HOST_RE = re.compile(r"(?<=://)localhost(?=[:/]|$)", re.IGNORECASE)


def normalize_loopback_url(value: str) -> str:
    """Use an explicit IPv4 loopback for local sidecars.

    Some macOS resolver configurations map ``localhost`` only to ``::1`` while
    9router/Ollama bind to ``127.0.0.1``.  Normalizing only the reserved
    localhost hostname avoids intermittent connection failures without adding
    a provider or model fallback.
    """

    return _LOOPBACK_HOST_RE.sub("127.0.0.1", str(value or "")).rstrip("/")


class Settings(BaseModel):
    app_name: str = "My AI Agent"
    app_version: str = "0.1.0"
    app_env: str = Field(default="development")
    app_host: str = "127.0.0.1"
    app_port: int = 7777
    # Ollama (local)
    ollama_host: str = "http://localhost:11434"
    default_model: str = DEFAULT_9ROUTER_MODEL
    router_model: str = DEFAULT_9ROUTER_MODEL
    vision_provider: str = "openai_compatible"
    vision_model: str = DEFAULT_9ROUTER_MODEL
    embedding_model: str = "embeddinggemma:300m"
    embedding_dim: int = Field(default=768, ge=1)
    embedding_max_token_size: int = Field(default=2048, ge=128)
    embedding_query_prefix: str = "task: search result | query: "
    embedding_document_prefix: str = "title: none | text: "
    # Hybrid RRF is the measured fast/default path. The former embedding-cosine
    # reranker both duplicated first-stage semantics and regressed the v3
    # retrieval benchmark; enable reranking only with an explicitly benchmarked
    # local cross-encoder (or for controlled embedding-mode comparisons).
    rerank_enabled: bool = False
    rerank_mode: str = "cross_encoder"
    rerank_cross_encoder_path: Path | None = None
    rerank_max_candidates: int = Field(default=20, ge=4, le=100)
    agentic_retrieval_decomposition_enabled: bool = True
    # Retrieval is deliberately bounded: one initial frontier plus at most one
    # evidence-conditioned hop, with a small parallel fan-out.
    agentic_retrieval_max_hops: int = Field(default=2, ge=1, le=2)
    agentic_retrieval_max_subqueries: int = Field(default=3, ge=1, le=3)
    agentic_retrieval_hop_timeout_seconds: float = Field(default=45.0, ge=5.0, le=120.0)
    request_timeout_seconds: float = 120.0
    # L2 is "sleep-time" work. A real idle window coalesces quick user bursts;
    # pending turns remain synchronously available to prompts meanwhile.
    memory_fold_debounce_seconds: float = 12.0
    memory_worker_shutdown_timeout_seconds: float = 3.0
    # Raw prompt/draft tracing is a local debugging capability with dual opt-in:
    # the server gate and an explicit per-run request must both be enabled.
    agent_debug_trace_enabled: bool = False
    agent_debug_trace_max_bytes: int = Field(default=65_536, ge=8_192, le=262_144)
    agent_debug_trace_retention_hours: int = Field(default=72, ge=1, le=720)
    agent_debug_trace_max_runs: int = Field(default=25, ge=1, le=500)
    # Provenance-backed paper cards are built offline and remain separately
    # gated from runtime use.  Deployments can backfill first, inspect coverage,
    # then enable the card-first path without changing canonical retrieval.
    paper_evidence_cards_enabled: bool = False
    paper_evidence_card_build_enabled: bool = False
    paper_evidence_card_model: str = DEFAULT_9ROUTER_MODEL
    paper_evidence_card_max_concurrency: int = Field(default=2, ge=1, le=4)
    paper_evidence_card_schema_version: str = "v1"
    paper_evidence_card_prompt_version: str = "v2"
    paper_section_streaming_enabled: bool = False
    # LLM provider selection: "ollama" | "openai_compatible"
    # Set to "openai_compatible" to use 9router / OpenRouter / any OpenAI-compatible proxy
    llm_provider: str = "openai_compatible"
    # Intent router and answer generation intentionally share GPT via 9router.
    router_llm_provider: str = "openai_compatible"
    openai_api_base: str = "http://localhost:20128/v1"
    openai_api_key: str = ""
    # LightRAG graph index (Codex via 9router for ainsert)
    lightrag_enabled: bool = True
    lightrag_llm_model: str = DEFAULT_9ROUTER_MODEL
    lightrag_llm_api_base: str = "http://localhost:20128/v1"
    lightrag_llm_api_key: str = "any"
    lightrag_llm_timeout_seconds: float = 300.0
    # Sol graph extraction prompts are large. Keep one in flight per process so
    # the local 9router route is not saturated by four simultaneous chunks.
    lightrag_llm_max_async: int = Field(default=1, ge=1, le=16)
    # Retry only transient timeout/connection failures with the exact same
    # model. Quota/status/model-policy errors remain fail-fast.
    lightrag_llm_timeout_retries: int = Field(default=1, ge=0, le=3)
    # A 1,200-token chunk containing several dense benchmark tables repeatedly
    # timed out Sol extraction. Smaller canonical chunks bound both the prompt
    # and the entity/relation response without changing the selected model.
    lightrag_chunk_token_size: int = Field(default=600, ge=256, le=2000)
    lightrag_chunk_overlap_token_size: int = Field(default=80, ge=0, le=500)
    # Kept for backwards-compatible settings parsing only.  Runtime deliberately
    # uses the configured approved model alone: quota/provider failures must be
    # surfaced and retried later, never hidden by silently switching models.
    lightrag_llm_fallback_models: list[str] = Field(default_factory=list)
    # "auto" keeps focused/direct QA on the measured fast hybrid path, while
    # unscoped discovery and cross-document synthesis use the knowledge graph.
    # Explicit legacy/lightrag/dual values remain available for diagnostics.
    retrieval_engine: str = "auto"  # auto | lightrag | legacy | dual
    data_dir: Path = PROJECT_ROOT / "data"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:1420",
            "http://127.0.0.1:1420",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    @property
    def sqlite_db_path(self) -> Path:
        return self.data_dir / "sqlite" / "app.db"

    @property
    def lancedb_path(self) -> Path:
        return self.data_dir / "lancedb"

    @property
    def artifacts_path(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def lightrag_working_dir(self) -> Path:
        return self.data_dir / "lightrag"

    @property
    def lightrag_llm_model_chain(self) -> list[str]:
        """The single approved LightRAG model; never a silent model chain."""
        model = (self.lightrag_llm_model or "").strip()
        if model not in APPROVED_9ROUTER_MODELS:
            raise ValueError(
                "LIGHTRAG_LLM_MODEL must be an approved cx/gpt-5.6 model; "
                "refusing a silent substitution"
            )
        return [model]


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_runtime_model_policy(settings: Settings) -> None:
    """Fail startup on accidental provider/model drift for Aya's fixed roles."""
    errors: list[str] = []
    role_models = {
        "DEFAULT_MODEL": settings.default_model,
        "ROUTER_MODEL": settings.router_model,
        "VISION_MODEL": settings.vision_model,
        "LIGHTRAG_LLM_MODEL": settings.lightrag_llm_model,
        "PAPER_EVIDENCE_CARD_MODEL": settings.paper_evidence_card_model,
    }
    for role, model in role_models.items():
        if model not in APPROVED_9ROUTER_MODELS:
            errors.append(f"{role}={model!r}")
    role_providers = {
        "LLM_PROVIDER": settings.llm_provider,
        "ROUTER_LLM_PROVIDER": settings.router_llm_provider,
        "VISION_PROVIDER": settings.vision_provider,
    }
    for role, provider in role_providers.items():
        if provider != "openai_compatible":
            errors.append(f"{role}={provider!r}")
    if settings.rerank_mode not in {"embedding", "cross_encoder"}:
        errors.append(f"RERANK_MODE={settings.rerank_mode!r}")
    if settings.retrieval_engine.lower() not in {"auto", "lightrag", "legacy", "dual"}:
        errors.append(f"RETRIEVAL_ENGINE={settings.retrieval_engine!r}")
    if settings.rerank_enabled and settings.rerank_mode == "cross_encoder":
        model_path = settings.rerank_cross_encoder_path
        if model_path is None or not model_path.expanduser().is_dir():
            errors.append(
                "RERANK_CROSS_ENCODER_PATH must point to a downloaded local model directory"
            )
    if settings.lightrag_llm_fallback_models:
        errors.append("LIGHTRAG_LLM_FALLBACK_MODELS must be empty")
    if errors:
        raise RuntimeError(
            "Aya model policy requires an approved cx/gpt-5.6 model through "
            "9router with no model "
            f"fallback; invalid settings: {', '.join(errors)}"
        )


@lru_cache
def get_settings() -> Settings:
    cors_origins = _split_csv(os.getenv("CORS_ORIGINS"))
    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "7777")),
        ollama_host=normalize_loopback_url(
            os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ),
        default_model=os.getenv("DEFAULT_MODEL", DEFAULT_9ROUTER_MODEL),
        # Router follows main LLM (9router/GPT) unless explicitly overridden.
        router_model=os.getenv("ROUTER_MODEL")
        or os.getenv("DEFAULT_MODEL", DEFAULT_9ROUTER_MODEL),
        vision_provider=os.getenv("VISION_PROVIDER", "openai_compatible"),
        vision_model=os.getenv("VISION_MODEL", DEFAULT_9ROUTER_MODEL),
        embedding_model=os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
        embedding_max_token_size=int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "2048")),
        embedding_query_prefix=os.getenv(
            "EMBEDDING_QUERY_PREFIX",
            "task: search result | query: ",
        ),
        embedding_document_prefix=os.getenv(
            "EMBEDDING_DOCUMENT_PREFIX",
            "title: none | text: ",
        ),
        rerank_enabled=os.getenv("RERANK_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
        rerank_mode=os.getenv("RERANK_MODE", "cross_encoder"),
        rerank_cross_encoder_path=(
            Path(os.environ["RERANK_CROSS_ENCODER_PATH"]).expanduser()
            if os.getenv("RERANK_CROSS_ENCODER_PATH")
            else None
        ),
        rerank_max_candidates=int(os.getenv("RERANK_MAX_CANDIDATES", "20")),
        agentic_retrieval_decomposition_enabled=os.getenv(
            "AGENTIC_RETRIEVAL_DECOMPOSITION_ENABLED",
            "true",
        ).lower()
        in {"1", "true", "yes", "on"},
        agentic_retrieval_max_hops=int(
            os.getenv("AGENTIC_RETRIEVAL_MAX_HOPS", "2")
        ),
        agentic_retrieval_max_subqueries=int(
            os.getenv("AGENTIC_RETRIEVAL_MAX_SUBQUERIES", "3")
        ),
        agentic_retrieval_hop_timeout_seconds=float(
            os.getenv("AGENTIC_RETRIEVAL_HOP_TIMEOUT_SECONDS", "45")
        ),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),
        memory_fold_debounce_seconds=float(os.getenv("MEMORY_FOLD_DEBOUNCE_SECONDS", "12")),
        memory_worker_shutdown_timeout_seconds=float(
            os.getenv("MEMORY_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "3")
        ),
        agent_debug_trace_enabled=os.getenv(
            "AGENT_DEBUG_TRACE_ENABLED", "false"
        ).lower()
        in {"1", "true", "yes", "on"},
        agent_debug_trace_max_bytes=int(
            os.getenv("AGENT_DEBUG_TRACE_MAX_BYTES", "65536")
        ),
        agent_debug_trace_retention_hours=int(
            os.getenv("AGENT_DEBUG_TRACE_RETENTION_HOURS", "72")
        ),
        agent_debug_trace_max_runs=int(
            os.getenv("AGENT_DEBUG_TRACE_MAX_RUNS", "25")
        ),
        paper_evidence_cards_enabled=os.getenv(
            "PAPER_EVIDENCE_CARDS_ENABLED", "false"
        ).lower()
        in {"1", "true", "yes", "on"},
        paper_evidence_card_build_enabled=os.getenv(
            "PAPER_EVIDENCE_CARD_BUILD_ENABLED", "false"
        ).lower()
        in {"1", "true", "yes", "on"},
        paper_evidence_card_model=os.getenv(
            "PAPER_EVIDENCE_CARD_MODEL", DEFAULT_9ROUTER_MODEL
        ),
        paper_evidence_card_max_concurrency=int(
            os.getenv("PAPER_EVIDENCE_CARD_MAX_CONCURRENCY", "2")
        ),
        paper_evidence_card_schema_version=os.getenv(
            "PAPER_EVIDENCE_CARD_SCHEMA_VERSION", "v1"
        ),
        paper_evidence_card_prompt_version=os.getenv(
            "PAPER_EVIDENCE_CARD_PROMPT_VERSION", "v2"
        ),
        paper_section_streaming_enabled=os.getenv(
            "PAPER_SECTION_STREAMING_ENABLED", "false"
        ).lower()
        in {"1", "true", "yes", "on"},
        data_dir=Path(os.getenv("APP_DATA_DIR", str(PROJECT_ROOT / "data"))),
        cors_origins=cors_origins or Settings().cors_origins,
        llm_provider=os.getenv("LLM_PROVIDER", "openai_compatible"),
        router_llm_provider=os.getenv("ROUTER_LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "openai_compatible"),
        openai_api_base=normalize_loopback_url(
            os.getenv("OPENAI_API_BASE", "http://localhost:20128/v1")
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        lightrag_enabled=os.getenv("LIGHTRAG_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
        lightrag_llm_model=os.getenv(
            "LIGHTRAG_LLM_MODEL",
            os.getenv("DEFAULT_MODEL", DEFAULT_9ROUTER_MODEL),
        ),
        lightrag_llm_api_base=normalize_loopback_url(
            os.getenv(
                "LIGHTRAG_LLM_API_BASE",
                os.getenv("OPENAI_API_BASE", "http://localhost:20128/v1"),
            )
        ),
        lightrag_llm_api_key=os.getenv(
            "LIGHTRAG_LLM_API_KEY",
            os.getenv("OPENAI_API_KEY", "any") or "any",
        ),
        lightrag_llm_timeout_seconds=float(os.getenv("LIGHTRAG_LLM_TIMEOUT_SECONDS", "300")),
        lightrag_llm_max_async=int(os.getenv("LIGHTRAG_LLM_MAX_ASYNC", "1")),
        lightrag_llm_timeout_retries=int(
            os.getenv("LIGHTRAG_LLM_TIMEOUT_RETRIES", "1")
        ),
        lightrag_chunk_token_size=int(os.getenv("LIGHTRAG_CHUNK_TOKEN_SIZE", "600")),
        lightrag_chunk_overlap_token_size=int(
            os.getenv("LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE", "80")
        ),
        lightrag_llm_fallback_models=_split_csv(os.getenv("LIGHTRAG_LLM_FALLBACK_MODELS")) or [],
        retrieval_engine=os.getenv("RETRIEVAL_ENGINE", "auto"),
    )
