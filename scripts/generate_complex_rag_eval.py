"""Automated Synthetic Complex RAG Benchmark Generator.

Generates multi-hop, implicit comparison, implicit figure, casual detour,
and hard negative abstention test cases directly over your local 21 papers.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Add backend to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.db.sqlite import connect
from app.rag.paper_facets import CORE_PAPER_FACETS


EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "retrieval_eval"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sqlite" / "app.db"


COMPLEX_TEST_TEMPLATES = [
    {
        "category": "implicit_comparison",
        "template": "Giữa mô hình {model_a} và mô hình {model_b} thì mô hình nào đạt kết quả cao hơn trên bộ dữ liệu {dataset}?",
        "facets": ["architecture", "benchmark_results"],
        "must_cover_all": True,
    },
    {
        "category": "implicit_visual_figure",
        "template": "Cho tôi xem sơ đồ minh họa {architecture_feature} trong bài báo {paper_title}",
        "facets": ["visual_evidence", "architecture"],
        "must_cover_all": True,
    },
    {
        "category": "casual_detour_resume",
        "turns": [
            {"message": "Bài {paper_a} đạt kết quả bao nhiêu?", "expected_route": "file_qa"},
            {"message": "À hôm nay thời tiết thế nào nhỉ?", "expected_route": "chat"},
            {"message": "Thế nếu so với bài {paper_b} thì bài nào tốt hơn?", "expected_route": "research"},
        ],
        "category": "conversational_memory",
    },
    {
        "category": "hard_negative_abstention",
        "template": "Bảng {table_num} của bài {paper_title} báo cáo chỉ số {fake_metric} trên bộ dữ liệu {fake_dataset} là bao nhiêu?",
        "facets": ["benchmark_results"],
        "must_cover_all": True,
        "expected_abstention": True,
    },
]


def load_canonical_documents() -> list[dict]:
    with connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, filename, title FROM documents ORDER BY indexed_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def load_canonical_tables() -> list[dict]:
    with connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, document_id, table_index, caption FROM document_tables LIMIT 30"
        ).fetchall()
        return [dict(row) for row in rows]


def load_canonical_figures() -> list[dict]:
    with connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, document_id, figure_index, caption FROM document_figures LIMIT 30"
        ).fetchall()
        return [dict(row) for row in rows]


def generate_complex_cases() -> list[dict]:
    docs = load_canonical_documents()
    figures = load_canonical_figures()
    tables = load_canonical_tables()
    cases: list[dict] = []

    case_idx = 1
    # 1. Implicit multi-paper comparisons
    for i in range(len(docs) - 1):
        doc_a = docs[i]
        doc_b = docs[i + 1]
        cases.append({
            "id": f"complex_{case_idx:03d}",
            "message": f"So sánh chi tiết về kiến trúc và kết quả thử nghiệm giữa {doc_a['filename']} và {doc_b['filename']}",
            "expected_route": "research",
            "expected_documents": [doc_a["id"], doc_b["id"]],
            "expected_facets": list(CORE_PAPER_FACETS),
            "must_cover_all": True,
            "query_type": "implicit_comparison",
        })
        case_idx += 1

    # 2. Implicit visual figure queries
    for fig in figures[:10]:
        caption_snippet = (fig.get("caption") or "sơ đồ kiến trúc").split(".")[0][:60]
        cases.append({
            "id": f"complex_{case_idx:03d}",
            "message": f"Hiển thị hình vẽ/sơ đồ về {caption_snippet} trong tài liệu {fig['document_id']}",
            "expected_route": "file_qa",
            "expected_documents": [fig["document_id"]],
            "expected_facets": ["visual_evidence", "architecture"],
            "must_cover_all": True,
            "query_type": "implicit_visual_figure",
            "expected_figure_id": fig["id"],
        })
        case_idx += 1

    # 3. Exact table queries
    for tbl in tables[:10]:
        cases.append({
            "id": f"complex_{case_idx:03d}",
            "message": f"Trích xuất nguyên văn Bảng {tbl['table_index']} trong file {tbl['document_id']}",
            "expected_route": "file_qa",
            "expected_documents": [tbl["document_id"]],
            "expected_facets": ["benchmark_results"],
            "must_cover_all": True,
            "query_type": "exact_table",
            "expected_table_id": tbl["id"],
        })
        case_idx += 1

    # 4. Hard negative abstention traps
    fake_metrics = ["BLEU-4", "ROUGE-L", "Exact Match 99.9%", "FID Score"]
    fake_datasets = ["ImageNet-1K", "SQuAD v2.0", "COCO 2017"]
    for idx, doc in enumerate(docs[:10]):
        fake_m = fake_metrics[idx % len(fake_metrics)]
        fake_d = fake_datasets[idx % len(fake_datasets)]
        cases.append({
            "id": f"complex_{case_idx:03d}",
            "message": f"Báo cáo chỉ số {fake_m} trên bộ dữ liệu {fake_d} trong file {doc['filename']} là bao nhiêu?",
            "expected_route": "file_qa",
            "expected_documents": [doc["id"]],
            "expected_facets": ["benchmark_results"],
            "must_cover_all": True,
            "query_type": "hard_negative_abstention",
            "expected_abstention": True,
        })
        case_idx += 1

    return cases


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cases = generate_complex_cases()
    output_file = EVAL_DIR / "complex_synthetic_eval.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"✅ Successfully generated {len(cases)} complex synthetic test cases in {output_file}")


if __name__ == "__main__":
    main()
