#!/usr/bin/env python3
"""Build and freeze Aya's dev/held-out conversational evaluation fixtures."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "retrieval_eval"
DB = ROOT / "data" / "sqlite" / "app.db"

DOC = {
    "router_vi": "bc2f83cb-d9ea-4157-80bb-5162f178ea05",
    "router_npm": "026a89c3-8989-4d21-8d3b-5ec87bbbc504",
    "aspire": "405459dc-9188-4b0b-b2f1-c74347a334c4",
    "lpmn": "148ce763-543e-4349-800d-875eea2c45cf",
    "cmdm": "d2e78acd-901f-40af-8bc1-01c89fa8de3a",
    "crab": "ebf777ce-7455-463a-aed0-b7f9b06bb79e",
    "dimensional": "09444939-8d3d-440d-8a19-e1e5bd0fb2d9",
    "mamba": "9d8be1eb-a127-45b0-b1ed-117ca2f3106d",
    "gban": "50ad3675-c7cf-4cdf-b846-afcabe9d66fc",
    "visec": "41e4f817-15eb-459a-9344-77b9f407570f",
    "jait": "31fc17d0-9b37-4ca8-b5bc-ce44b2843ee9",
    "kst": "adbb18f4-6ceb-465f-b6c3-0168659fc392",
    "msf": "6a4156b1-3fcf-431c-86ec-37707859d0b6",
    "multimodal_vad": "20de57c1-701f-4cde-82a0-dcd8b4413823",
    "rag_plan": "e1b97c3d-efab-4c22-b00a-cce24bd251af",
    "storage_plan": "90a74e86-8910-4b46-be8f-00a771954096",
    "mac_plan": "50281e0b-c4b6-41d0-a8ca-7a8f3352bb54",
    "robust_av": "ff3363ff-4b19-4ef2-8924-889869d1390e",
    "whiser": "6d72767b-4ee2-424e-ac70-b252f76ed64e",
    "demo": "3d0363cb-2f7e-4c75-a6d5-eeb8886ce04e",
    "wav2small": "bac594d6-64fe-41d8-b18c-f24c626813a2",
}

TABLE = {
    "aspire_2": "a7b425c9-d142-53bb-8945-3c11150b52de",
    "visec_2": "33e86dde-bf21-5ce1-93f0-443cecfb7a96",
    "whiser_1": "f8e76965-7112-58f1-beaf-f9f30d3a776d",
    "kst_2": "7409830c-d04b-5834-9ba6-3c9167ecb78b",
    "cmdm_5": "0fd36d13-7f19-5709-aa82-e278354626e9",
    "cmdm_6": "f9810119-04f7-5d8e-8e89-1b95be7e3b06",
    "msf_3": "fb82eb63-6448-50d8-aa7c-cb7db464eef8",
    "lpmn_2": "b551b6e4-a8c0-5a43-ae8f-d8ee890623c4",
    "mamba_2": "5066674b-1a83-5738-b004-eb93b9266007",
    "robust_2": "f0761ab7-dbc8-5383-a7ae-4e09d585d04f",
    "multimodal_1": "8d8f2927-94c6-5c85-8b9b-4d84d4a872c5",
    "jait_4": "efdca2a4-13d2-5b31-8d48-0f23e8b319d7",
}

F = {
    "task": ["task"],
    "task_contribution": ["task", "contributions"],
    "architecture": ["architecture"],
    "dataset": ["dataset_setup"],
    "results": ["benchmark_results"],
    "architecture_results": ["architecture", "benchmark_results"],
    "compare": [
        "task",
        "architecture",
        "dataset_setup",
        "benchmark_results",
        "contributions",
    ],
}


def case(
    group: str,
    turn: int,
    message: str,
    *,
    language: str,
    documents: list[str] | None = None,
    facets: list[str] | None = None,
    route: str | list[str] = "file_qa",
    must_cover_all: bool = False,
    tables: list[str] | None = None,
    figures: list[str] | None = None,
    abstain: bool = False,
) -> dict:
    ids = [DOC[name] for name in documents or []]
    return {
        "case_id": f"{group}-t{turn}",
        "conversation_group": group,
        "turn_index": turn,
        "language": language,
        "message": message,
        "expected_route": route,
        "expected_document_ids": ids,
        "must_cover_all": must_cover_all,
        "expected_facets": {document_id: list(facets or []) for document_id in ids},
        "acceptable_evidence_groups": [],
        "forbidden_document_ids": [],
        "expected_artifacts": {
            "table_ids": list(tables or []),
            "figure_ids": list(figures or []),
        },
        "expected_abstention": abstain,
        "answer_must_contain": [],
        "answer_must_not_contain": [],
        "latency_class": "generation_bound" if route != "chat" else "chat",
    }


MULTI_ROUTE = ["file_qa", "research"]


DEV = [
    case("dev01", 1, "Bài ASPIRE giải quyết task gì?", language="vi", documents=["aspire"], facets=F["task"]),
    case("dev01", 2, "Nói ngắn gọn thôi, kiến trúc của nó thế nào?", language="vi", documents=["aspire"], facets=F["architecture"]),
    case("dev01", 3, "Chuyển sang ViSEC: dataset nào được dùng?", language="vi", documents=["visec"], facets=F["dataset"]),
    case("dev01", 4, "So sánh ASPIRE với ViSEC về kết quả", language="vi", documents=["aspire", "visec"], facets=F["results"], route=MULTI_ROUTE, must_cover_all=True),
    case("dev02", 1, "What is the main task of WhiSER?", language="en", documents=["whiser"], facets=F["task"]),
    case("dev02", 2, "Show Table 1 from WhiSER.", language="en", documents=["whiser"], facets=F["results"], tables=[TABLE["whiser_1"]]),
    case("dev02", 3, "Let's talk about lunch for a moment.", language="en", route="chat"),
    case("dev02", 4, "Return to the previous paper and explain its contribution.", language="en", documents=["whiser"], facets=F["task_contribution"]),
    case("dev03", 1, "KST dùng architecture gì?", language="vi", documents=["kst"], facets=F["architecture"]),
    case("dev03", 2, "JAIT và KST khác nhau thế nào?", language="vi", documents=["jait", "kst"], facets=F["compare"], route=MULTI_ROUTE, must_cover_all=True),
    case("dev03", 3, "Ý tôi là GBAN, không phải JAIT.", language="vi", documents=["gban"], facets=F["task_contribution"]),
    case("dev03", 4, "Đưa kết quả của GBAN", language="vi", documents=["gban"], facets=F["results"]),
    case("dev04", 1, "Summarize the LPMN paper's task and contribution.", language="en", documents=["lpmn"], facets=F["task_contribution"]),
    case("dev04", 2, "Give me Table 2 from LPMN.", language="en", documents=["lpmn"], facets=F["results"], tables=[TABLE["lpmn_2"]]),
    case("dev04", 3, "Compare LPMN against wav2small.", language="en", documents=["lpmn", "wav2small"], facets=F["compare"], route=MULTI_ROUTE, must_cover_all=True),
    case("dev04", 4, "Which datasets do both of them use?", language="en", documents=["lpmn", "wav2small"], facets=F["dataset"], route=MULTI_ROUTE, must_cover_all=True),
    case("dev05", 1, "Bài MSF-SER làm gì?", language="vi", documents=["msf"], facets=F["task_contribution"]),
    case("dev05", 2, "Cho bảng 3 MSF-SER.", language="vi", documents=["msf"], facets=F["results"], tables=[TABLE["msf_3"]]),
    case("dev05", 3, "Đừng lấy MSF-SER; lấy bài Mamba fusion và nói dataset.", language="vi", documents=["mamba"], facets=F["dataset"]),
    case("dev05", 4, "So sánh kết quả hai bài đó.", language="vi", documents=["msf", "mamba"], facets=F["results"], route=MULTI_ROUTE, must_cover_all=True),
]


HELDOUT = [
    # 9 Vietnamese groups = 36 turns.
    case("h01", 1, "Bài ASPIRE làm gì và đóng góp chính là gì?", language="vi", documents=["aspire"], facets=F["task_contribution"]),
    case("h01", 2, "Tối nay uống trà hay cà phê nhỉ?", language="vi", route="chat"),
    case("h01", 3, "Quay lại bài trước, kiến trúc của nó ra sao?", language="vi", documents=["aspire"], facets=F["architecture"]),
    case("h01", 4, "So sánh ASPIRE với ViSEC về architecture, dataset và results.", language="vi", documents=["aspire", "visec"], facets=["architecture", "dataset_setup", "benchmark_results"], route=MULTI_ROUTE, must_cover_all=True),

    case("h02", 1, "WhiSER nghiên cứu bài toán gì?", language="vi", documents=["whiser"], facets=F["task"]),
    case("h02", 2, "Đưa nguyên Bảng 1 của WhiSER.", language="vi", documents=["whiser"], facets=F["results"], tables=[TABLE["whiser_1"]]),
    case("h02", 3, "Còn wav2small có contribution gì?", language="vi", documents=["wav2small"], facets=F["task_contribution"]),
    case("h02", 4, "Đối chiếu kiến trúc và kết quả WhiSER với wav2small.", language="vi", documents=["whiser", "wav2small"], facets=F["architecture_results"], route=MULTI_ROUTE, must_cover_all=True),

    case("h03", 1, "KST có kiến trúc như nào?", language="vi", documents=["kst"], facets=F["architecture"]),
    case("h03", 2, "JAIT với KST khác nhau ở task và architecture thế nào?", language="vi", documents=["jait", "kst"], facets=["task", "architecture"], route=MULTI_ROUTE, must_cover_all=True),
    case("h03", 3, "Không, JAIT chỉ là ví dụ; lấy dataset của GBAN cơ.", language="vi", documents=["gban"], facets=F["dataset"]),
    case("h03", 4, "Đối chiếu kết quả GBAN và CRAB.", language="vi", documents=["gban", "crab"], facets=F["results"], route=MULTI_ROUTE, must_cover_all=True),

    case("h04", 1, "CMDM giải quyết task gì và đề xuất gì?", language="vi", documents=["cmdm"], facets=F["task_contribution"]),
    case("h04", 2, "Cho tôi Bảng 5 của CMDM.", language="vi", documents=["cmdm"], facets=F["results"], tables=[TABLE["cmdm_5"]]),
    case("h04", 3, "Bảng 6 của CMDM thì sao?", language="vi", documents=["cmdm"], facets=F["results"], tables=[TABLE["cmdm_6"]]),
    case("h04", 4, "Đừng lấy CMDM nữa; tóm tắt contribution của CRAB.", language="vi", documents=["crab"], facets=F["task_contribution"]),

    case("h05", 1, "Bài MSF-SER làm nhiệm vụ gì?", language="vi", documents=["msf"], facets=F["task"]),
    case("h05", 2, "Đưa Bảng 3 bài MSF-SER.", language="vi", documents=["msf"], facets=F["results"], tables=[TABLE["msf_3"]]),
    case("h05", 3, "Tạm dừng, kể một câu vui ngắn đi.", language="vi", route="chat"),
    case("h05", 4, "Quay lại paper vừa rồi, contribution và architecture là gì?", language="vi", documents=["msf"], facets=["contributions", "architecture"]),

    case("h06", 1, "ViSEC dùng những dataset và protocol nào?", language="vi", documents=["visec"], facets=F["dataset"]),
    case("h06", 2, "Đưa Bảng 2 ViSEC.", language="vi", documents=["visec"], facets=F["results"], tables=[TABLE["visec_2"]]),
    case("h06", 3, "Cà phê sữa hay cà phê đen ngon hơn?", language="vi", route="chat"),
    case("h06", 4, "Giờ so sánh paper trước với ASPIRE trên cả năm khía cạnh.", language="vi", documents=["visec", "aspire"], facets=F["compare"], route=MULTI_ROUTE, must_cover_all=True),

    case("h07", 1, "LPMN làm gì và điểm mới ở đâu?", language="vi", documents=["lpmn"], facets=F["task_contribution"]),
    case("h07", 2, "Cho Bảng 2 của LPMN.", language="vi", documents=["lpmn"], facets=F["results"], tables=[TABLE["lpmn_2"]]),
    case("h07", 3, "So LPMN với KST về kiến trúc.", language="vi", documents=["lpmn", "kst"], facets=F["architecture"], route=MULTI_ROUTE, must_cover_all=True),
    case("h07", 4, "Hai bài đó dùng dataset nào?", language="vi", documents=["lpmn", "kst"], facets=F["dataset"], route=MULTI_ROUTE, must_cover_all=True),

    case("h08", 1, "Paper Mamba-based fusion giải quyết task và dataset gì?", language="vi", documents=["mamba"], facets=["task", "dataset_setup"]),
    case("h08", 2, "Đưa Bảng 2 của paper Mamba fusion.", language="vi", documents=["mamba"], facets=F["results"], tables=[TABLE["mamba_2"]]),
    case("h08", 3, "So sánh nó với MSF-SER.", language="vi", documents=["mamba", "msf"], facets=F["compare"], route=MULTI_ROUTE, must_cover_all=True),
    case("h08", 4, "Chỉ tập trung benchmark results của cả hai.", language="vi", documents=["mamba", "msf"], facets=F["results"], route=MULTI_ROUTE, must_cover_all=True),

    case("h09", 1, "Robust Audio-Visual Fusion có architecture gì?", language="vi", documents=["robust_av"], facets=F["architecture"]),
    case("h09", 2, "Đưa Bảng 2 của Robust Audio-Visual Fusion.", language="vi", documents=["robust_av"], facets=F["results"], tables=[TABLE["robust_2"]]),
    case("h09", 3, "So sánh Robust Audio-Visual Fusion và GBAN.", language="vi", documents=["robust_av", "gban"], facets=F["compare"], route=MULTI_ROUTE, must_cover_all=True),
    case("h09", 4, "Hai bài đóng góp khác nhau ở đâu?", language="vi", documents=["robust_av", "gban"], facets=["contributions"], route=MULTI_ROUTE, must_cover_all=True),

    # 6 English groups = 24 turns.
    case("h10", 1, "What task does Multimodal Recognition of Valence, Arousal and Dominance address?", language="en", documents=["multimodal_vad"], facets=F["task"]),
    case("h10", 2, "Show Table 1 from that paper.", language="en", documents=["multimodal_vad"], facets=F["results"], tables=[TABLE["multimodal_1"]]),
    case("h10", 3, "Compare it with Dimensional Emotion Detection from Categorical Emotion.", language="en", documents=["multimodal_vad", "dimensional"], facets=F["compare"], route=MULTI_ROUTE, must_cover_all=True),
    case("h10", 4, "Focus on the datasets and benchmark results of both.", language="en", documents=["multimodal_vad", "dimensional"], facets=["dataset_setup", "benchmark_results"], route=MULTI_ROUTE, must_cover_all=True),

    case("h11", 1, "Summarize the task and contribution of PROJECT_PLAN_ADDENDUM_RAG_AGENT_MEMORY_VISUAL.", language="en", documents=["rag_plan"], facets=F["task_contribution"]),
    case("h11", 2, "What architecture does that plan propose?", language="en", documents=["rag_plan"], facets=F["architecture"]),
    case("h11", 3, "Compare that RAG-memory plan with PROJECT_PLAN_ADDENDUM_STORAGE_CATALOG_LANCEDB.", language="en", documents=["rag_plan", "storage_plan"], facets=F["compare"], route=MULTI_ROUTE, must_cover_all=True),
    case("h11", 4, "Now contrast the RAG-memory plan with demo.md on task only.", language="en", documents=["rag_plan", "demo"], facets=F["task"], route=MULTI_ROUTE, must_cover_all=True),

    case("h12", 1, "What is PROJECT_PLAN_ADDENDUM_STORAGE_CATALOG_LANCEDB about?", language="en", documents=["storage_plan"], facets=F["task_contribution"]),
    case("h12", 2, "Switch to demo.md and describe its task.", language="en", documents=["demo"], facets=F["task"]),
    case("h12", 3, "Use PROJECT_PLAN_mac_ai_agent instead; summarize its architecture.", language="en", documents=["mac_plan"], facets=F["architecture"]),
    case("h12", 4, "Compare the storage plan and the Mac AI agent plan.", language="en", documents=["storage_plan", "mac_plan"], facets=F["compare"], route=MULTI_ROUTE, must_cover_all=True),

    case("h13", 1, "What does the paper file '9router - npm.pdf' document?", language="en", documents=["router_npm"], facets=F["task_contribution"]),
    case("h13", 2, "Now use '9Router là gì - LLM Gateway cho AI coding tools.pdf'.", language="en", documents=["router_vi"], facets=F["task_contribution"]),
    case("h13", 3, "Compare those two 9router documents.", language="en", documents=["router_npm", "router_vi"], facets=F["compare"], route=MULTI_ROUTE, must_cover_all=True),
    case("h13", 4, "Not the npm document; return to the Vietnamese 9Router document's contribution.", language="en", documents=["router_vi"], facets=["contributions"]),

    case("h14", 1, "Show Table 4 from JAIT-V16N8-1127.", language="en", documents=["jait"], facets=F["results"], tables=[TABLE["jait_4"]]),
    case("h14", 2, "Show Table 2 from KST.", language="en", documents=["kst"], facets=F["results"], tables=[TABLE["kst_2"]]),
    case("h14", 3, "Compare JAIT and KST using their tasks, architectures and datasets.", language="en", documents=["jait", "kst"], facets=["task", "architecture", "dataset_setup"], route=MULTI_ROUTE, must_cover_all=True),
    case("h14", 4, "Pause the papers: what is a transformer in general?", language="en", route="chat"),

    case("h15", 1, "Give me Table 99 from ASPIRE.", language="en", documents=["aspire"], facets=F["results"], abstain=True),
    case("h15", 2, "Give me Figure 99 from ViSEC.", language="en", documents=["visec"], facets=["visual_evidence"], abstain=True),
    case("h15", 3, "Compare ASPIRE and KST only on an experiment detail that is absent from both sources.", language="en", documents=["aspire", "kst"], facets=F["results"], route=MULTI_ROUTE, must_cover_all=True, abstain=True),
    case("h15", 4, "Forget the papers for this turn and explain recursion generally.", language="en", route="chat"),
]


def _validate(cases: list[dict], *, expected_count: int, heldout: bool) -> None:
    assert len(cases) == expected_count
    ids = [case["case_id"] for case in cases]
    assert len(ids) == len(set(ids))
    groups = Counter(case["conversation_group"] for case in cases)
    assert all(count == 4 for count in groups.values())
    assert all(case["turn_index"] in {1, 2, 3, 4} for case in cases)
    if heldout:
        assert groups == Counter({f"h{index:02d}": 4 for index in range(1, 16)})
        languages = Counter(case["language"] for case in cases)
        assert languages == {"vi": 36, "en": 24}
        mentions = Counter(
            document_id
            for case in cases
            for document_id in case["expected_document_ids"]
        )
        missing = {name: mentions[document_id] for name, document_id in DOC.items() if mentions[document_id] < 2}
        assert not missing, f"documents below two held-out appearances: {missing}"
    messages = [" ".join(case["message"].lower().split()) for case in cases]
    assert len(messages) == len(set(messages))


def _write_jsonl(path: Path, cases: list[dict]) -> str:
    rendered = "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases)
    path.write_text(rendered, encoding="utf-8")
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _corpus_fingerprint() -> tuple[str, list[dict]]:
    with sqlite3.connect(DB) as connection:
        rows = connection.execute(
            "SELECT id, filename, content_hash FROM documents ORDER BY id"
        ).fetchall()
    documents = [
        {"document_id": row[0], "filename": row[1], "content_hash": row[2]}
        for row in rows
    ]
    rendered = json.dumps(documents, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest(), documents


def main() -> int:
    _validate(DEV, expected_count=20, heldout=False)
    _validate(HELDOUT, expected_count=60, heldout=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dev_path = OUT / "conversational-dev-v1.jsonl"
    heldout_path = OUT / "conversational-heldout-v1.jsonl"
    dev_sha = _write_jsonl(dev_path, DEV)
    heldout_sha = _write_jsonl(heldout_path, HELDOUT)
    corpus_sha, documents = _corpus_fingerprint()
    manifest = {
        "schema_version": "v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "policy": {
            "dev_turns": 20,
            "heldout_turns": 60,
            "heldout_groups": 15,
            "heldout_group_size": 4,
            "heldout_open_requires_flag": "--allow-heldout",
            "retire_version_after_tuning_on_heldout": True,
        },
        "files": {
            dev_path.name: {"sha256": dev_sha, "turns": len(DEV)},
            heldout_path.name: {"sha256": heldout_sha, "turns": len(HELDOUT)},
        },
        "coverage": {
            "languages": Counter(case["language"] for case in HELDOUT),
            "routes": Counter(str(case["expected_route"]) for case in HELDOUT),
            "must_cover_all": sum(bool(case["must_cover_all"]) for case in HELDOUT),
            "exact_artifact": sum(
                bool(case["expected_artifacts"]["table_ids"] or case["expected_artifacts"]["figure_ids"])
                for case in HELDOUT
            ),
            "abstention": sum(bool(case["expected_abstention"]) for case in HELDOUT),
            "document_mentions": Counter(
                document_id
                for case in HELDOUT
                for document_id in case["expected_document_ids"]
            ),
        },
        "corpus": {
            "fingerprint": corpus_sha,
            "documents": documents,
        },
    }
    (OUT / "conversational-heldout-v1.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"dev_sha256": dev_sha, "heldout_sha256": heldout_sha, "corpus_sha256": corpus_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
