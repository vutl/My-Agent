from typing import Annotated, Any


def _with_citation_ids(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of the prompt's sources numbered S1..Sn (stable list order)."""
    return [{**document, "citation_id": f"S{index}"} for index, document in enumerate(documents, start=1)]


def _citation_map(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    for document in documents:
        citations.append({
            "id": document["citation_id"],
            "document_id": document.get("document_id"),
            "filename": document.get("filename"),
            "title": document.get("title") or document.get("document_title"),
            "page_number": document.get("page_number"),
            "source_path": document.get("source_path"),
            "caption": document.get("caption"),
            "whole_document": bool(document.get("full_document")),
        })
    return citations

