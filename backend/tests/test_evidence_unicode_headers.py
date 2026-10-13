from app.services.evidence_validator import validate_answer_claims


def test_unicode_math_metric_headers_preserve_dimension_identity() -> None:
    documents = [
        {
            "document_id": "paper-1",
            "filename": "ASPIRE.pdf",
            "content": (
                "| Model | Modality | CCC 𝐴 | CCC𝑉 |\n"
                "|---|---|---:|---:|\n"
                "| ASPIRE | A+T | 0.714 | 0.740 |"
            ),
        }
    ]

    result = validate_answer_claims(
        answer="ASPIRE reports CCC A 0.714 and CCC V 0.740.",
        documents=documents,
        focus_document_ids=["paper-1"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value, claim.qualifiers) for claim in result.checked_claims] == [
        ("ccc", "0.714", ("arousal",)),
        ("ccc", "0.74", ("valence",)),
    ]
