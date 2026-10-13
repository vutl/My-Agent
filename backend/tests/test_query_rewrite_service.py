

def test_specific_questions_are_not_diluted_with_overview_anchors():
    from app.services.query_rewrite_service import enrich_retrieval_query

    specific = enrich_retrieval_query(
        "ASPIRE dùng WavLM variant nào cho nhánh global acoustic và pooling kiểu gì?",
        topic="ASPIRE", entities=["ASPIRE"], answer_intent="define", focus_document_ids=["aspire"],
    )
    assert "introduction abstract" not in specific and "global acoustic" in specific
    assert specific.startswith("ASPIRE ")


def test_formula_asks_keep_their_own_topic_and_loss_asks_keep_training_terms():
    from app.services.query_rewrite_service import enrich_retrieval_query

    attention = enrich_retrieval_query(
        "Show the attention formula ASPIRE uses in its cross-modal interaction.",
        topic="ASPIRE", entities=["ASPIRE"], answer_intent="direct_answer", focus_document_ids=["aspire"],
    )
    assert "optimizer" not in attention and "equation" in attention
    loss = enrich_retrieval_query(
        "Write out the total training loss that ASPIRE minimizes.",
        topic="ASPIRE", entities=["ASPIRE"], answer_intent="direct_answer", focus_document_ids=["aspire"],
    )
    assert "loss" in loss and "minimize" in loss
