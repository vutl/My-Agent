from app.services.rag_service import _catalog_document_aliases, _infer_catalog_mention, _search_words

ROWS = [
    {"id": "d1", "filename": "ORION.pdf", "title": "ORION: Orbital Reasoning for Image Outlier Novelty", "title_guess": "", "keywords_json": "[]"},
    {"id": "d2", "filename": "TideNet.pdf", "title": "TideNet: Coastal Flood Forecasting with Graph Networks", "title_guess": "", "keywords_json": '["storm-surge"]'},
    {"id": "d3", "filename": "Lumen.pdf", "title": "A Survey of Low-Light Image Enhancement", "title_guess": "", "keywords_json": "[]"},
]


def _infer(query, library=frozenset()):
    aliases_by_document, owners = {}, {}
    for row in ROWS:
        aliases = _catalog_document_aliases(filename=row["filename"], title=row["title"],
                                            title_guess=row["title_guess"], keywords_json=row["keywords_json"])
        aliases_by_document[row["id"]] = aliases
        for alias in aliases:
            owners.setdefault(alias, set()).add(row["id"])
    mention = _infer_catalog_mention(_search_words(query), ROWS, aliases_by_document, owners,
                                     word_in_library=lambda word: word in library)
    return mention.document_id if mention else None


def test_misspelled_names_resolve_to_the_single_closest_document():
    assert _infer("What does ORIOM do?") == "d1"
    assert _infer("tidnet ket qua the nao") == "d2"
    assert _infer("Tide Nett results") == "d2"


def test_real_library_words_are_never_corrected():
    # "orion" typo distance 1 from "orien"; if the library uses "orien" it is a real word.
    assert _infer("what is orien", library={"orien"}) is None


def test_description_resolves_by_rare_title_words_with_a_document_word():
    assert _infer("the paper on coastal flood forecasting") == "d2"
    assert _infer("bài về low light image enhancement") == "d3"
    # Without a document word, a description is not treated as a paper reference.
    assert _infer("coastal flood forecasting") is None


def test_unrelated_and_generic_questions_resolve_to_nothing():
    for query in ("what is the weather today", "which model is better", "show the results table"):
        assert _infer(query) is None


def test_short_words_are_not_typos_of_short_names():
    rows = [{"id": "g", "filename": "GBAN.pdf", "title": "Gated Bidirectional Alignment Network", "title_guess": "",
             "keywords_json": "[]"}]
    aliases = _catalog_document_aliases(filename="GBAN.pdf", title=rows[0]["title"], title_guess="", keywords_json="[]")
    owners = {alias: {"g"} for alias in aliases}
    mention = _infer_catalog_mention(_search_words("Benchmark multi-label bản final giữ bao nhiêu tổ hợp nhãn?"),
                                     rows, {"g": aliases}, owners, word_in_library=lambda word: False)
    assert mention is None
    typo = _infer_catalog_mention(_search_words("gbann dùng dataset nào?"), rows, {"g": aliases}, owners,
                                  word_in_library=lambda word: False)
    assert typo is not None and typo.document_id == "g"


def test_two_misspelled_names_both_resolve():
    rows = [
        {"id": "a", "filename": "ASPIRE.pdf", "title": "ASPIRE: Attentive Semantic-Prosody Interaction", "title_guess": "", "keywords_json": "[]"},
        {"id": "c", "filename": "CRAB.pdf", "title": "Crab: Multi Layer Contrastive Supervision", "title_guess": "", "keywords_json": "[]"},
        {"id": "k", "filename": "KST.pdf", "title": "Key-Sparse Transformer", "title_guess": "", "keywords_json": "[]"},
    ]
    aliases_by_document, owners = {}, {}
    for row in rows:
        aliases = _catalog_document_aliases(filename=row["filename"], title=row["title"], title_guess="", keywords_json="[]")
        aliases_by_document[row["id"]] = aliases
        for alias in aliases:
            owners.setdefault(alias, set()).add(row["id"])
    found = _infer_catalog_mention(_search_words("aspier vs crabb cai nao it tham so hon"), rows, aliases_by_document,
                                   owners, word_in_library=lambda word: False, return_all=True)
    assert [mention.document_id for mention in found] == ["a", "c"]


def test_vietnamese_words_are_not_misspelled_names():
    from app.services.rag_service import _accented_words

    rows = [{"id": "v", "filename": "ViSEC.pdf", "title": "ViSEC: Vietnamese speech emotion corpus", "title_guess": "",
             "keywords_json": "[]"}]
    aliases = _catalog_document_aliases(filename="ViSEC.pdf", title=rows[0]["title"], title_guess="", keywords_json="[]")
    owners = {alias: {"v"} for alias in aliases}
    query = "Cho mình xin một lời khuyên để tập trung làm việc tốt hơn."
    assert _infer_catalog_mention(_search_words(query), rows, {"v": aliases}, owners,
                                  word_in_library=lambda word: False, native_words=_accented_words(query)) is None
    typo = "visex dung dataset nao"
    found = _infer_catalog_mention(_search_words(typo), rows, {"v": aliases}, owners,
                                   word_in_library=lambda word: False, native_words=_accented_words(typo))
    assert found is not None and found.document_id == "v"


def test_title_acronym_is_a_name_even_when_it_is_an_app_word():
    rows = [
        {"id": "r", "filename": "en_paper_rag_knowledge_intensive_nlp.pdf",
         "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "title_guess": "", "keywords_json": "[]"},
        {"id": "l", "filename": "en_paper_lost_in_the_middle_long_context.pdf",
         "title": "Lost in the Middle: How Language Models Use Long Contexts", "title_guess": "", "keywords_json": "[]"},
    ]
    aliases_by_document = {
        row["id"]: _catalog_document_aliases(filename=row["filename"], title=row["title"], title_guess="", keywords_json="[]")
        for row in rows
    }
    assert aliases_by_document["r"][("rag",)] == (2, "title_acronym")
    assert ("rag",) not in aliases_by_document["l"]


def test_title_head_before_colon_is_a_name():
    aliases = _catalog_document_aliases(
        filename="en_paper_retoolsql_agentic_rl_text_to_sql.pdf",
        title="ReToolSQL: Agentic Reinforcement Learning for Robust Text-to-SQL", title_guess="", keywords_json="[]",
    )
    assert aliases[("retoolsql",)] == (2, "title_name")
    plain = _catalog_document_aliases(
        filename="notes.txt", title="Source: https://example.org/page", title_guess="", keywords_json="[]",
    )
    assert ("source",) not in plain


def test_reporting_periods_that_disagree_conflict():
    from app.services.rag_service import period_marks, periods_conflict

    query = period_marks("According to the NSO's Q3 2026 socio-economic report, first nine months")
    august = period_marks("en_nso_socio_economic_report_aug2026.docx Report Socio-Economic Performance in August and 8 Months of 2026")
    third = period_marks("Báo Cáo Tình Hình Kinh Tế - Xã Hội Quý III Và 9 Tháng Năm 2026")
    assert periods_conflict(query, august) and not periods_conflict(query, third)
    assert period_marks("you may ask")["periods"] == set()
    assert period_marks("worldbank update may2026")["periods"] == {"m5"}
    # A year asked about is content, not the document's identity.
    assert not periods_conflict(period_marks("the 2027 projection"), period_marks("Summary of Economic Projections, September 16, 2026"))


def test_lowercase_filename_acronym_is_a_name_when_the_document_writes_it_in_capitals():
    body = "The FOMC met in September. FOMC participants submitted projections of GDP and GDP growth."
    kwargs = dict(filename="en_fed_fomc_summary_sep2026.pdf", title="Summary of Economic Projections", title_guess="",
                  keywords_json="[]", body_text=body)
    assert ("fomc",) in _catalog_document_aliases(**kwargs, phrase_document_count=lambda phrase: 2)
    # A capitalised word most documents use (GDP) is a topic, not a name.
    assert ("fomc",) not in _catalog_document_aliases(**kwargs, phrase_document_count=lambda phrase: 9)


def test_editions_of_one_series_are_told_apart_by_the_named_period():
    from app.services.rag_service import CatalogDocumentMention, _disambiguate_by_period, period_marks

    shared = CatalogDocumentMention(surface="fomc", start_word=0, end_word=1, document_id=None,
                                    candidate_ids=("jun", "sep"), alias_source="filename_acronym", strength=2)
    identity = {"jun": "en_scan_fed_fomc_sep_jun2026.pdf", "sep": "Summary of Economic Projections, September 16, 2026"}
    identity["jun"] = "en_scan_fed_fomc_projections_jun2026.pdf"
    one = _disambiguate_by_period([shared], period_marks("FOMC tháng 9/2026 projections"), identity)
    assert [m.document_id for m in one] == ["sep"]
    both = _disambiguate_by_period([shared], period_marks("between the June and September FOMC projections"), identity)
    assert sorted(m.document_id for m in both) == ["jun", "sep"]
