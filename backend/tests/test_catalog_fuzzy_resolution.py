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
