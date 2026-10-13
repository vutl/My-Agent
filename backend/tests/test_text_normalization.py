from app.rag.text_normalization import needs_search_normalization, searchable_text


def test_math_letters_become_plain_searchable_words():
    assert searchable_text("L𝑡𝑜𝑡𝑎𝑙 = L 𝐶𝐸 + 𝜆 𝑎𝑟𝑜") == "L total = L CE + λ aro"
    assert searchable_text("CCC 𝐴 and ℒ") == "CCC A and L"


def test_numbers_scripts_and_vietnamese_are_untouched():
    for text in ("10² và x₁", "Việt Nam đạt 81,2%", "plain text"):
        assert searchable_text(text) == text
        assert not needs_search_normalization(text)
