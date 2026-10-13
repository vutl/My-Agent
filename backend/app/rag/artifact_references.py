"""Printed artifact references, independent of runtime IDs and array positions."""
from dataclasses import dataclass
import re


# Vietnamese "mô hình" means "model", not "figure"; "mô hình 2" is model 2.
_KIND = (
    r"(?:fig(?:ure)?s?\.?|(?<!mô\s)(?<!mo\s)hình|biểu\s+đồ|sơ\s+đồ|đồ\s+thị|charts?|exhibits?"
    r"|tables?|tab\.|bảng)"
)
# Roman labels are printed upper-case (TABLE IV). Lower-case or other
# letters are ordinary words ("mix", "V-JEPA", appendix "Table C").
_NUMBER = (
    r"(?:[1-9]\d{0,3}(?:\.\d{1,3})*(?:[a-h](?!\w)|\([a-h]\))?"
    r"|[a-z]\.?\d{1,3}"
    r"|(?-i:[IVX]{1,6})(?![-\w]))"
)
_REFERENCE = re.compile(
    rf"(?<!\w)(?P<kind>{_KIND})\s*(?:#|no\.?\s*|số\s*)?(?P<number>{_NUMBER})(?!\w)", re.I,
)
_CONTINUATION = re.compile(
    rf"\s*(?P<join>,|&|and|và|với|to|through|[-–])\s*"
    rf"(?:(?P<kind>{_KIND})\s*)?(?P<number>{_NUMBER})(?!\w)", re.I,
)
_SUBFIGURE = re.compile(r"(?:[a-h]|\([a-h]\))$", re.I)
_ROMAN = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
          (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
          (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))


def _number(value: str) -> str | None:
    # Sub-panels (Fig. 2a, Figure 3(b)) resolve to their parent artifact.
    if value[:1].isdigit():
        value = _SUBFIGURE.sub("", value)
    value = re.sub(r"^([A-Za-z])\.(\d)", r"\1\2", value)  # Table A.2 -> A2
    upper = value.upper()
    if re.fullmatch(r"[IVXLCDM]+", upper):
        remaining = upper
        total = 0
        for number, glyph in _ROMAN:
            while remaining.startswith(glyph):
                total += number
                remaining = remaining[len(glyph):]
        if remaining or not 0 < total < 4000:
            return None
        # Reject noncanonical Roman strings rather than guessing e.g. IIV.
        check, amount = "", total
        for number, glyph in _ROMAN:
            count, amount = divmod(amount, number)
            check += glyph * count
        return str(total) if check == upper else None
    return upper


def _kind(value: str) -> str:
    return "table" if value.casefold().startswith(("table", "tab.", "bảng", "exhibit")) else "figure"


@dataclass(frozen=True)
class ArtifactReference:
    kind: str
    number: str


def extract_artifact_references(text: str) -> tuple[ArtifactReference, ...]:
    """Parse authored labels, not unresolved TABREF/FIGREF tokens.

    Arabic, canonical Roman, appendix and hierarchical labels retain printed
    identity. Bounded plural/range expansion never infers an artifact's index.
    """
    references: list[ArtifactReference] = []

    def add(kind, number):
        reference = ArtifactReference(kind, number)
        if reference not in references:
            references.append(reference)

    for match in _REFERENCE.finditer(text):
        kind = _kind(match["kind"])
        previous = _number(match["number"])
        if previous is None:
            continue
        add(kind, previous)
        position = match.end()
        for _ in range(20):
            following = _CONTINUATION.match(text, position)
            if following is None or (following["kind"] and _kind(following["kind"]) != kind):
                break
            number = _number(following["number"])
            if number is None:
                break
            if not following["kind"] and re.fullmatch(r"(?:19|20)\d\d", number):
                break  # "Table 2 and 2023 filings": a year, not Table 2023
            if following["join"].casefold() in {"to", "through", "-", "–"}:
                if not (previous.isdigit() and number.isdigit() and 0 < int(number) - int(previous) <= 20):
                    break
                for value in range(int(previous) + 1, int(number) + 1):
                    add(kind, str(value))
            else:
                add(kind, number)
            previous, position = number, following.end()
    return tuple(references)
