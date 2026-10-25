"""Compact, user-facing capability summaries derived from registered tools."""

from __future__ import annotations

from functools import lru_cache
from weakref import WeakKeyDictionary

from app.tools.registry import ToolRegistry

_DESCRIPTIONS = {
    "library": {"vi": "Thư viện: tìm và đọc tài liệu đã lập chỉ mục.",
                "en": "Library: search and read indexed documents."},
    "web": {"vi": "Web: tìm thông tin mới và đọc trang web.",
            "en": "Web: search for current information and read web pages."},
    "files": {"vi": "File: tìm, đọc, tạo, sửa, di chuyển và bỏ file vào Thùng rác.",
              "en": "Files: find, read, create, edit, move and trash files."},
    "code": {"vi": "Code: chạy Python/lệnh trong sandbox, phân tích dữ liệu và vẽ biểu đồ.",
             "en": "Code: run sandboxed Python/commands, analyze data and create charts."},
    "reminders": {"vi": "Lời nhắc: đặt, xem và hủy nhắc việc, kể cả nhắc lặp lại.",
                  "en": "Reminders: set, list and cancel reminders, including recurring ones."},
    "apple": {"vi": "Apple: quản lý Reminders, xem/thêm sự kiện Calendar, tạo ghi chú Notes.",
              "en": "Apple: manage Reminders, view/add Calendar events and create Notes."},
    "apps": {"vi": "Ứng dụng: mở ứng dụng và liên kết trong trình duyệt.",
             "en": "Apps: open applications and links in the browser."},
    "music": {"vi": "Nhạc: tìm/phát nhạc và điều khiển Spotify.",
              "en": "Music: find/play music and control Spotify."},
}
_APPROVALS = {
    "vi": "Việc có rủi ro tôi sẽ hỏi cậu trước.",
    "en": "I will ask you before risky actions.",
}
_SHEETS: WeakKeyDictionary = WeakKeyDictionary()


def _language(language: str) -> str:
    return "en" if language.lower() in {"en", "english"} else "vi"


def capability_groups(registry: ToolRegistry) -> list[str]:
    return list(dict.fromkeys(registry.get(name).group for name in registry.names()))


def capability_sheet(registry: ToolRegistry, language: str = "vi") -> str:
    language = _language(language)
    tools = [registry.get(name) for name in registry.names()]
    # Registries can grow after construction; do not return a stale cached sheet.
    signature = tuple((tool.name, tool.group, tool.label) for tool in tools)
    cached = _SHEETS.setdefault(registry, {})
    if language in cached and cached[language][0] == signature:
        return cached[language][1]
    lines = []
    for group in capability_groups(registry):
        description = _DESCRIPTIONS.get(group, {}).get(language)
        if description is None:
            description = ", ".join(tool.label or tool.name for tool in tools if tool.group == group)
        lines.append(f"- {description}")
    sheet = "\n".join([*lines, _APPROVALS[language]])
    cached[language] = (signature, sheet)
    return sheet


@lru_cache(maxsize=1)
def _description_registry() -> ToolRegistry:
    from app.tools import build_registry

    return build_registry()


def capability_prompt(language: str = "vi", registry: ToolRegistry | None = None) -> str:
    """Shared prompt section; chat also knows the app's always-present library."""
    language = _language(language)
    describing_app = registry is None
    if registry is None:
        registry = _description_registry()
    sheet = capability_sheet(registry, language)
    if describing_app and "library" not in capability_groups(registry):
        # The executable registry only registers library search with a searcher.
        sheet = f"- {_DESCRIPTIONS['library'][language]}\n{sheet}"
    if language == "vi":
        return (
            f"Những việc Aya làm được:\n{sheet}\n"
            "Khi cậu hỏi tôi làm được gì hoặc nhờ làm việc thuộc danh sách này, trả lời dựa trên "
            "danh sách; không nói tôi không làm được việc đã liệt kê. Chỉ nói đã hoàn thành khi có kết quả công cụ."
        )
    return (
        f"What Aya can do:\n{sheet}\n"
        "When asked about your abilities or to do something on this list, answer from the list; "
        "never claim you cannot do a listed capability. Only claim completion after a tool result."
    )
