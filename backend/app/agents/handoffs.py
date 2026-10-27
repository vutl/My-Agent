"""Control-only tools advertised on router-bypass turns, never task steps."""


def router_handoff_tools(library_titles: list[tuple[str, str]]) -> list[dict]:
    titles = "; ".join(" ".join(title.split())[:180] for _, title in library_titles[:60])
    return [
        {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": {argument: {"type": "string"}},
                           "required": [argument], "additionalProperties": False},
        }}
        for name, argument, description in (
            ("ask_library", "question",
             "Use when the user asks about the content of their documents / library, or about a topic "
             f"that one of these library titles covers: {titles or '(empty library)'}. "
             "Never answer such questions from your own knowledge. This hands off the original request to library retrieval; "
             "call it before any other tools. Handoffs stay available after read-only lookups (search, fetch, read) "
             "but not after an action that changed something."),
            ("plan_task", "goal",
             "Use when the request needs at least 3 actions, at least 2 different tools, or an explicit sequence. "
             "This hands off the original request to the task planner; call it before any other tools. "
             "Handoffs stay available after read-only lookups but not after an action that changed something."),
        )
    ]
