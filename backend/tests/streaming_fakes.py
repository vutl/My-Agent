"""Small streaming fixtures shared by agent route regressions."""


async def tool_turn_events(turn):
    for offset in range(0, len(turn.content), 5):
        yield "text", turn.content[offset:offset + 5]
    for index, call in enumerate(turn.tool_calls):
        yield "tool_call_start", {"index": index, "id": call.id, "name": call.name}
        yield "tool_call_args", {"index": index, "delta": call.arguments}
    yield "turn", turn
