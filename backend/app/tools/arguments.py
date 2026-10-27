"""Conservative JSON argument repair and the schema subset used by Aya tools."""

import json
import math
from typing import Any


def _invalid_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def parse_arguments(raw: str) -> tuple[dict[str, Any] | None, bool]:
    """Keep a complete leading object; never guess missing syntax or values."""
    decoder = json.JSONDecoder(parse_constant=_invalid_constant)
    try:
        text = (raw or "{}").lstrip()
        value, end = decoder.raw_decode(text)
        if isinstance(value, dict):
            return value, bool(text[end:].strip())
    except (TypeError, ValueError, AttributeError):
        pass
    return None, False


def validate_arguments(value: Any, schema: dict[str, Any], path: str = "") -> str | None:
    """Return the first actionable problem, without coercing model input."""
    expected = schema.get("type")
    matches = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "boolean": type(value) is bool,
        "integer": type(value) is int or (type(value) is float and math.isfinite(value) and value.is_integer()),
        "number": type(value) is int or (type(value) is float and math.isfinite(value)), "null": value is None,
    }
    location = f'"{path}"' if path else "arguments"
    if expected and not matches.get(expected, False):
        return f"{location} must be {expected}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{location} must be one of {json.dumps(schema['enum'], ensure_ascii=False)}"
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                return f'missing "{path + "." if path else ""}{key}"'
        properties = schema.get("properties", {})
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else key
            if key in properties:
                error = validate_arguments(item, properties[key], child_path)
                if error:
                    return error
            elif schema.get("additionalProperties") is False:
                return f'unknown "{child_path}"'
    elif isinstance(value, list):
        for bound, compare in (("minItems", len(value) < schema.get("minItems", 0)),
                               ("maxItems", len(value) > schema.get("maxItems", math.inf))):
            if compare:
                return f"{location} violates {bound}={schema[bound]}"
        for index, item in enumerate(value):
            error = validate_arguments(item, schema.get("items", {}), f"{path}[{index}]")
            if error:
                return error
    elif type(value) in {int, float}:
        if "minimum" in schema and value < schema["minimum"]:
            return f"{location} must be >= {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{location} must be <= {schema['maximum']}"
    return None
