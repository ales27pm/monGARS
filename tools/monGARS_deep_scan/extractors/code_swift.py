from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

from ..utils.text_clean import find_dialog_blocks, split_paragraphs
from .types import ExtractionRecord

_USER_ROLES = {"user", "client", "utilisateur", "moi", "tu", "vous"}

_BLOCK_DOC_PATTERN = re.compile(r"/\*\*(?P<body>[\s\S]*?)\*/", re.MULTILINE)
_TRIPLE_STRING_PATTERN = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:prompt|template|instruction|system)"
    r"[A-Za-z0-9_]*|(?:prompt|template|instruction|system)[A-Za-z0-9_]*)"
    r"\s*(?::[^=\n]+)?=\s*\"\"\"(?P<body>[\s\S]*?)\"\"\"",
    re.IGNORECASE | re.MULTILINE,
)
_TOOL_DEFINITION_PATTERN = re.compile(
    r"ToolDefinition\s*\(\s*"
    r'id:\s*"(?P<id>(?:[^"\\]|\\.)+)"\s*,\s*'
    r'name:\s*"(?P<name>(?:[^"\\]|\\.)+)"\s*,[\s\S]*?'
    r'description:\s*"(?P<description>(?:[^"\\]|\\.)+)"\s*,[\s\S]*?'
    r"requiresApproval:\s*(?P<approval>true|false)",
    re.MULTILINE,
)
_COMPACT_TOOL_HEADER_PATTERN = re.compile(
    r'^tool\s*\(\s*"(?P<id>(?:[^"\\]|\\.)+)"\s*,\s*'
    r'"(?P<name>(?:[^"\\]|\\.)+)"\s*,\s*'
    r'"(?P<description>(?:[^"\\]|\\.)+)"',
    re.MULTILINE,
)
_COMPACT_TOOL_POLICY_PATTERN = re.compile(
    r",\s*\.(?P<risk>low|moderate|high|critical)\s*,\s*"
    r"(?P<approval>true|false)\s*,\s*(?P<background>true|false)"
    r"(?:\s*,\s*[0-9_]+)?\s*\)$",
    re.MULTILINE,
)
_LITERAL_COMPACT_TOOL_CALL_PATTERN = re.compile(r'\btool\s*\(\s*"(?:[^"\\]|\\.)*"')

_TOOL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._][a-z0-9]+)*$")
_ENUM_TOKEN_PATTERN = re.compile(r"^\.([A-Za-z_][A-Za-z0-9_]*)$")
_INTEGER_PATTERN = re.compile(r"^[0-9][0-9_]*$")
_ARGUMENT_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
_TOOL_CATEGORIES = {
    "productivity",
    "communication",
    "location",
    "media",
    "health",
    "knowledge",
}
_TOOL_PERMISSIONS = {
    "calendar",
    "reminders",
    "contacts",
    "location",
    "photos",
    "camera",
    "health",
    "motion",
    "alarms",
    "notifications",
}
_TOOL_RISKS = {"low", "moderate", "high", "critical"}
_ARGUMENT_TYPES = {
    "string": "string",
    "number": "number",
    "boolean": "bool",
    "array": "array",
    "object": "object",
    "enumeration": "enum",
}
_JSON_SCHEMA_TYPES = {
    "string": "string",
    "number": "number",
    "bool": "boolean",
    "array": "array",
    "object": "object",
    "enum": "string",
}


class SwiftToolExtractionError(ValueError):
    """Raised when a canonical Swift tool declaration cannot be decoded."""


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _swift_unescape(value: str) -> str:
    """Decode the small escape subset used in catalog string literals."""

    return (
        value.replace(r"\n", "\n")
        .replace(r"\t", "\t")
        .replace(r"\"", '"')
        .replace(r"\\", "\\")
    )


def _split_top_level(value: str) -> list[str]:
    """Split a Swift argument list without splitting nested literals."""

    parts: list[str] = []
    start = 0
    round_depth = 0
    square_depth = 0
    brace_depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(value):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "(":
            round_depth += 1
        elif character == ")":
            round_depth -= 1
        elif character == "[":
            square_depth += 1
        elif character == "]":
            square_depth -= 1
        elif character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth -= 1
        elif (
            character == ","
            and round_depth == 0
            and square_depth == 0
            and brace_depth == 0
        ):
            parts.append(value[start:index].strip())
            start = index + 1
        if min(round_depth, square_depth, brace_depth) < 0:
            raise SwiftToolExtractionError("unbalanced Swift expression")
    if in_string or round_depth or square_depth or brace_depth:
        raise SwiftToolExtractionError("unterminated Swift expression")
    parts.append(value[start:].strip())
    return parts


def _parse_string_literal(token: str) -> str:
    token = token.strip()
    if not (token.startswith('"') and token.endswith('"')):
        raise SwiftToolExtractionError(f"expected string literal, got {token!r}")
    try:
        decoded = json.loads(token)
    except json.JSONDecodeError as exc:
        raise SwiftToolExtractionError(
            f"invalid Swift string literal: {token!r}"
        ) from exc
    if not isinstance(decoded, str):
        raise SwiftToolExtractionError("decoded Swift literal is not a string")
    return decoded


def _parse_enum_token(token: str, *, label: str) -> str:
    match = _ENUM_TOKEN_PATTERN.fullmatch(token.strip())
    if match is None:
        raise SwiftToolExtractionError(f"expected {label} enum token, got {token!r}")
    return match.group(1)


def _parse_bool(token: str, *, label: str) -> bool:
    value = token.strip()
    if value not in {"true", "false"}:
        raise SwiftToolExtractionError(f"expected {label} boolean, got {token!r}")
    return value == "true"


def _parse_string_array(token: str) -> list[str]:
    value = token.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise SwiftToolExtractionError(f"expected string array, got {token!r}")
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [_parse_string_literal(part) for part in _split_top_level(inner)]


def _parse_argument_call(token: str) -> dict:
    value = token.strip()
    if not (value.startswith("arg(") and value.endswith(")")):
        raise SwiftToolExtractionError(f"expected arg(...) declaration, got {token!r}")
    parts = _split_top_level(value[4:-1])
    if not parts or not parts[0]:
        raise SwiftToolExtractionError("tool argument is missing a name")

    name = _parse_string_literal(parts[0])
    if _ARGUMENT_NAME_PATTERN.fullmatch(name) is None:
        raise SwiftToolExtractionError(f"invalid tool argument name {name!r}")
    argument_type = "string"
    required = True
    allowed_values: list[str] | None = None
    positional_type_seen = False
    seen_labels: set[str] = set()
    for part in parts[1:]:
        if ":" not in part:
            if positional_type_seen:
                raise SwiftToolExtractionError(
                    f"tool argument {name!r} has multiple positional types"
                )
            raw_type = _parse_enum_token(part, label="argument type")
            if raw_type not in _ARGUMENT_TYPES:
                raise SwiftToolExtractionError(
                    f"tool argument {name!r} has unknown type {raw_type!r}"
                )
            argument_type = _ARGUMENT_TYPES[raw_type]
            positional_type_seen = True
            continue
        label, raw_value = (piece.strip() for piece in part.split(":", 1))
        if label in seen_labels:
            raise SwiftToolExtractionError(
                f"tool argument {name!r} repeats label {label!r}"
            )
        seen_labels.add(label)
        if label == "required":
            required = _parse_bool(raw_value, label="required")
        elif label == "allowed":
            parsed_values = _parse_string_array(raw_value)
            if any(not item for item in parsed_values):
                raise SwiftToolExtractionError(
                    f"tool argument {name!r} has an empty allowed value"
                )
            if len(parsed_values) != len(set(parsed_values)):
                raise SwiftToolExtractionError(
                    f"tool argument {name!r} repeats an allowed value"
                )
            allowed_values = sorted(parsed_values)
        else:
            raise SwiftToolExtractionError(
                f"tool argument {name!r} has unknown label {label!r}"
            )

    if argument_type == "enum" and not allowed_values:
        raise SwiftToolExtractionError(
            f"enumeration argument {name!r} must declare allowed values"
        )
    if allowed_values is not None and argument_type != "enum":
        raise SwiftToolExtractionError(
            f"non-enumeration argument {name!r} declares allowed values"
        )
    return {
        "name": name,
        "type": argument_type,
        "required": required,
        "allowed_values": allowed_values,
    }


def _parse_argument_array(token: str) -> list[dict]:
    value = token.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise SwiftToolExtractionError(f"expected argument array, got {token!r}")
    inner = value[1:-1].strip()
    if not inner:
        return []
    arguments = [_parse_argument_call(part) for part in _split_top_level(inner)]
    names = [argument["name"] for argument in arguments]
    if len(names) != len(set(names)):
        raise SwiftToolExtractionError(
            "tool declaration contains duplicate argument names"
        )
    return arguments


def _json_schema(arguments: list[dict]) -> dict:
    properties: dict[str, dict] = {}
    required: list[str] = []
    for argument in arguments:
        schema = {"type": _JSON_SCHEMA_TYPES[argument["type"]]}
        if argument["allowed_values"] is not None:
            schema["enum"] = argument["allowed_values"]
        properties[argument["name"]] = schema
        if argument["required"]:
            required.append(argument["name"])
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _parse_compact_tool_call(call: str) -> dict:
    open_paren = call.find("(")
    if open_paren < 0 or not call.endswith(")"):
        raise SwiftToolExtractionError("invalid tool(...) expression")
    parts = _split_top_level(call[open_paren + 1 : -1])
    if len(parts) not in {9, 10}:
        raise SwiftToolExtractionError(
            f"tool(...) expects 9 or 10 arguments, found {len(parts)}"
        )

    tool_id = _parse_string_literal(parts[0])
    if _TOOL_ID_PATTERN.fullmatch(tool_id) is None:
        raise SwiftToolExtractionError(f"invalid canonical tool ID {tool_id!r}")
    arguments = _parse_argument_array(parts[4])
    category = _parse_enum_token(parts[3], label="category")
    if category not in _TOOL_CATEGORIES:
        raise SwiftToolExtractionError(f"unknown tool category {category!r}")
    permission = (
        None
        if parts[5].strip() == "nil"
        else _parse_enum_token(parts[5], label="permission")
    )
    if permission is not None and permission not in _TOOL_PERMISSIONS:
        raise SwiftToolExtractionError(f"unknown tool permission {permission!r}")
    risk = _parse_enum_token(parts[6], label="risk")
    if risk not in _TOOL_RISKS:
        raise SwiftToolExtractionError(f"unknown tool risk {risk!r}")
    max_output = 2_400
    if len(parts) == 10:
        raw_max = parts[9].strip()
        if _INTEGER_PATTERN.fullmatch(raw_max) is None:
            raise SwiftToolExtractionError(
                f"invalid maximum output character count {raw_max!r}"
            )
        max_output = max(256, int(raw_max.replace("_", "")))

    display_name = _parse_string_literal(parts[1])
    description = _parse_string_literal(parts[2])
    if not display_name.strip() or not description.strip():
        raise SwiftToolExtractionError(
            f"tool {tool_id!r} must have a display name and description"
        )
    supports_background = _parse_bool(parts[8], label="supportsBackgroundExecution")
    return {
        "id": tool_id,
        # Keep the legacy `name`/`supports_background` keys for consumers of
        # the original scanner while exposing the complete native contract.
        "name": display_name,
        "display_name": display_name,
        "description": description,
        "category": category,
        "arguments": arguments,
        "permission": permission,
        "risk": risk,
        "requires_approval": _parse_bool(parts[7], label="requiresApproval"),
        "supports_background": supports_background,
        "supports_background_execution": supports_background,
        "maximum_output_characters": max_output,
        "json_schema": _json_schema(arguments),
        "source_language": "swift",
    }


def _clean_doc_block(body: str) -> str:
    cleaned: list[str] = []
    for line in body.splitlines():
        value = re.sub(r"^\s*\* ?", "", line).rstrip()
        cleaned.append(value)
    return "\n".join(cleaned).strip()


def _line_doc_blocks(text: str) -> Iterable[tuple[str, int, int]]:
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if not re.match(r"^\s*///", lines[index]):
            index += 1
            continue
        start = index + 1
        content: list[str] = []
        while index < len(lines):
            match = re.match(r"^\s*///\s?(.*)$", lines[index])
            if match is None:
                break
            content.append(match.group(1).rstrip())
            index += 1
        value = "\n".join(content).strip()
        if value:
            yield value, start, index


def _balanced_tool_calls(text: str) -> Iterable[tuple[str, int, int, int]]:
    """Yield compact `tool(...)` calls while respecting strings and nesting."""

    for match in re.finditer(r"\btool\s*\(", text):
        index = match.end() - 1
        depth = 0
        in_string = False
        escaped = False
        while index < len(text):
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    yield (
                        text[match.start() : end],
                        _line_number(text, match.start()),
                        _line_number(text, end),
                        match.start(),
                    )
                    break
            index += 1


def _content_records(
    *,
    path: Path,
    content: str,
    start_line: int,
    end_line: int,
    type_label: str,
) -> List[ExtractionRecord]:
    records: List[ExtractionRecord] = []
    dialog_blocks = find_dialog_blocks(content.splitlines())
    for block in dialog_blocks:
        user_lines = [
            line["content"] for line in block["lines"] if line["role"] in _USER_ROLES
        ]
        assistant_lines = [
            line["content"]
            for line in block["lines"]
            if line["role"] not in _USER_ROLES
        ]
        if user_lines and assistant_lines:
            records.append(
                ExtractionRecord.for_sft(
                    instruction="\n".join(user_lines),
                    output="\n".join(assistant_lines),
                    source_file=str(path),
                    start_line=start_line + block["start_line"] - 1,
                    end_line=start_line + block["end_line"] - 1,
                    type_label=f"{type_label}_dialog",
                )
            )

    for paragraph, paragraph_start, paragraph_end in split_paragraphs(content):
        records.append(
            ExtractionRecord.for_embedding(
                text=paragraph,
                source_file=str(path),
                start_line=start_line + paragraph_start - 1,
                end_line=min(end_line, start_line + paragraph_end - 1),
                type_label=type_label,
            )
        )
    return records


def extract_agent_tool_definitions(
    path: Path, text: str, *, strict: bool = False
) -> List[ExtractionRecord]:
    """Extract complete canonical ``tool(...)`` contracts from Swift.

    The generic deep scanner is intentionally tolerant and can skip an
    unrelated helper named ``tool``.  Manifest generation sets ``strict`` so
    any declaration that starts with a literal tool ID but cannot be decoded
    aborts the build instead of silently publishing a partial catalog.
    """

    records: List[ExtractionRecord] = []
    failures: list[str] = []
    literal_call_count = 0
    literal_starts = {
        match.start(): _line_number(text, match.start())
        for match in _LITERAL_COMPACT_TOOL_CALL_PATTERN.finditer(text)
    }
    balanced_literal_starts: set[int] = set()
    for call, start, end, call_offset in _balanced_tool_calls(text):
        if call_offset not in literal_starts:
            continue
        balanced_literal_starts.add(call_offset)
        literal_call_count += 1
        header = _COMPACT_TOOL_HEADER_PATTERN.search(call)
        if header is None:
            failures.append(
                f"{path}:{start}: canonical tool declarations must use literal "
                "ID, display name, and description values"
            )
            continue
        try:
            output = _parse_compact_tool_call(call)
        except SwiftToolExtractionError as exc:
            failures.append(f"{path}:{start}: {exc}")
            continue
        records.append(
            ExtractionRecord.for_agent(
                instruction=("Register the exact iOS tool schema " f"{output['id']}."),
                output=output,
                source_file=str(path),
                start_line=start,
                end_line=end,
                type_label="swift_agent_tool_definition",
            )
        )

    for offset in sorted(set(literal_starts) - balanced_literal_starts):
        failures.append(
            f"{path}:{literal_starts[offset]}: unterminated canonical tool declaration"
        )
        literal_call_count += 1

    if strict and failures:
        raise SwiftToolExtractionError("; ".join(failures))
    if strict and literal_call_count != len(records):
        raise SwiftToolExtractionError(
            "canonical tool extraction was incomplete: "
            f"found {literal_call_count} literal calls, decoded {len(records)}"
        )
    return records


def extract(path: Path, text: str) -> List[ExtractionRecord]:
    """Extract auditable training records from Swift without executing it."""

    records: List[ExtractionRecord] = []
    seen_content: set[tuple[str, int, str]] = set()

    for content, start, end in _line_doc_blocks(text):
        key = (content, start, "swift_doc_comment")
        if key in seen_content:
            continue
        seen_content.add(key)
        records.extend(
            _content_records(
                path=path,
                content=content,
                start_line=start,
                end_line=end,
                type_label="swift_doc_comment",
            )
        )

    for match in _BLOCK_DOC_PATTERN.finditer(text):
        content = _clean_doc_block(match.group("body"))
        if not content:
            continue
        start = _line_number(text, match.start())
        end = _line_number(text, match.end())
        key = (content, start, "swift_doc_comment")
        if key in seen_content:
            continue
        seen_content.add(key)
        records.extend(
            _content_records(
                path=path,
                content=content,
                start_line=start,
                end_line=end,
                type_label="swift_doc_comment",
            )
        )

    for match in _TRIPLE_STRING_PATTERN.finditer(text):
        content = match.group("body").strip()
        if not content:
            continue
        start = _line_number(text, match.start("body"))
        end = _line_number(text, match.end("body"))
        records.extend(
            _content_records(
                path=path,
                content=content,
                start_line=start,
                end_line=end,
                type_label="swift_prompt",
            )
        )

    for match in _TOOL_DEFINITION_PATTERN.finditer(text):
        start = _line_number(text, match.start())
        end = _line_number(text, match.end())
        tool_id = _swift_unescape(match.group("id"))
        records.append(
            ExtractionRecord.for_agent(
                instruction=f"Register the iOS tool capability {tool_id}.",
                output={
                    "id": tool_id,
                    "name": _swift_unescape(match.group("name")),
                    "description": _swift_unescape(match.group("description")),
                    "requires_approval": match.group("approval") == "true",
                    "source_language": "swift",
                },
                source_file=str(path),
                start_line=start,
                end_line=end,
                type_label="swift_tool_definition",
            )
        )

    records.extend(extract_agent_tool_definitions(path, text))

    return records
