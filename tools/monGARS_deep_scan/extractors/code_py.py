from __future__ import annotations

import ast
from pathlib import Path
from typing import List

from ..utils.text_clean import find_dialog_blocks, split_paragraphs, strip_code_fences
from .types import ExtractionRecord

_USER_ROLES = {"user", "client", "utilisateur", "moi", "tu", "vous"}


def _docstring_node(node: ast.AST) -> ast.AST | None:
    body = getattr(node, "body", None)
    if body and isinstance(body, list) and body:
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            return first
    return None


def _extract_docstrings(tree: ast.AST) -> List[tuple[str, int, int, str]]:
    results: List[tuple[str, int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            docstring = ast.get_docstring(node, clean=False)
            if not docstring:
                continue
            doc_node = _docstring_node(node) or node
            start = getattr(doc_node, "lineno", 1)
            end = getattr(doc_node, "end_lineno", start)
            label = "python_docstring"
            if isinstance(node, ast.ClassDef):
                label = "python_class_docstring"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                label = "python_function_docstring"
            elif isinstance(node, ast.Module):
                label = "python_module_docstring"
            results.append((docstring, start, end, label))
    return results


def _extract_prompt_strings(tree: ast.AST) -> List[tuple[str, int, int]]:
    prompts: List[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            if not targets:
                continue
            name = targets[0].id.lower()
            if any(keyword in name for keyword in ("prompt", "template", "dialog")):
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    start = getattr(value, "lineno", getattr(node, "lineno", 1))
                    end = getattr(value, "end_lineno", start)
                    prompts.append((value.value, start, end))
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name):
                name = target.id.lower()
                if any(keyword in name for keyword in ("prompt", "template", "dialog")):
                    value = node.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        start = getattr(value, "lineno", getattr(node, "lineno", 1))
                        end = getattr(value, "end_lineno", start)
                        prompts.append((value.value, start, end))
    return prompts


def extract(path: Path, text: str) -> List[ExtractionRecord]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    records: List[ExtractionRecord] = []

    for content, start, end, label in _extract_docstrings(tree):
        content = content.strip()
        if not content:
            continue
        dialog_blocks = find_dialog_blocks(content.splitlines())
        if dialog_blocks:
            for block in dialog_blocks:
                user_lines = [
                    line["content"]
                    for line in block["lines"]
                    if line["role"] in _USER_ROLES
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
                            start_line=start,
                            end_line=end,
                            type_label=f"{label}_dialog",
                        )
                    )
        paragraphs = split_paragraphs(strip_code_fences(content))
        for paragraph, para_start, para_end in paragraphs:
            absolute_start = start + para_start - 1
            absolute_end = start + para_end - 1
            records.append(
                ExtractionRecord.for_embedding(
                    text=paragraph,
                    source_file=str(path),
                    start_line=absolute_start,
                    end_line=absolute_end,
                    type_label=label,
                )
            )

    for prompt_text, start, end in _extract_prompt_strings(tree):
        cleaned = prompt_text.strip()
        if not cleaned:
            continue
        dialog_blocks = find_dialog_blocks(cleaned.splitlines())
        if dialog_blocks:
            for block in dialog_blocks:
                user_lines = [
                    line["content"]
                    for line in block["lines"]
                    if line["role"] in _USER_ROLES
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
                            start_line=start,
                            end_line=end,
                            type_label="python_prompt_dialog",
                        )
                    )
        paragraphs = split_paragraphs(strip_code_fences(cleaned))
        for paragraph, para_start, para_end in paragraphs:
            absolute_start = start + para_start - 1
            absolute_end = start + para_end - 1
            records.append(
                ExtractionRecord.for_embedding(
                    text=paragraph,
                    source_file=str(path),
                    start_line=absolute_start,
                    end_line=absolute_end,
                    type_label="python_prompt",
                )
            )

    return records
