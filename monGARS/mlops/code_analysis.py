"""Static analysis helpers for locating LLM integrations in the codebase."""

from __future__ import annotations

import ast
import dataclasses
import logging
import warnings
from pathlib import Path
from textwrap import indent
from typing import Iterable, Iterator, Literal, Sequence

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class LLMUsage:
    """Description of an LLM-oriented callsite discovered in the tree."""

    file_path: Path
    line: int
    symbol: str
    framework: str
    call: str
    snippet: str

    @property
    def module(self) -> str:
        return ".".join(self.symbol.split("::")[0].split("."))


def _iter_python_files(root: Path, *, ignore: Sequence[str]) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if any(part in ignore for part in path.parts):
            continue
        yield path


class _Analyzer(ast.NodeVisitor):
    """AST visitor that records LLM callsites along with their scope."""

    def __init__(self, source: str, path: Path, frameworks: Sequence[str]) -> None:
        self._source = source.splitlines()
        self._path = path
        self._frameworks = tuple(frameworks)
        self._import_aliases: dict[str, str] = {}
        self._direct_imports: dict[str, str] = {}
        self._scope: list[str] = []
        self.usages: list[LLMUsage] = []

    # -- scope helpers -------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast API
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    # -- import bookkeeping -------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            name = alias.name.split(".")[0]
            asname = alias.asname or name
            if name in self._frameworks:
                self._import_aliases[asname] = name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = (node.module or "").split(".")[0]
        if module not in self._frameworks:
            self.generic_visit(node)
            return
        for alias in node.names:
            target = alias.asname or alias.name
            self._direct_imports[target] = module
        self.generic_visit(node)

    # -- callsite capture ----------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        framework = self._resolve_framework(node.func)
        if framework:
            symbol = self._resolve_symbol()
            call = self._format_call(node.func)
            snippet = self._extract_snippet(node.lineno)
            usage = LLMUsage(
                file_path=self._path,
                line=node.lineno,
                symbol=symbol,
                framework=framework,
                call=call,
                snippet=snippet,
            )
            self.usages.append(usage)
        self.generic_visit(node)

    # -- helpers -------------------------------------------------------
    def _resolve_symbol(self) -> str:
        if not self._scope:
            return "<module>"
        return "::".join(self._scope)

    def _resolve_framework(self, func: ast.AST) -> str | None:
        root = self._root_name(func)
        if root is None:
            return None
        if root in self._import_aliases:
            return self._import_aliases[root]
        if root in self._direct_imports:
            return self._direct_imports[root]
        return None

    def _root_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._root_name(node.value)
        if isinstance(node, ast.Call):
            return self._root_name(node.func)
        return None

    def _format_call(self, node: ast.AST) -> str:
        if isinstance(node, ast.Attribute):
            return f"{self._format_call(node.value)}.{node.attr}"
        if isinstance(node, ast.Name):
            return node.id
        return ast.unparse(node) if hasattr(ast, "unparse") else "<expr>"

    def _extract_snippet(self, line: int, context: int = 2) -> str:
        start = max(0, line - 1 - context)
        end = min(len(self._source), line + context)
        segment = self._source[start:end]
        numbered = [
            f"{start + idx + 1:04d}: {text}" for idx, text in enumerate(segment)
        ]
        return "\n".join(numbered)


_DEFAULT_FRAMEWORKS = (
    "transformers",
    "torch",
    "unsloth",
    "llm2vec",
    "peft",
)
_IGNORE_PARTS = (".venv", "tests", "build", "dist", "__pycache__")


def scan_llm_usage(
    root: Path,
    *,
    frameworks: Sequence[str] = _DEFAULT_FRAMEWORKS,
    ignore_parts: Sequence[str] = _IGNORE_PARTS,
) -> list[LLMUsage]:
    """Return all LLM callsites beneath ``root`` for the supplied frameworks."""

    usages: list[LLMUsage] = []
    for path in _iter_python_files(root, ignore=ignore_parts):
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            logger.debug("Skipping non-UTF8 file: %s", path, exc_info=True)
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source)
        except SyntaxError:
            logger.debug("Skipping unparsable file: %s", path, exc_info=True)
            continue
        analyzer = _Analyzer(source, path, frameworks)
        analyzer.visit(tree)
        usages.extend(analyzer.usages)
    usages.sort(key=lambda usage: (usage.file_path, usage.line))
    return usages


def render_usage_report(usages: Iterable[LLMUsage]) -> str:
    """Render a Markdown report summarising discovered callsites."""

    lines = ["# monGARS LLM Integration Report", ""]
    grouped: dict[str, list[LLMUsage]] = {}
    for usage in usages:
        grouped.setdefault(usage.framework, []).append(usage)
    if not grouped:
        lines.append("No LLM callsites detected.")
        return "\n".join(lines)
    for framework in sorted(grouped):
        entries = grouped[framework]
        lines.append(f"## {framework} ({len(entries)})")
        lines.append("")
        for entry in entries:
            lines.append(
                f"- `{entry.symbol}` — `{entry.call}` @ `{entry.file_path}:{entry.line}`"
            )
            lines.append("")
            lines.append("```python")
            lines.append(indent(entry.snippet, ""))
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def build_strategy_recommendation(usage: LLMUsage) -> str:
    """Generate a deterministic fine-tuning suggestion for a callsite."""

    base = (
        f"Focus on `{usage.symbol}` which relies on `{usage.call}` from the {usage.framework} stack. "
        f"Ensure the workflow in `{usage.file_path}` remains deterministic and well-tested."
    )
    if usage.framework == "transformers":
        extra = (
            " Use QLoRA adapters with unsloth-backed 4-bit loaders, keep gradient checkpointing enabled,"
            " and export adapters alongside tokenizer artefacts."
        )
    elif usage.framework == "unsloth":
        extra = (
            " Prime `FastLanguageModel` early so attention rewiring happens before model loading,"
            " then validate 4-bit quantisation savings with embedding smoke tests."
        )
    elif usage.framework == "llm2vec":
        extra = (
            " Provide pooled embeddings using mean reduction and assert hidden-state availability"
            " before constructing the wrapper."
        )
    elif usage.framework == "peft":
        extra = " Ensure LoRA target modules cover projection layers and persist adapter config for reuse."
    else:
        extra = (
            " Optimise tensor allocations, respect VRAM budgets, and propagate informative telemetry"
            " back into the evolution engine."
        )
    return base + extra


@dataclasses.dataclass(frozen=True, slots=True)
class ModuleInteraction:
    """Represents a dependency edge between two Python modules."""

    source_path: Path
    line: int
    source_module: str
    target_module: str
    import_names: tuple[str, ...]
    kind: Literal["import", "from"]
    snippet: str

    @property
    def description(self) -> str:
        target = f"{self.target_module} ({', '.join(self.import_names)})".strip()
        return f"{self.source_module} -> {target}"


def _module_name_from_path(root: Path, path: Path) -> tuple[str, list[str], list[str]]:
    rel = path.relative_to(root)
    parts = list(rel.with_suffix("").parts)
    is_package = bool(parts and parts[-1] == "__init__")
    if is_package:
        parts = parts[:-1]
    module_name = ".".join(parts) if parts else path.stem
    module_parts = module_name.split(".") if module_name else []
    if is_package:
        anchor = module_parts
    else:
        anchor = module_parts[:-1]
    return module_name, module_parts, anchor


def _resolve_relative_module(
    anchor: Sequence[str], level: int, module: str | None
) -> str | None:
    if level == 0:
        base_parts = list(anchor)
    else:
        if level - 1 > len(anchor):
            return None
        cutoff = len(anchor) - (level - 1)
        base_parts = list(anchor[:cutoff])
    if module:
        base_parts.extend(module.split("."))
    target = ".".join(part for part in base_parts if part)
    return target or None


def _extract_snippet_from_lines(
    lines: Sequence[str], line: int, context: int = 2
) -> str:
    start = max(0, line - 1 - context)
    end = min(len(lines), line + context)
    segment = lines[start:end]
    numbered = [f"{start + idx + 1:04d}: {text}" for idx, text in enumerate(segment)]
    return "\n".join(numbered)


class _ModuleGraphVisitor(ast.NodeVisitor):
    def __init__(
        self,
        source_lines: Sequence[str],
        path: Path,
        module_name: str,
        anchor: Sequence[str],
        packages: Sequence[str],
    ) -> None:
        self._lines = source_lines
        self._path = path
        self._module_name = module_name
        self._anchor = tuple(anchor)
        self._packages = tuple(packages)
        self.interactions: list[ModuleInteraction] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast API
        for alias in node.names:
            target = alias.name
            if not self._matches_package(target):
                continue
            bound = alias.asname or alias.name.split(".")[-1]
            snippet = _extract_snippet_from_lines(self._lines, node.lineno)
            self.interactions.append(
                ModuleInteraction(
                    source_path=self._path,
                    line=node.lineno,
                    source_module=self._module_name,
                    target_module=target,
                    import_names=(bound,),
                    kind="import",
                    snippet=snippet,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - ast API
        target = node.module
        if node.level:
            target = _resolve_relative_module(self._anchor, node.level, target)
        if not target or not self._matches_package(target):
            self.generic_visit(node)
            return
        names = tuple(alias.asname or alias.name for alias in node.names)
        snippet = _extract_snippet_from_lines(self._lines, node.lineno)
        self.interactions.append(
            ModuleInteraction(
                source_path=self._path,
                line=node.lineno,
                source_module=self._module_name,
                target_module=target,
                import_names=names,
                kind="from",
                snippet=snippet,
            )
        )
        self.generic_visit(node)

    def _matches_package(self, module: str) -> bool:
        return any(
            module == package or module.startswith(f"{package}.")
            for package in self._packages
        )


def scan_module_interactions(
    root: Path,
    *,
    packages: Sequence[str] = ("monGARS", "modules"),
    ignore_parts: Sequence[str] = _IGNORE_PARTS,
) -> list[ModuleInteraction]:
    """Discover intra-repository dependencies between Python modules."""

    interactions: list[ModuleInteraction] = []
    for path in _iter_python_files(root, ignore=ignore_parts):
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            logger.debug("Skipping non-UTF8 file: %s", path, exc_info=True)
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source)
        except SyntaxError:
            logger.debug("Skipping unparsable file: %s", path, exc_info=True)
            continue
        module_name, _, anchor = _module_name_from_path(root, path)
        visitor = _ModuleGraphVisitor(
            source.splitlines(), path, module_name, anchor, packages
        )
        visitor.visit(tree)
        interactions.extend(visitor.interactions)
    interactions.sort(
        key=lambda item: (item.source_path, item.line, item.target_module)
    )
    return interactions


__all__ = [
    "LLMUsage",
    "ModuleInteraction",
    "build_strategy_recommendation",
    "scan_module_interactions",
    "render_usage_report",
    "scan_llm_usage",
]
