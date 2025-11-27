"""Utilities for generating and maintaining scoped ``AGENTS.md`` guides."""

# The module docstring is intentionally concise; CLI help strings provide
# extended usage instructions while keeping imports lightweight.

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "agents" / "agents_config.json"


class AgentsConfigError(RuntimeError):
    """Raised when the configuration cannot be processed."""


@dataclass
class Section:
    heading: str
    bullets: Sequence[str] = field(default_factory=list)
    paragraphs: Sequence[str] = field(default_factory=list)


@dataclass
class FileProfile:
    path: Path
    title: str
    scope: str
    dynamic_notes: Sequence[str]
    roadmap_focus: Sequence[Mapping[str, str]]
    sections: Sequence[Section]


def load_config(config_path: Path = CONFIG_PATH) -> Mapping[str, Any]:
    if not config_path.exists():
        raise AgentsConfigError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise AgentsConfigError(f"Invalid JSON configuration: {exc}") from exc


def parse_sections(raw_sections: Sequence[Mapping[str, Any]]) -> List[Section]:
    sections: List[Section] = []
    for entry in raw_sections:
        heading = entry.get("heading")
        if not heading:
            raise AgentsConfigError("Each section requires a heading")
        bullets = entry.get("bullets", [])
        paragraphs = entry.get("paragraphs", [])
        sections.append(
            Section(heading=heading, bullets=bullets, paragraphs=paragraphs)
        )
    return sections


def validate_dynamic_notes(raw_notes: Any, context: str) -> List[str]:
    if raw_notes is None:
        return []
    if not isinstance(raw_notes, list):
        raise AgentsConfigError(f"`{context}` must be a list when provided")

    for idx, note in enumerate(raw_notes):
        if not isinstance(note, str):
            raise AgentsConfigError(
                f"`{context}[{idx}]` must be a string; got {type(note).__name__}"
            )

    return raw_notes


def merge_dynamic_notes(
    base_notes: Sequence[str] | None, specific_notes: Sequence[str] | None
) -> List[str]:
    merged: List[str] = []
    for note in list(base_notes or []) + list(specific_notes or []):
        if note and note not in merged:
            merged.append(note)
    return merged


def load_profiles(config: Mapping[str, Any]) -> List[FileProfile]:
    files = config.get("files")
    if not isinstance(files, list):
        raise AgentsConfigError("Configuration must define a `files` list")

    defaults = config.get("defaults", {})
    if defaults is not None and not isinstance(defaults, Mapping):
        raise AgentsConfigError("`defaults` must be a mapping when provided")

    base_dynamic_notes = validate_dynamic_notes(
        defaults.get("dynamic_notes", []), "defaults.dynamic_notes"
    )

    profiles: List[FileProfile] = []
    for idx, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise AgentsConfigError(
                f"Each file entry must be a mapping; got {type(entry).__name__}"
            )
        path_value = entry.get("path")
        if not path_value:
            raise AgentsConfigError("Each file entry must include a `path`")
        title = entry.get("title")
        scope = entry.get("scope")
        file_dynamic_notes = validate_dynamic_notes(
            entry.get("dynamic_notes"), f"files[{idx}].dynamic_notes"
        )
        dynamic_notes = merge_dynamic_notes(base_dynamic_notes, file_dynamic_notes)
        roadmap_focus = entry.get("roadmap_focus", [])
        raw_sections = entry.get("sections", [])
        if not title or not scope:
            raise AgentsConfigError(
                f"File `{path_value}` is missing `title` or `scope` metadata"
            )
        profiles.append(
            FileProfile(
                path=Path(path_value),
                title=title,
                scope=scope,
                dynamic_notes=dynamic_notes,
                roadmap_focus=roadmap_focus,
                sections=parse_sections(raw_sections),
            )
        )
    return profiles


def parse_roadmap(roadmap_path: Path) -> Mapping[str, List[str]]:
    if not roadmap_path.exists():
        raise AgentsConfigError(f"Roadmap file not found: {roadmap_path}")
    phases: dict[str, List[str]] = {}
    current_phase: str | None = None
    current_entry: list[str] = []

    def flush_entry() -> None:
        nonlocal current_entry
        if current_phase and current_entry:
            phases.setdefault(current_phase, []).append(" ".join(current_entry).strip())
        current_entry = []

    with roadmap_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.startswith("## "):
                flush_entry()
                current_phase = raw_line.strip()[3:].strip()
                phases.setdefault(current_phase, [])
                continue

            if raw_line.startswith("- ") and current_phase:
                flush_entry()
                current_entry = [raw_line[2:].strip()]
                continue

            if raw_line.startswith("  ") and current_entry:
                current_entry.append(raw_line.strip())
                continue

            flush_entry()
    flush_entry()
    return phases


def format_paragraph(text: str, indent: int = 0) -> List[str]:
    wrapper = textwrap.TextWrapper(width=120, subsequent_indent=" " * indent)
    return wrapper.fill(text).splitlines() or [""]


def render_profile(profile: FileProfile, roadmap: Mapping[str, List[str]]) -> str:
    lines: List[str] = [f"# {profile.title}", ""]
    lines.append(
        "> ⚠️ Auto-generated by `scripts/manage_agents.py`. Update `configs/agents/agents_config.json` and rerun the script instead of editing this file manually."
    )
    lines.append("")
    lines.extend(["## Scope", ""])
    lines.extend(format_paragraph(profile.scope))
    lines.append("")

    if profile.dynamic_notes:
        lines.extend(["## Automation", ""])
        for note in profile.dynamic_notes:
            note_lines = format_paragraph(note, indent=2)
            if len(note_lines) == 1:
                lines.append(f"- {note_lines[0]}")
            else:
                lines.append(f"- {note_lines[0]}")
                for continuation in note_lines[1:]:
                    lines.append(f"  {continuation}")
        lines.append("")

    if profile.roadmap_focus:
        lines.extend(["## Roadmap Alignment", ""])
        for focus in profile.roadmap_focus:
            phase = focus.get("phase", "")
            label = focus.get("label", phase)
            phase_tasks = roadmap.get(phase)
            lines.append(f"- **{label}**")
            if not phase_tasks:
                lines.append(f"  - _(No matching roadmap entries for `{phase}`)_")
            else:
                for task in phase_tasks:
                    lines.append(f"  - {task}")
        lines.append("")

    for section in profile.sections:
        lines.extend([f"## {section.heading}", ""])
        for paragraph in section.paragraphs:
            lines.extend(format_paragraph(paragraph))
            lines.append("")
        if section.bullets:
            for bullet in section.bullets:
                bullet_lines = format_paragraph(bullet, indent=2)
                lines.append(f"- {bullet_lines[0]}")
                for continuation in bullet_lines[1:]:
                    lines.append(f"  {continuation}")
            lines.append("")
        if section.paragraphs:
            lines.append("")
    # Remove duplicate blank lines
    cleaned: List[str] = []
    for line in lines:
        if line == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(line)
    if cleaned and cleaned[-1] != "":
        cleaned.append("")
    return "\n".join(cleaned)


def write_file(target_path: Path, content: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def refresh_agents(paths: Sequence[str] | None = None) -> List[Path]:
    config = load_config()
    profiles = load_profiles(config)
    roadmap_path = REPO_ROOT / config.get("roadmap", {}).get("file", "ROADMAP.md")
    roadmap = parse_roadmap(roadmap_path)

    selected_paths = {Path(p) for p in paths} if paths else None
    updated: List[Path] = []
    for profile in profiles:
        if selected_paths and profile.path not in selected_paths:
            continue
        target_path = REPO_ROOT / profile.path
        content = render_profile(profile, roadmap)
        write_file(target_path, content)
        updated.append(target_path)
    return updated


def ensure_unique_path(config: Mapping[str, Any], path: Path) -> None:
    for entry in config.get("files", []):
        if Path(entry.get("path")) == path:
            raise AgentsConfigError(
                f"Configuration already contains an entry for {path}"
            )


def create_profile(
    directory: Path,
    title: str,
    scope: str,
    roadmap_focus: Sequence[str],
    config_path: Path = CONFIG_PATH,
) -> Path:
    config = load_config(config_path)
    target_file = (directory / "AGENTS.md") if directory.is_dir() else directory
    if target_file.suffix != ".md":
        target_file = target_file / "AGENTS.md"
    relative_path = target_file.relative_to(REPO_ROOT)
    ensure_unique_path(config, relative_path)

    roadmap_entries = (
        [{"phase": phase, "label": phase} for phase in roadmap_focus]
        if roadmap_focus
        else []
    )

    defaults = config.get("defaults", {})
    if defaults is not None and not isinstance(defaults, Mapping):
        raise AgentsConfigError("`defaults` must be a mapping when provided")
    base_dynamic_notes = validate_dynamic_notes(
        defaults.get("dynamic_notes", []), "defaults.dynamic_notes"
    )

    default_section = Section(
        heading="Implementation Checklist",
        bullets=[
            "Inherit global guardrails from the repository root and document subsystem-specific rules here.",
            "List required tests and tooling so contributors keep the suite green.",
        ],
        paragraphs=[
            "Tailor this section by editing `configs/agents/agents_config.json` and rerunning the refresh command.",
        ],
    )
    new_entry = {
        "path": str(relative_path),
        "title": title,
        "scope": scope,
        "dynamic_notes": [
            "Generated via `scripts/manage_agents.py create`. Update the shared config and refresh to customise sections.",
        ],
        "roadmap_focus": roadmap_entries,
        "sections": [
            {
                "heading": default_section.heading,
                "bullets": list(default_section.bullets),
                "paragraphs": list(default_section.paragraphs),
            }
        ],
    }
    config.setdefault("files", []).append(new_entry)
    config["files"] = sorted(config["files"], key=lambda item: item["path"])
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    # Render immediately so the new file exists.
    roadmap = parse_roadmap(
        REPO_ROOT / config.get("roadmap", {}).get("file", "ROADMAP.md")
    )
    profile = FileProfile(
        path=relative_path,
        title=title,
        scope=scope,
        dynamic_notes=merge_dynamic_notes(
            base_dynamic_notes,
            [
                "Generated via `scripts/manage_agents.py create`. Update the shared config and refresh to customise sections.",
            ],
        ),
        roadmap_focus=new_entry["roadmap_focus"],
        sections=[default_section],
    )
    content = render_profile(profile, roadmap)
    write_file(REPO_ROOT / relative_path, content)
    return REPO_ROOT / relative_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage scoped AGENTS files")
    subparsers = parser.add_subparsers(dest="command")

    refresh_parser = subparsers.add_parser(
        "refresh", help="Regenerate configured AGENTS files"
    )
    refresh_parser.add_argument(
        "paths",
        nargs="*",
        help="Optional subset of AGENTS.md paths to refresh (relative to repo root)",
    )

    create_parser = subparsers.add_parser(
        "create", help="Scaffold a new AGENTS file and config entry"
    )
    create_parser.add_argument(
        "target", help="Directory or file path for the new AGENTS guide"
    )
    create_parser.add_argument(
        "--title", required=True, help="Title for the generated AGENTS file"
    )
    create_parser.add_argument(
        "--scope", required=True, help="Scope description for the AGENTS file"
    )
    create_parser.add_argument(
        "--roadmap-phase",
        action="append",
        default=[],
        help="Roadmap phase heading to associate with this scope. Use multiple flags for more than one phase.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "refresh"):
        selected_paths = getattr(args, "paths", None)
        updated = refresh_agents(selected_paths)
        for path in updated:
            print(f"Updated {path.relative_to(REPO_ROOT)}")
        return

    if args.command == "create":
        target = Path(args.target)
        if not target.is_absolute():
            target = (REPO_ROOT / target).resolve()
        updated_path = create_profile(
            directory=target,
            title=args.title,
            scope=args.scope,
            roadmap_focus=args.roadmap_phase,
        )
        print(f"Created {updated_path.relative_to(REPO_ROOT)}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
