from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tools.monGARS_deep_scan.extractors import (
    code_py,
    code_swift,
    configs_yaml,
    dockerfiles,
    html_jsx,
    shells,
)


def test_python_docstring_extraction_produces_dialog_and_embedding():
    text = textwrap.dedent(
        '''"""
User: Salut, peux-tu m'aider avec le pipeline?
Assistant: Bien sûr, on va régler ça icitte sans stress.
"""

def helper():
    """Cette fonction décrit comment magasiner les étapes du workflow en détail prolongé pour dépasser les soixante caractères."""
    pass
'''
    )
    records = code_py.extract(Path("module.py"), text)
    dialog_records = [r for r in records if r.dataset == "sft"]
    embedding_records = [r for r in records if r.dataset == "embeddings"]
    assert dialog_records, "Expected a dialog record from the module docstring"
    assert embedding_records, "Expected embedding paragraphs from docstrings"
    assert dialog_records[0].source_file == "module.py"
    assert dialog_records[0].start_line == 1


def test_yaml_workflow_step_extraction():
    text = textwrap.dedent(
        """
name: Example workflow
description: |
  Ce pipeline décrit comment préparer une poutine maison avec des patates croustillantes et une sauce maison riche.
jobs:
  build:
    steps:
      - name: Install deps
        run: pip install .
        shell: bash
"""
    )
    records = configs_yaml.extract(Path(".github/workflows/example.yml"), text)
    agent_records = [r for r in records if r.dataset == "agent"]
    assert agent_records, "Expected workflow step to produce agent record"
    step = agent_records[0]
    assert step.output["run"] == "pip install ."
    assert step.start_line >= 1


def test_dockerfile_parses_run_commands():
    text = textwrap.dedent(
        """
FROM python:3.11-slim
RUN echo "Salut" && echo "poutine pour tout le monde"
CMD [\"python\", \"app.py\"]
"""
    )
    records = dockerfiles.extract(Path("Dockerfile"), text)
    assert any(r.dataset == "agent" and r.type_label == "docker_run" for r in records)


def test_shell_comment_embedding_and_usage():
    text = textwrap.dedent(
        """
# Ce script explique comment magasiner au dépanneur pour le brunch dominical avec beaucoup de détails.
echo "Usage: ./script.sh --help"
"""
    )
    records = shells.extract(Path("script.sh"), text)
    assert any(r.dataset == "embeddings" for r in records)
    assert any(r.dataset == "agent" for r in records)


def test_html_dialog_and_paragraph():
    text = textwrap.dedent(
        """
<html>
  <body>
    <p>Ce paragraphe décrit une aventure au dépanneur avec beaucoup de texte pour dépasser la limite fixée par l'extracteur.</p>
    <div>User: Bonjour, peux-tu trouver ma tuque?</div>
    <div>Assistant: Ben oui, regarde dans le char stationné icitte.</div>
  </body>
</html>
"""
    )
    records = html_jsx.extract(Path("template.html"), text)
    assert any(r.dataset == "embeddings" for r in records)
    assert any(r.dataset == "sft" for r in records)


def test_swift_extracts_prompt_dialog_docs_and_tool_contract():
    text = textwrap.dedent(
        '''
        /// This documented runtime contract stays long enough to become a
        /// provenance-backed embedding record for the local iOS agent kernel.
        struct AgentKernel {}

        let systemPrompt = """
        User: Find the nearest pharmacy.
        Assistant: I will use the local maps search after checking location access.
        """

        let tool = ToolDefinition(
          id: "maps.search",
          name: "Search Nearby",
          category: .location,
          description: "Find nearby places without using general web search.",
          icon: "map",
          tint: "teal",
          requiresApproval: false,
          permissionKey: "NSLocationWhenInUseUsageDescription"
        )
        '''
    )

    records = code_swift.extract(Path("AgentKernel.swift"), text)

    assert any(r.dataset == "embeddings" for r in records)
    assert any(
        r.dataset == "sft" and r.type_label == "swift_prompt_dialog" for r in records
    )
    tool = next(r for r in records if r.type_label == "swift_tool_definition")
    assert tool.output == {
        "id": "maps.search",
        "name": "Search Nearby",
        "description": "Find nearby places without using general web search.",
        "requires_approval": False,
        "source_language": "swift",
    }
    assert tool.start_line > 1


def test_swift_extracts_compact_agent_tool_catalog_entries():
    text = """
    static let all = [
      tool("calendar.create", "Create Event", "Add an event.", .productivity,
           [arg("title"), arg("startsInMinutes", .number)], .calendar,
           .high, true, false),
      tool("weather", "Current Weather", "Read current conditions.", .location,
           [arg("location", required: false)], .location,
           .low, false, true, 4_000),
    ]
    """

    records = code_swift.extract(Path("AgentToolCatalog.swift"), text)
    tools = [
        record.output
        for record in records
        if record.type_label == "swift_agent_tool_definition"
    ]
    assert [tool["id"] for tool in tools] == ["calendar.create", "weather"]
    assert tools[0]["display_name"] == "Create Event"
    assert tools[0]["category"] == "productivity"
    assert tools[0]["permission"] == "calendar"
    assert tools[0]["requires_approval"] is True
    assert tools[0]["supports_background_execution"] is False
    assert tools[0]["maximum_output_characters"] == 2_400
    assert tools[0]["arguments"] == [
        {
            "name": "title",
            "type": "string",
            "required": True,
            "allowed_values": None,
        },
        {
            "name": "startsInMinutes",
            "type": "number",
            "required": True,
            "allowed_values": None,
        },
    ]
    assert tools[0]["json_schema"] == {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "startsInMinutes": {"type": "number"},
        },
        "required": ["title", "startsInMinutes"],
        "additionalProperties": False,
    }

    assert tools[1]["display_name"] == "Current Weather"
    assert tools[1]["requires_approval"] is False
    assert tools[1]["supports_background_execution"] is True
    assert tools[1]["maximum_output_characters"] == 4_000
    assert tools[1]["json_schema"]["required"] == []


@pytest.mark.parametrize(
    "malformed",
    [
        (
            'tool("invalid.computed", computedName, "Description", .knowledge, '
            "[], nil, .low, false, false)"
        ),
        'tool("invalid.unterminated", "Name", "Description", .knowledge, [',
    ],
)
def test_swift_strict_tool_extraction_rejects_malformed_literal_calls(
    malformed: str,
) -> None:
    valid = (
        'tool("valid.tool", "Name", "Description", .knowledge, '
        "[], nil, .low, false, false)"
    )

    with pytest.raises(code_swift.SwiftToolExtractionError):
        code_swift.extract_agent_tool_definitions(
            Path("AgentToolCatalog.swift"), f"{valid}\n{malformed}", strict=True
        )
