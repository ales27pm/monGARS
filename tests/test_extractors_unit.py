from __future__ import annotations

import textwrap
from pathlib import Path

from tools.monGARS_deep_scan.extractors import (
    code_py,
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
