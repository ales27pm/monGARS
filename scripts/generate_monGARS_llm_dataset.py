from __future__ import annotations

import argparse
import ast
import copy
import json
import logging
import random
import subprocess
import textwrap
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleSummary:
    path: Path
    module_path: str
    docstring: str | None
    classes: list[str]
    functions: list[str]
    async_functions: list[str]
    constants: list[str]
    imports: list[str]
    has_logging: bool
    has_metrics: bool


@dataclass(frozen=True)
class Scenario:
    slug: str
    title: str
    summary: str
    details: list[str]
    metrics: list[str]
    risks: list[str]
    references: list[str]


@dataclass(frozen=True)
class Playbook:
    slug: str
    issue: str
    summary: str
    diagnostics: list[str]
    mitigations: list[str]
    validation: list[str]
    references: list[str]


@dataclass(frozen=True)
class AlignmentPlan:
    slug: str
    goal: str
    focus: str
    constraints: list[str]
    evaluation: list[str]
    references: list[str]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_python_module(path: Path) -> ModuleSummary | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Unable to read %s: %s", path, exc)
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        LOGGER.warning("Unable to parse %s: %s", path, exc)
        return None

    docstring = ast.get_docstring(tree)
    classes: list[str] = []
    functions: list[str] = []
    async_functions: list[str] = []
    constants: list[str] = []
    imports: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            async_functions.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants.append(target.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level_prefix = "." * node.level
            target = f"{level_prefix}{module}"
            if target:
                imports.append(target)

    module_path = str(path.relative_to(repo_root()))
    normalised_imports = sorted({imp for imp in imports if imp})
    return ModuleSummary(
        path=path,
        module_path=module_path,
        docstring=docstring,
        classes=sorted(classes),
        functions=sorted(functions),
        async_functions=sorted(async_functions),
        constants=sorted(constants),
        imports=normalised_imports,
        has_logging="logging" in normalised_imports,
        has_metrics="opentelemetry" in "".join(normalised_imports),
    )


def collect_module_summaries(root: Path) -> list[ModuleSummary]:
    summaries: list[ModuleSummary] = []
    for path in sorted(root.glob("monGARS/**/*.py")):
        if path.name == "__init__.py":
            continue
        summary = _load_python_module(path)
        if summary:
            summaries.append(summary)
    for path in sorted(root.glob("modules/**/*.py")):
        if path.name == "__init__.py":
            continue
        summary = _load_python_module(path)
        if summary:
            summaries.append(summary)
    return summaries


def _format_list(items: Sequence[str], bullet: str = "- ") -> str:
    if not items:
        return "- None"
    return "\n".join(f"{bullet}{item}" for item in items)


def _format_ordered(items: Sequence[str]) -> str:
    if not items:
        return "1. None"
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def _internal_imports(summary: ModuleSummary) -> list[str]:
    return [
        imp
        for imp in summary.imports
        if imp.startswith("monGARS") or imp.startswith("modules") or imp.startswith(".")
    ]


def _external_imports(summary: ModuleSummary) -> list[str]:
    return [
        imp
        for imp in summary.imports
        if not (
            imp.startswith("monGARS")
            or imp.startswith("modules")
            or imp.startswith(".")
            or imp.startswith("typing")
        )
    ]


def _function_groups(functions: Sequence[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "async": [],
        "cache": [],
        "record": [],
        "diagnose": [],
        "train": [],
        "helpers": [],
    }
    for name in functions:
        lowered = name.lower()
        if lowered.startswith("async"):
            groups["async"].append(name)
        if lowered.startswith("record") or "record" in lowered:
            groups["record"].append(name)
        if lowered.startswith("diagnose") or "diagnose" in lowered:
            groups["diagnose"].append(name)
        if lowered.startswith("train") or "train" in lowered:
            groups["train"].append(name)
        if lowered.startswith("cache"):
            groups["cache"].append(name)
        if lowered.startswith(("get", "build", "create", "ensure")):
            groups["helpers"].append(name)
    return groups


def build_module_examples(summaries: Sequence[ModuleSummary]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary in summaries:
        groups = _function_groups(summary.functions + summary.async_functions)
        internal = _internal_imports(summary)
        external = _external_imports(summary)

        overview_prompt = textwrap.dedent(
            f"""
            Summarise the responsibilities of `{summary.module_path}` including key classes and functions.
            """
        ).strip()
        overview_response = textwrap.dedent(
            f"""
            Module path: `{summary.module_path}`
            Docstring: {summary.docstring or 'Not documented.'}

            Classes:
            {_format_list(summary.classes)}

            Functions:
            {_format_list(summary.functions)}

            Async functions:
            {_format_list(summary.async_functions)}
            """
        ).strip()
        records.append(
            {
                "prompt": overview_prompt,
                "response": overview_response,
                "metadata": {
                    "category": "module_reference",
                    "subcategory": "overview",
                    "module": summary.module_path,
                },
            }
        )

        dependency_prompt = textwrap.dedent(
            f"""
            Describe external and internal dependencies imported by `{summary.module_path}`.
            """
        ).strip()
        dependency_response = textwrap.dedent(
            f"""
            Internal imports:
            {_format_list(internal)}

            External imports:
            {_format_list(external)}

            The module relies on these packages to coordinate with the broader monGARS stack.
            """
        ).strip()
        records.append(
            {
                "prompt": dependency_prompt,
                "response": dependency_response,
                "metadata": {
                    "category": "module_reference",
                    "subcategory": "dependencies",
                    "module": summary.module_path,
                },
            }
        )

        async_prompt = textwrap.dedent(
            f"""
            Highlight asynchronous entry points exposed by `{summary.module_path}` and explain their roles.
            """
        ).strip()
        async_details = (
            _format_list(summary.async_functions)
            if summary.async_functions
            else "- This module does not define async functions; callers operate synchronously."
        )
        async_response = textwrap.dedent(
            f"""
            Async surfaces for `{summary.module_path}`:
            {async_details}

            When integrating with these functions ensure event loop scheduling and timeout handling mirror the
            patterns in the core conversation pipeline.
            """
        ).strip()
        records.append(
            {
                "prompt": async_prompt,
                "response": async_response,
                "metadata": {
                    "category": "module_reference",
                    "subcategory": "async",
                    "module": summary.module_path,
                },
            }
        )

        testing_prompt = textwrap.dedent(
            f"""
            Outline testing strategies for `{summary.module_path}` focusing on critical functions.
            """
        ).strip()
        important_functions = (
            groups["diagnose"] or groups["record"] or summary.functions
        )
        testing_response = textwrap.dedent(
            f"""
            Recommended testing focus for `{summary.module_path}`:

            - Unit test coverage for: {_format_list(important_functions, bullet='')}
            - Mock dependencies from: {_format_list(internal, bullet='')}
            - Validate error handling paths reflected in docstring `{summary.docstring or 'Not documented.'}`
            - Exercise async interfaces with pytest-asyncio where available.
            """
        ).strip()
        records.append(
            {
                "prompt": testing_prompt,
                "response": testing_response,
                "metadata": {
                    "category": "module_reference",
                    "subcategory": "testing",
                    "module": summary.module_path,
                },
            }
        )

        observability_prompt = textwrap.dedent(
            f"""
            Explain how `{summary.module_path}` handles logging, metrics, and observability hooks.
            """
        ).strip()
        observability_sections = []
        if summary.has_logging:
            observability_sections.append(
                "- Imports `logging` and emits structured logs for key operations."
            )
        else:
            observability_sections.append(
                "- Does not import `logging`; rely on callers for logging context."
            )
        if summary.has_metrics:
            observability_sections.append(
                "- Integrates with OpenTelemetry metrics for runtime visibility."
            )
        else:
            observability_sections.append(
                "- No direct metrics integration detected in this module."
            )
        observability_sections.append(
            f"- Internal collaborators: {_format_list(internal, bullet='')}."
        )
        observability_response = "\n".join(observability_sections)
        records.append(
            {
                "prompt": observability_prompt,
                "response": observability_response,
                "metadata": {
                    "category": "module_reference",
                    "subcategory": "observability",
                    "module": summary.module_path,
                },
            }
        )

        extension_prompt = textwrap.dedent(
            f"""
            Identify extension points within `{summary.module_path}` for developers adding new behaviour.
            """
        ).strip()
        helper_functions = groups["helpers"] or summary.functions
        extension_response = textwrap.dedent(
            f"""
            Extension points:
            {_format_list(helper_functions)}

            Classes available for subclassing:
            {_format_list(summary.classes)}
            """
        ).strip()
        records.append(
            {
                "prompt": extension_prompt,
                "response": extension_response,
                "metadata": {
                    "category": "module_reference",
                    "subcategory": "extension",
                    "module": summary.module_path,
                },
            }
        )

        data_prompt = textwrap.dedent(
            f"""
            Document key constants or configuration touchpoints exposed by `{summary.module_path}`.
            """
        ).strip()
        data_response = textwrap.dedent(
            f"""
            Configuration constants:
            {_format_list(summary.constants)}

            When integrating ensure these align with settings defined in `monGARS.config`.
            """
        ).strip()
        records.append(
            {
                "prompt": data_prompt,
                "response": data_response,
                "metadata": {
                    "category": "module_reference",
                    "subcategory": "configuration",
                    "module": summary.module_path,
                },
            }
        )

        integration_prompt = textwrap.dedent(
            f"""
            Map how `{summary.module_path}` collaborates with other modules in the monGARS stack.
            """
        ).strip()
        integration_response = textwrap.dedent(
            f"""
            Internal collaborators:
            {_format_list(internal)}

            External libraries:
            {_format_list(external)}

            The module exposes {len(summary.functions) + len(summary.async_functions)} callable(s)
            that can be orchestrated via the ConversationalModule or EvolutionEngine depending on context.
            """
        ).strip()
        records.append(
            {
                "prompt": integration_prompt,
                "response": integration_response,
                "metadata": {
                    "category": "module_reference",
                    "subcategory": "integration",
                    "module": summary.module_path,
                },
            }
        )
    return records


ARCHITECTURE_SCENARIOS: list[Scenario] = [
    Scenario(
        slug="core-turn",
        title="ConversationalModule orchestration",
        summary=(
            "ConversationalModule enriches user queries with curiosity, reasoning, adaptive tone, and persistence."
        ),
        details=[
            "_handle_image injects captions when available",
            "_augment_with_curiosity queries CuriosityEngine.detect_gaps",
            "_refine_query relies on AdvancedReasoner.reason",
            "LLMIntegration.generate_response executes the prompt",
            "AdaptiveResponseGenerator and MimicryModule tailor persona",
            "PersistenceRepository.save_interaction captures audit trail",
            "EvolutionEngine.record_memory_sample stores curated snippets",
        ],
        metrics=[
            "llm.ray.requests/latency",
            "conversation processing_time",
            "memory sample ingestion counts",
        ],
        risks=[
            "LLM timeout leads to empty response",
            "Persistence failure may drop history",
            "Persona cache miss increases latency",
        ],
        references=[
            "monGARS/core/conversation.py",
            "monGARS/core/evolution_engine.py",
        ],
    ),
    Scenario(
        slug="multimodal",
        title="Image caption augmented flow",
        summary="Orchestrator integrates ImageCaptioning, curiosity, and neuro-symbolic reasoning before LLM execution.",
        details=[
            "ImageCaptioning.generate_caption enriches prompt",
            "CuriosityEngine.detect_gaps appends missing context",
            "AdvancedReasoner.reason returns structured refinements",
            "AdaptiveResponseGenerator adjusts style",
        ],
        metrics=[
            "Caption generation latency",
            "Curiosity gap hit rate",
            "LLMIntegration latency histogram",
        ],
        risks=[
            "Captioning failure reverts to baseline text",
            "Curiosity wait_for timeout",
            "Reasoner exception falls back to raw prompt",
        ],
        references=[
            "monGARS/core/orchestrator.py",
            "monGARS/core/mains_virtuelles.py",
        ],
    ),
    Scenario(
        slug="memory",
        title="Memory consolidation",
        summary="EvolutionEngine aggregates MemoryService entries with telemetry for retraining.",
        details=[
            "Hippocampus history hydrates short-term context",
            "EvolutionEngine.record_memory_sample queues entries",
            "Long haul validation schedules regression checks",
        ],
        metrics=[
            "Memory deque size",
            "Training event throughput",
            "Validation success rate",
        ],
        risks=[
            "Deque overflow when TTL misconfigured",
            "Validation job failure",
        ],
        references=[
            "monGARS/core/hippocampus.py",
            "monGARS/core/evolution_engine.py",
        ],
    ),
    Scenario(
        slug="model-slots",
        title="Model slot reconciliation",
        summary="ModelSlotManager and LLMModelManager coordinate adapter lifecycle and Ray deployment updates.",
        details=[
            "ensure_models_installed downloads manifests",
            "ModelSlotManager.reconcile_slots updates worker pools",
            "LLMIntegration caches adapter metadata",
        ],
        metrics=[
            "Adapter install duration",
            "Ray scaling events",
            "Cache hit ratios",
        ],
        risks=[
            "Manifest mismatch",
            "Slot reconciliation failure",
            "Adapter download errors",
        ],
        references=[
            "monGARS/core/model_slot_manager.py",
            "monGARS/core/model_manager.py",
        ],
    ),
    Scenario(
        slug="observability",
        title="Reinforcement observability",
        summary="Telemetry pipeline captures reward signals, system stats, and publishes to dashboards.",
        details=[
            "collect_observation assembles payload",
            "emit_reward_signal forwards to exporters",
            "EvolutionEngine ingests metrics during train_cycle",
        ],
        metrics=[
            "Reward signal throughput",
            "Exporter success/failure counters",
            "Dashboard freshness",
        ],
        risks=[
            "Exporter outage",
            "Payload schema drift",
        ],
        references=[
            "monGARS/core/reinforcement_observability.py",
            "monGARS/core/sustainability_dashboard.py",
        ],
    ),
    Scenario(
        slug="safety",
        title="Operator approval safety net",
        summary="OperatorApprovalService ensures sensitive actions require manual review and secure logging.",
        details=[
            "submit_request stores approval ticket",
            "approve_request validates entitlements",
            "notifications inform stakeholders",
        ],
        metrics=[
            "Approval turnaround time",
            "Denied request count",
            "Audit log completeness",
        ],
        risks=[
            "Missing entitlements",
            "Notification delivery failure",
        ],
        references=[
            "monGARS/core/operator_approvals.py",
            "monGARS/core/security.py",
        ],
    ),
]


OPERATIONS_PLAYBOOKS: list[Playbook] = [
    Playbook(
        slug="latency",
        issue="LLM latency spike",
        summary="Investigate LLMIntegration latency regressions via EvolutionEngine diagnostics.",
        diagnostics=[
            "Review llm.ray.latency histogram",
            "Inspect EvolutionEngine.diagnose_performance",
            "Check ModelSlotManager reconciliation logs",
        ],
        mitigations=[
            "Scale workers with apply_optimizations",
            "Warm adapters using ensure_models_installed",
            "Enable Unsloth acceleration",
        ],
        validation=[
            "Replay smoke prompts",
            "Ensure latency returns under SLA",
        ],
        references=[
            "monGARS/core/llm_integration.py",
            "monGARS/core/evolution_engine.py",
        ],
    ),
    Playbook(
        slug="memory-pressure",
        issue="Memory backlog",
        summary="Hippocampus buffers and persistence throughput require balancing to avoid backlog.",
        diagnostics=[
            "Inspect Hippocampus TTL configuration",
            "Review PersistenceRepository latency",
            "Check EvolutionEngine memory deque size",
        ],
        mitigations=[
            "Adjust TTL and deque maxlen",
            "Batch persistence writes",
            "Trigger record_memory_sample pruning",
        ],
        validation=[
            "Confirm history retrieval works",
            "Monitor memory deque for stability",
        ],
        references=[
            "monGARS/core/hippocampus.py",
            "monGARS/core/persistence.py",
        ],
    ),
    Playbook(
        slug="persona",
        issue="Persona drift",
        summary="Adaptive response caches may expire causing persona drift across conversations.",
        diagnostics=[
            "Check AdaptiveResponseGenerator cache hits",
            "Review MimicryModule profile freshness",
            "Audit PersonalityEngine feature extraction",
        ],
        mitigations=[
            "Warm caches for active users",
            "Increase TTL or persistence",
            "Refresh mimicry lexicon",
        ],
        validation=[
            "Manual review of transcripts",
            "Track persona drift metric",
        ],
        references=[
            "monGARS/core/dynamic_response.py",
            "monGARS/core/mimicry.py",
        ],
    ),
    Playbook(
        slug="worker",
        issue="Worker crash loop",
        summary="Ray workers may crash due to adapter issues or resource exhaustion.",
        diagnostics=[
            "Check Ray dashboard for restarts",
            "Inspect slot manager allocation",
            "Review EvolutionEngine optimization logs",
        ],
        mitigations=[
            "Scale down problematic adapters",
            "Increase resource requests",
            "Rollback adapter manifest",
        ],
        validation=[
            "Ensure pods remain running",
            "Monitor llm.ray.failures",
        ],
        references=[
            "monGARS/core/model_slot_manager.py",
            "monGARS/core/evolution_engine.py",
        ],
    ),
    Playbook(
        slug="validation",
        issue="Long haul validation failure",
        summary="Regression suites may fail when adapters drift or datasets change.",
        diagnostics=[
            "Inspect long_haul_validation logs",
            "Compare adapter commits",
            "Replay failing prompts",
        ],
        mitigations=[
            "Reschedule validation with debug",
            "Rollback adapter",
            "Update evaluation thresholds",
        ],
        validation=[
            "Confirm train_cycle success",
            "Review reward distribution",
        ],
        references=[
            "monGARS/core/long_haul_validation.py",
            "monGARS/core/evolution_engine.py",
        ],
    ),
    Playbook(
        slug="telemetry",
        issue="Telemetry outage",
        summary="Missing metrics indicate exporter or instrumentation failure.",
        diagnostics=[
            "Check reinforcement observability logs",
            "Validate exporter configuration",
            "Confirm dashboard freshness",
        ],
        mitigations=[
            "Restart exporters",
            "Fallback to file exporter",
            "Patch instrumentation",
        ],
        validation=[
            "Ensure metrics visible",
            "Cross-check Ray stats",
        ],
        references=[
            "monGARS/core/reinforcement_observability.py",
            "monGARS/core/llm_integration.py",
        ],
    ),
]


ALIGNMENT_PLANS: list[AlignmentPlan] = [
    AlignmentPlan(
        slug="rag",
        goal="Accurate retrieval grounded responses",
        focus="Leverage EmbeddingSystem and PersistenceRepository",
        constraints=[
            "Responses cite retrieved document identifiers",
            "Indicate when recall insufficient",
            "Avoid hallucinated sources",
        ],
        evaluation=[
            "Manual review of citations",
            "Monitor used_fallback flag",
            "Check retrieval latency",
        ],
        references=[
            "monGARS/core/embeddings.py",
            "monGARS/core/persistence.py",
        ],
    ),
    AlignmentPlan(
        slug="persona",
        goal="Maintain persona alignment across turns",
        focus="Exercise AdaptiveResponseGenerator and MimicryModule",
        constraints=[
            "Tone must stay consistent",
            "Cache hits remain above 80%",
            "Profile updates recorded",
        ],
        evaluation=[
            "Persona drift metrics",
            "Profile change diffs",
            "Qualitative transcript review",
        ],
        references=[
            "monGARS/core/dynamic_response.py",
            "monGARS/core/mimicry.py",
        ],
    ),
    AlignmentPlan(
        slug="safety",
        goal="Enforce operator approvals",
        focus="Verify OperatorApprovalService and security policies",
        constraints=[
            "Sensitive actions require approval token",
            "Audit logs stored",
            "Errors provide remediation guidance",
        ],
        evaluation=[
            "Blocked unsafe prompts count",
            "Approval turnaround time",
            "Audit completeness",
        ],
        references=[
            "monGARS/core/operator_approvals.py",
            "monGARS/core/security.py",
        ],
    ),
    AlignmentPlan(
        slug="telemetry",
        goal="High fidelity telemetry narratives",
        focus="Ensure reinforcement observability instrumentation",
        constraints=[
            "Responses mention llm.ray.* metrics",
            "Include exporter configuration",
            "Provide debugging fallback",
        ],
        evaluation=[
            "Telemetry coverage",
            "Exporter success rates",
            "Alert accuracy",
        ],
        references=[
            "monGARS/core/reinforcement_observability.py",
            "docs/architecture/module_interactions.md",
        ],
    ),
]


def build_architecture_examples() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scenario in ARCHITECTURE_SCENARIOS:
        flow_prompt = textwrap.dedent(
            f"""
            Walk through the {scenario.title} pipeline and call out module interactions.
            """
        ).strip()
        flow_response = textwrap.dedent(
            f"""
            Summary: {scenario.summary}

            Call order:
            {_format_ordered(scenario.details)}

            Key metrics:
            {_format_list(scenario.metrics)}
            """
        ).strip()
        records.append(
            {
                "prompt": flow_prompt,
                "response": flow_response,
                "metadata": {
                    "category": "architecture",
                    "subcategory": "flow",
                    "slug": scenario.slug,
                },
            }
        )

        observability_prompt = textwrap.dedent(
            f"""
            Which observability hooks are most important when operating the {scenario.title} flow?
            """
        ).strip()
        observability_response = textwrap.dedent(
            f"""
            Focus metrics and signals:
            {_format_list(scenario.metrics)}

            Primary log sources:
            {_format_list(scenario.references)}
            """
        ).strip()
        records.append(
            {
                "prompt": observability_prompt,
                "response": observability_response,
                "metadata": {
                    "category": "architecture",
                    "subcategory": "observability",
                    "slug": scenario.slug,
                },
            }
        )

        risk_prompt = textwrap.dedent(
            f"""
            Document resilience considerations and risks for the {scenario.title} architecture segment.
            """
        ).strip()
        risk_response = textwrap.dedent(
            f"""
            Risks:
            {_format_list(scenario.risks)}

            Mitigation references:
            {_format_list(scenario.references)}
            """
        ).strip()
        records.append(
            {
                "prompt": risk_prompt,
                "response": risk_response,
                "metadata": {
                    "category": "architecture",
                    "subcategory": "resilience",
                    "slug": scenario.slug,
                },
            }
        )
    return records


def build_operations_examples() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for playbook in OPERATIONS_PLAYBOOKS:
        triage_prompt = textwrap.dedent(
            f"""
            Outline triage steps for the {playbook.issue} scenario.
            """
        ).strip()
        triage_response = textwrap.dedent(
            f"""
            Summary: {playbook.summary}

            Diagnostics:
            {_format_list(playbook.diagnostics)}
            """
        ).strip()
        records.append(
            {
                "prompt": triage_prompt,
                "response": triage_response,
                "metadata": {
                    "category": "operations",
                    "subcategory": "diagnostics",
                    "slug": playbook.slug,
                },
            }
        )

        remediation_prompt = textwrap.dedent(
            f"""
            Provide remediation guidance for the {playbook.issue} runbook.
            """
        ).strip()
        remediation_response = textwrap.dedent(
            f"""
            Mitigations:
            {_format_list(playbook.mitigations)}

            Validation checks:
            {_format_list(playbook.validation)}
            """
        ).strip()
        records.append(
            {
                "prompt": remediation_prompt,
                "response": remediation_response,
                "metadata": {
                    "category": "operations",
                    "subcategory": "remediation",
                    "slug": playbook.slug,
                },
            }
        )

        documentation_prompt = textwrap.dedent(
            f"""
            Capture documentation references for the {playbook.issue} playbook.
            """
        ).strip()
        documentation_response = textwrap.dedent(
            f"""
            Source references:
            {_format_list(playbook.references)}

            Keep this playbook aligned with production configuration and review after each incident.
            """
        ).strip()
        records.append(
            {
                "prompt": documentation_prompt,
                "response": documentation_response,
                "metadata": {
                    "category": "operations",
                    "subcategory": "references",
                    "slug": playbook.slug,
                },
            }
        )
    return records


def build_alignment_examples() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for plan in ALIGNMENT_PLANS:
        plan_prompt = textwrap.dedent(
            f"""
            Design a dataset slice to support the goal "{plan.goal}".
            """
        ).strip()
        plan_response = textwrap.dedent(
            f"""
            Focus area: {plan.focus}

            Constraints:
            {_format_list(plan.constraints)}

            Evaluation:
            {_format_list(plan.evaluation)}
            """
        ).strip()
        records.append(
            {
                "prompt": plan_prompt,
                "response": plan_response,
                "metadata": {
                    "category": "alignment",
                    "subcategory": "dataset_design",
                    "slug": plan.slug,
                },
            }
        )

        evaluation_prompt = textwrap.dedent(
            f"""
            Recommend evaluation metrics for the "{plan.goal}" plan.
            """
        ).strip()
        evaluation_response = textwrap.dedent(
            f"""
            Metrics to monitor:
            {_format_list(plan.evaluation)}

            References:
            {_format_list(plan.references)}
            """
        ).strip()
        records.append(
            {
                "prompt": evaluation_prompt,
                "response": evaluation_response,
                "metadata": {
                    "category": "alignment",
                    "subcategory": "metrics",
                    "slug": plan.slug,
                },
            }
        )

        guardrail_prompt = textwrap.dedent(
            f"""
            Specify guardrails for prompts targeting "{plan.goal}".
            """
        ).strip()
        guardrail_response = textwrap.dedent(
            f"""
            Guardrails:
            {_format_list(plan.constraints)}

            Source references:
            {_format_list(plan.references)}
            """
        ).strip()
        records.append(
            {
                "prompt": guardrail_prompt,
                "response": guardrail_response,
                "metadata": {
                    "category": "alignment",
                    "subcategory": "guardrails",
                    "slug": plan.slug,
                },
            }
        )
    return records


def load_openapi_document(root: Path) -> dict[str, Any] | None:
    openapi_path = root / "openapi.json"
    if not openapi_path.exists():
        LOGGER.warning("openapi.json not found at %s", openapi_path)
        return None
    try:
        with openapi_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        LOGGER.error("Unable to parse openapi.json: %s", exc)
        return None


def _summarise_schema(schema: dict[str, Any] | None) -> str:
    if not schema:
        return "No structured payload required."
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines: list[str] = []
    for name, details in props.items():
        field_type = details.get("type", "object")
        description = details.get("description", "")
        modifier = "required" if name in required else "optional"
        snippet = f"- {name} ({field_type}, {modifier})"
        if description:
            snippet += f": {description.strip()}"
        lines.append(snippet)
    if not lines:
        return "Payload shape defined but properties are unspecified."
    return "\n".join(lines)


def _summarise_parameters(parameters: Sequence[dict[str, Any]] | None) -> str:
    if not parameters:
        return "- No query or path parameters documented."
    lines: list[str] = []
    for param in parameters:
        name = param.get("name", "unknown")
        param_in = param.get("in", "query")
        required = "required" if param.get("required") else "optional"
        description = param.get("description", "").strip()
        snippet = f"- {name} ({param_in}, {required})"
        if description:
            snippet += f": {description}"
        lines.append(snippet)
    return "\n".join(lines)


def build_api_examples(doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not doc:
        return []
    paths = doc.get("paths", {})
    examples: list[dict[str, Any]] = []
    for url, methods in sorted(paths.items()):
        for method, spec in sorted(methods.items()):
            if method.lower() not in {"get", "post", "put", "delete", "patch"}:
                continue
            summary = spec.get("summary") or spec.get("operationId") or ""
            request_schema: dict[str, Any] | None = None
            request_body = spec.get("requestBody")
            if isinstance(request_body, dict):
                content = request_body.get("content", {})
                json_body = content.get("application/json")
                if isinstance(json_body, dict):
                    request_schema = json_body.get("schema")
            responses = spec.get("responses", {})
            response_schema_desc = []
            for status, details in sorted(responses.items()):
                description = details.get("description", "")
                response_schema_desc.append(f"{status}: {description}")
            parameters = spec.get("parameters")

            usage_prompt = textwrap.dedent(
                f"""
                Provide a production-ready walkthrough for `{method.upper()} {url}`.
                """
            ).strip()
            usage_response = textwrap.dedent(
                f"""
                Endpoint: `{method.upper()} {url}`
                Summary: {summary or 'No summary provided.'}
                Authentication: Bearer token from `/token` unless endpoint is `/healthz` or `/ready`.

                Request payload guidance:
                {_summarise_schema(request_schema)}

                Response expectations:
                {_format_list(response_schema_desc)}
                """
            ).strip()
            examples.append(
                {
                    "prompt": usage_prompt,
                    "response": usage_response,
                    "metadata": {
                        "category": "api",
                        "subcategory": "usage",
                        "method": method.upper(),
                        "path": url,
                        "source": "openapi.json",
                    },
                }
            )

            payload_prompt = textwrap.dedent(
                f"""
                Summarise request parameters and payload structure for `{method.upper()} {url}`.
                """
            ).strip()
            payload_response = textwrap.dedent(
                f"""
                Parameters:
                {_summarise_parameters(parameters)}

                JSON body schema:
                {_summarise_schema(request_schema)}
                """
            ).strip()
            examples.append(
                {
                    "prompt": payload_prompt,
                    "response": payload_response,
                    "metadata": {
                        "category": "api",
                        "subcategory": "payload",
                        "method": method.upper(),
                        "path": url,
                        "source": "openapi.json",
                    },
                }
            )

            testing_prompt = textwrap.dedent(
                f"""
                Recommend testing strategies for `{method.upper()} {url}` including negative cases.
                """
            ).strip()
            testing_response = textwrap.dedent(
                f"""
                Suggested tests:
                - Happy path with valid authentication and payload.
                - Authentication failure resulting in 401.
                - Schema validation errors asserting 422 responses.
                - Observability assertions ensuring request/response metadata is logged.
                """
            ).strip()
            examples.append(
                {
                    "prompt": testing_prompt,
                    "response": testing_response,
                    "metadata": {
                        "category": "api",
                        "subcategory": "testing",
                        "method": method.upper(),
                        "path": url,
                        "source": "openapi.json",
                    },
                }
            )
    return examples


def build_dataset(openapi_doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    root = repo_root()
    module_summaries = collect_module_summaries(root)
    records: list[dict[str, Any]] = []
    records.extend(build_module_examples(module_summaries))
    records.extend(build_architecture_examples())
    records.extend(build_operations_examples())
    records.extend(build_alignment_examples())
    records.extend(build_api_examples(openapi_doc))
    return records


def detect_git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    records: list[dict[str, Any]],
    train_count: int,
    val_count: int,
    val_ratio: float,
    revision: str | None,
) -> None:
    counts = Counter(
        record.get("metadata", {}).get("category", "unknown") for record in records
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "category_distribution": counts,
        "git_revision": revision,
        "split_counts": {"train": train_count, "val": val_count},
        "val_ratio": val_ratio,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def split_dataset(
    records: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(len(records) * val_ratio))
    val_indices = sorted(indices[:val_size])
    train_indices = sorted(indices[val_size:])
    train_records = [records[idx] for idx in train_indices]
    val_records = [records[idx] for idx in val_indices]
    return train_records, val_records


def annotate_split(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for record in records:
        clone = copy.deepcopy(record)
        metadata = clone.setdefault("metadata", {})
        metadata["split"] = split
        annotated.append(clone)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the monGARS LLM dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override for the combined JSONL output path.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=None,
        help="Optional override for the train split JSONL path.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=None,
        help="Optional override for the validation split JSONL path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional override for the generation report path.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (0-1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Seed for deterministic shuffling when splitting.",
    )
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        raise SystemExit("Validation ratio must be between 0 and 1.")

    root = repo_root()
    openapi_doc = load_openapi_document(root)
    records = build_dataset(openapi_doc)

    if not records:
        raise SystemExit("No dataset records were generated; check source material.")

    train_records, val_records = split_dataset(
        records, val_ratio=args.val_ratio, seed=args.seed
    )
    annotated_train = annotate_split(train_records, "train")
    annotated_val = annotate_split(val_records, "val")
    combined_records = annotated_train + annotated_val

    default_output = root / "datasets" / "monGARS_llm" / "monGARS_llm_dataset.jsonl"
    default_train_output = root / "datasets" / "monGARS_llm" / "monGARS_llm_train.jsonl"
    default_val_output = root / "datasets" / "monGARS_llm" / "monGARS_llm_val.jsonl"

    output_path = args.output or default_output
    train_output = args.train_output or default_train_output
    val_output = args.val_output or default_val_output

    write_jsonl(output_path, combined_records)
    write_jsonl(train_output, annotated_train)
    write_jsonl(val_output, annotated_val)

    LOGGER.info(
        "Wrote %s total records (%s train / %s val)",
        len(combined_records),
        len(annotated_train),
        len(annotated_val),
    )

    default_report = root / "datasets" / "monGARS_llm" / "generation_report.json"
    report_path = args.report or default_report
    revision = detect_git_revision(root)
    write_report(
        report_path,
        combined_records,
        train_count=len(annotated_train),
        val_count=len(annotated_val),
        val_ratio=args.val_ratio,
        revision=revision,
    )
    LOGGER.info("Wrote generation report to %s", report_path)


if __name__ == "__main__":
    main()
