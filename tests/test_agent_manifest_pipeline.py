from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.monGARS_deep_scan.agent_manifest_pipeline import (
    AgentManifestPipelineError,
    build_agent_behavior_manifest,
    build_agent_manifest_pipeline,
    load_runtime_audits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _owned_e2e(results: list[dict], *, includes_static: bool = False) -> dict:
    return {
        "exportPolicy": {
            "sourceLayer": "e2eTestReport",
            "ownsLiveE2EScenarios": True,
            "includesDeterministicStaticScenarios": includes_static,
        },
        "payload": {"results": results},
    }


def _model_evidence_event() -> dict:
    return {
        "phase": "model-evidence",
        "message": (
            "runtime=agent-model, kind=model-backed, stage=agent-json-final, "
            "parseError=none"
        ),
    }


def test_manifest_preserves_53_26_22_native_contract_parity() -> None:
    manifest, extractions = build_agent_behavior_manifest(REPO_ROOT)

    assert manifest["contractCounts"] == {
        "tools": 53,
        "approvalTools": 26,
        "intents": 22,
        "roles": 5,
    }
    assert len({tool["id"] for tool in manifest["tools"]}) == 53
    assert len({intent["id"] for intent in manifest["intents"]}) == 22
    assert len(extractions) == 75

    rag = next(tool for tool in manifest["tools"] if tool["id"] == "rag.search")
    source_scope = next(
        argument for argument in rag["arguments"] if argument["name"] == "sourceScope"
    )
    assert source_scope == {
        "name": "sourceScope",
        "type": "enum",
        "required": False,
        "allowedValues": ["all", "documents", "notes", "photos"],
    }
    assert rag["jsonSchema"]["additionalProperties"] is False
    assert rag["maximumOutputCharacters"] == 3_000
    assert len(manifest["sourceIntegrity"]["files"]) == 4
    trigger = next(tool for tool in manifest["tools"] if tool["id"] == "trigger.create")
    assert "conditional_schedule_arguments" in {
        rule["kind"] for rule in trigger["validationRules"]
    }


def test_validation_helper_drift_is_rejected_fail_closed(tmp_path: Path) -> None:
    source = (
        REPO_ROOT
        / "mobile-app/ios/MonGARSCoreML/Sources/MonGARSCoreML/AgentToolValidation.swift"
    )
    changed = source.read_text(encoding="utf-8").replace("number.isFinite,", "true,", 1)
    assert changed != source.read_text(encoding="utf-8")
    candidate = tmp_path / "AgentToolValidation.swift"
    candidate.write_text(changed, encoding="utf-8")

    with pytest.raises(AgentManifestPipelineError, match="AgentToolValidation changed"):
        build_agent_behavior_manifest(REPO_ROOT, validation_path=candidate)


def test_pipeline_is_byte_deterministic_and_emits_every_role_lane(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_agent_manifest_pipeline(REPO_ROOT, first)
    build_agent_manifest_pipeline(REPO_ROOT, second)

    assert _artifact_bytes(first) == _artifact_bytes(second)
    acceptance = {
        row["toolID"]: row["input"]["arguments"]
        for row in _jsonl(first / "dataset" / "eval_scenarios.jsonl")
        if row["scenarioType"] == "tool_schema_acceptance"
    }
    assert acceptance["trigger.create"] == {
        "title": "example",
        "prompt": "example",
        "schedule": "relative",
        "inMinutes": 1,
    }
    assert acceptance["trigger.cancel"] == {"title": "example"}
    assert acceptance["alarm.schedule"] == {"title": "example", "inMinutes": 1}
    assert acceptance["outlook.message.move"] == {
        "messageId": "example",
        "destination": "inbox",
    }
    for role in ("cortex", "executor", "mouth", "mimicry", "rem"):
        role_dir = first / "roles" / role
        train_sft = _jsonl(role_dir / "train_sft.jsonl")
        validation_sft = _jsonl(role_dir / "validation_sft.jsonl")
        train_dpo = _jsonl(role_dir / "train_dpo.jsonl")
        validation_dpo = _jsonl(role_dir / "validation_dpo.jsonl")
        assert train_sft and validation_sft and train_dpo and validation_dpo
        assert {row["id"] for row in train_sft}.isdisjoint(
            row["id"] for row in validation_sft
        )
        assert all(row["chosen"] != row["rejected"] for row in train_dpo)
        assert {row["sourceSFTRecordID"] for row in train_dpo} == {
            row["id"] for row in train_sft
        }
        assert {row["sourceSFTRecordID"] for row in validation_dpo} == {
            row["id"] for row in validation_sft
        }


@pytest.mark.parametrize(
    "payload",
    (
        b'{"failures": [}',
        b'{"failures": "not-a-list"}',
        b'{"results": [{"scenarioID": "missing-outcome"}]}',
        b'{"results": []}',
        (
            b'{"exportPolicy":{"sourceLayer":"runtimeScenarioRunner.staticChecks"},'
            b'"payload":{"results":[{"passed":true}]}}'
        ),
        (
            b'{"exportPolicy":{"sourceLayer":"e2eTestReport",'
            b'"ownsLiveE2EScenarios":false,'
            b'"includesDeterministicStaticScenarios":false},'
            b'"payload":{"results":[{"requiresAgentRun":true,'
            b'"evidenceMode":"modelBackedRequired","passed":true}]}}'
        ),
    ),
)
def test_malformed_runtime_audit_is_rejected_fail_closed(
    tmp_path: Path, payload: bytes
) -> None:
    audit = tmp_path / "audit.json"
    audit.write_bytes(payload)

    with pytest.raises(AgentManifestPipelineError):
        load_runtime_audits([audit])


def test_lumen_text_report_does_not_hide_strict_failure_lines(tmp_path: Path) -> None:
    audit = tmp_path / "e2e.txt"
    audit.write_text(
        "E2E Test Report\n"
        "✅ Training eval: nominal route\n"
        "Prompt: show weather\n"
        "FAIL approval-hidden: protected action bypassed approval\n",
        encoding="utf-8",
    )

    result = load_runtime_audits([audit])

    assert result.observed_records == 2
    assert result.total_failures == 1
    assert result.live_evidence_inputs == 0


def test_only_owned_live_records_survive_a_mixed_lumen_report(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "mixed-e2e.json"
    audit.write_text(
        json.dumps(
            _owned_e2e(
                [
                    {
                        "scenarioID": "routing-static",
                        "requiresAgentRun": False,
                        "evidenceMode": "routingOnly",
                        "passed": True,
                    },
                    {
                        "scenarioID": "model-live",
                        "requiresAgentRun": True,
                        "evidenceMode": "modelBackedRequired",
                        "passed": True,
                        "failures": [],
                        "events": [_model_evidence_event()],
                    },
                    {
                        "scenarioID": "policy-first-live",
                        "requiresAgentRun": True,
                        "evidenceMode": "policyFirstAllowed",
                        "passed": True,
                        "failures": [],
                        "events": [
                            {
                                "phase": "model-evidence",
                                "message": (
                                    "runtime=deterministic-compatibility, "
                                    "kind=policy-first-deterministic, "
                                    "stage=compatibility-clarification, "
                                    "parseError=none"
                                ),
                            }
                        ],
                    },
                    {
                        "scenarioID": "foundation-chat-live",
                        "requiresAgentRun": True,
                        "evidenceMode": "modelBackedRequired",
                        "passed": True,
                        "failures": [],
                        "events": [
                            {
                                "phase": "model-evidence",
                                "message": (
                                    "runtime=foundationModels, kind=model-backed, "
                                    "stage=chat-text-turn, parseError=none"
                                ),
                            }
                        ],
                    },
                ],
                includes_static=True,
            )
        ),
        encoding="utf-8",
    )

    result = load_runtime_audits([audit])

    assert result.observed_records == 3
    assert result.total_failures == 0
    assert result.live_evidence_inputs == 1


@pytest.mark.parametrize(
    "result",
    (
        {
            "scenarioID": "contradictory-pass",
            "requiresAgentRun": True,
            "evidenceMode": "modelBackedRequired",
            "passed": True,
            "failures": ["model was not loaded"],
            "events": [_model_evidence_event()],
        },
        {
            "scenarioID": "no-model-event",
            "requiresAgentRun": True,
            "evidenceMode": "modelBackedRequired",
            "passed": True,
            "failures": [],
            "events": [
                {
                    "phase": "model-evidence",
                    "message": "No model loaded; routing-only checks completed.",
                }
            ],
        },
        {
            "scenarioID": "fake-routing-evidence",
            "requiresAgentRun": True,
            "evidenceMode": "modelBackedRequired",
            "passed": True,
            "failures": [],
            "events": [
                {
                    "phase": "model-evidence",
                    "message": (
                        "runtime=none, kind=routingOnly, stage=routing-only, "
                        "parseError=none"
                    ),
                }
            ],
        },
        {
            "scenarioID": "made-up-model-evidence",
            "requiresAgentRun": True,
            "evidenceMode": "modelBackedRequired",
            "passed": True,
            "failures": [],
            "events": [
                {
                    "phase": "model-evidence",
                    "message": (
                        "runtime=garbage, kind=model-backed, stage=made-up, "
                        "parseError=none"
                    ),
                }
            ],
        },
        {
            "scenarioID": "near-prefix-model-evidence",
            "requiresAgentRun": True,
            "evidenceMode": "modelBackedRequired",
            "passed": True,
            "failures": [],
            "events": [
                {
                    "phase": "model-evidence",
                    "message": (
                        "runtime=agent-model, kind=model-backed, "
                        "stage=agentjson-made-up, parseError=none"
                    ),
                }
            ],
        },
    ),
)
def test_contradictory_or_no_model_live_evidence_is_rejected(
    tmp_path: Path, result: dict
) -> None:
    audit = tmp_path / "invalid-live.json"
    audit.write_text(json.dumps(_owned_e2e([result])), encoding="utf-8")

    with pytest.raises(AgentManifestPipelineError):
        load_runtime_audits([audit])


def test_policyless_json_is_diagnostic_and_cannot_close_live_loop(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "legacy.json"
    audit.write_text('{"results":[{"passed":true}]}', encoding="utf-8")

    result = load_runtime_audits([audit])

    assert result.observed_records == 1
    assert result.live_evidence_inputs == 0


def test_unmarked_top_level_e2e_scenario_results_are_rejected(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "unmarked-top-level.json"
    audit.write_text(
        json.dumps(
            {
                "exportPolicy": {
                    "sourceLayer": "e2eTestReport",
                    "ownsLiveE2EScenarios": True,
                    "includesDeterministicStaticScenarios": False,
                },
                "scenarioResults": [
                    {"scenarioID": "unmarked", "passed": True, "failures": []}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentManifestPipelineError, match="runtime marker"):
        load_runtime_audits([audit])


def test_top_level_e2e_preserves_explicit_failed_outcome(tmp_path: Path) -> None:
    audit = tmp_path / "failed-top-level.json"
    audit.write_text(
        json.dumps(
            {
                "exportPolicy": {
                    "sourceLayer": "e2eTestReport",
                    "ownsLiveE2EScenarios": True,
                    "includesDeterministicStaticScenarios": False,
                },
                "scenarioResults": [
                    {
                        "scenarioID": "explicit-failure",
                        "requiresAgentRun": True,
                        "evidenceMode": "modelBackedRequired",
                        "passed": False,
                        "events": [_model_evidence_event()],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = load_runtime_audits([audit])

    assert result.live_evidence_inputs == 1
    assert result.total_failures == 1


def test_json_shaped_secrets_are_removed_from_repair_artifacts(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "secret.json"
    audit.write_text(
        json.dumps(
            {
                "failures": [
                    {
                        "scenarioID": "secret-shaped-diagnostic",
                        "toolID": "token=hf_toolleak",
                        "intent": "credential=intentleak",
                        "message": {
                            "access_token": "hf_supersecret",
                            "nested": {"password": "hunter2"},
                            "contact": "private@example.com",
                            "note": (
                                "token=hf_freeform secret=hidden "
                                "credential=credvalue, "
                                "Authorization: Bearer authleak, "
                                "Cookie: sessionid=cookieleak; preference=x"
                            ),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "secret-bundle"

    build_agent_manifest_pipeline(REPO_ROOT, output, runtime_audit_paths=[audit])

    bundle = b"\n".join(
        path.read_bytes() for path in output.rglob("*") if path.is_file()
    )
    assert b"hf_supersecret" not in bundle
    assert b"hunter2" not in bundle
    assert b"private@example.com" not in bundle
    assert b"hf_freeform" not in bundle
    assert b"hidden" not in bundle
    assert b"credvalue" not in bundle
    assert b"authleak" not in bundle
    assert b"cookieleak" not in bundle
    assert b"hf_toolleak" not in bundle
    assert b"intentleak" not in bundle
    assert b"<redacted-secret>" in bundle
    assert b"<redacted-email>" in bundle


def test_audit_tool_and_intent_identifiers_are_bounded_and_hashed(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "identifier-bounds.json"
    oversized = "a" * 10_000
    audit.write_text(
        json.dumps(
            {
                "failures": [
                    {
                        "toolID": f"cookie={oversized}",
                        "intent": f"private_key={oversized}",
                        "message": "failed",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = load_runtime_audits([audit])
    failure = result.failures[0]

    assert failure["toolID"].startswith("redacted-invalid-tool-id-")
    assert failure["intent"].startswith("redacted-invalid-intent-")
    assert oversized not in json.dumps(failure)


def test_diagnostic_failure_keeps_runtime_evidence_gap_open(tmp_path: Path) -> None:
    audit = tmp_path / "diagnostic.json"
    audit.write_text(
        '{"failures":[{"scenarioID":"diagnostic-failure","message":"failed"}]}',
        encoding="utf-8",
    )
    output = tmp_path / "diagnostic-bundle"

    result = build_agent_manifest_pipeline(
        REPO_ROOT, output, runtime_audit_paths=[audit]
    )

    gap_ids = {gap["id"] for gap in result.improvement_report["gaps"]}
    assert "runtime_evidence_missing" in gap_ids
    assert len(result.improvement_report["gaps"]) == 2


def test_runtime_failure_becomes_bounded_repair_and_all_hashes_verify(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "e2e.json"
    audit.write_text(
        json.dumps(
            _owned_e2e(
                [
                    {
                        "scenarioID": "calendar-approval-gate",
                        "requiresAgentRun": True,
                        "evidenceMode": "modelBackedRequired",
                        "passed": False,
                        "toolID": "calendar.create",
                        "intent": "calendar",
                        "failures": [
                            "protected action reached execution without approval"
                        ],
                        "events": [_model_evidence_event()],
                    },
                    {
                        "scenarioID": "weather-route",
                        "requiresAgentRun": True,
                        "evidenceMode": "modelBackedRequired",
                        "passed": True,
                        "failures": [],
                        "events": [_model_evidence_event()],
                    },
                ]
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "bundle"
    result = build_agent_manifest_pipeline(
        REPO_ROOT, output, runtime_audit_paths=[audit]
    )

    repairs = _jsonl(output / "dataset" / "runtime_audit_repairs.jsonl")
    assert len(repairs) == 1
    assert repairs[0]["agentRole"] == "rem"
    assert repairs[0]["metadata"]["bounded"] is True
    assert result.improvement_report["runtimeAudit"]["repairSamples"] == 1
    assert len(result.improvement_report["gaps"]) == 1

    hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
    for relative, expected in hashes["files"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected

    index = json.loads((output / "artifact_index.json").read_text(encoding="utf-8"))
    for entry in index["files"]:
        artifact = output / entry["path"]
        assert artifact.stat().st_size == entry["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]

    sidecar_hash = (output / "artifact_index.sha256").read_text().split()[0]
    assert (
        sidecar_hash
        == hashlib.sha256((output / "artifact_index.json").read_bytes()).hexdigest()
    )
    indexed_paths = {entry["path"] for entry in index["files"]}
    output_paths = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert output_paths == indexed_paths | {
        "artifact_index.json",
        "artifact_index.sha256",
    }
    assert set(hashes["files"]) == indexed_paths - {"artifact_hashes.json"}

    dataset_manifest = json.loads(
        (output / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        dataset_manifest["datasetDigest"] == result.improvement_report["datasetDigest"]
    )
    for relative, metadata in dataset_manifest["artifacts"].items():
        artifact = output / relative
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == metadata["sha256"]
