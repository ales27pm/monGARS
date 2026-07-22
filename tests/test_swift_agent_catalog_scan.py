from pathlib import Path

from tools.monGARS_deep_scan.extractors import code_swift


def test_swift_agent_catalog_is_fully_available_to_dataset_pipeline() -> None:
    catalog_path = Path(
        "mobile-app/ios/MonGARSCoreML/Sources/MonGARSCoreML/AgentToolCatalog.swift"
    )
    records = code_swift.extract(
        catalog_path,
        catalog_path.read_text(encoding="utf-8"),
    )
    tools = [
        record.output
        for record in records
        if record.type_label == "swift_agent_tool_definition"
    ]

    assert len(tools) == 53
    assert len({tool["id"] for tool in tools}) == 53
    assert sum(bool(tool["requires_approval"]) for tool in tools) == 26
