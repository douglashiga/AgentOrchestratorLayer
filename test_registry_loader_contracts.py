from registry.db import RegistryDB
from registry.domain_registry import HandlerRegistry
from registry.loader import RegistryLoader


def test_loader_normalizes_metadata_into_method_spec(tmp_path) -> None:
    db = RegistryDB(db_path=str(tmp_path / "registry.db"))
    registry = HandlerRegistry()
    loader = RegistryLoader(db=db, runtime_registry=registry)

    metadata = loader._normalize_capability_metadata(
        domain_name="finance",
        capability_name="get_stock_price",
        description="Get stock price",
        schema={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
        metadata={
            "workflow": {
                "execution_mode": "dag",
                "nodes": [
                    {"id": "resolve", "kind": "resolve"},
                    {"id": "fetch", "kind": "call"},
                ],
                "edges": [{"from_node": "resolve", "to_node": "fetch"}],
            },
            "policy": {
                "default_timeout_seconds": 20,
                "human_validation": {"enabled": True},
            },
        },
    )

    assert metadata["domain"] == "finance"
    assert metadata["schema"]["required"] == ["symbol"]
    assert metadata["parameter_specs"]["symbol"]["type"] == "string"
    assert metadata["parameter_specs"]["symbol"]["required"] is True
    assert isinstance(metadata.get("method_spec"), dict)
    assert metadata["method_spec"]["method"] == "get_stock_price"
    assert metadata["method_spec"]["workflow"]["execution_mode"] == "dag"


def test_loader_parameter_specs_merge_schema_and_overrides(tmp_path) -> None:
    db = RegistryDB(db_path=str(tmp_path / "registry.db"))
    registry = HandlerRegistry()
    loader = RegistryLoader(db=db, runtime_registry=registry)

    metadata = loader._normalize_capability_metadata(
        domain_name="ops",
        capability_name="restart_job",
        description="Restart background job",
        schema={
            "type": "object",
            "properties": {
                "environment": {
                    "type": "string",
                    "enum": ["DEV", "PROD"],
                    "description": "Target environment",
                },
                "region": {
                    "type": "string",
                    "default": "us-east-1",
                },
            },
            "required": ["environment"],
        },
        metadata={
            "parameter_specs": {
                "environment": {
                    "normalization": {"case": "upper"},
                    "examples": ["prod"],
                }
            }
        },
    )

    specs = metadata.get("parameter_specs", {})
    assert specs["environment"]["required"] is True
    assert specs["environment"]["normalization"]["case"] == "upper"
    assert specs["environment"]["enum"] == ["DEV", "PROD"]
    assert specs["region"]["default"] == "us-east-1"
