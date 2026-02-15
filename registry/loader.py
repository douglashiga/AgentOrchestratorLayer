"""
Registry Loader — Syncs DB Configuration with Runtime Registry.

Responsibility:
- Read active domains from RegistryDB
- Instantiate appropriate handlers (Mobile/Local)
- Register to HandlerRegistry
"""

import logging
import json
import os
from typing import Any

from registry.db import RegistryDB
from registry.domain_registry import HandlerRegistry
from registry.http_handler import HttpDomainHandler

# Local handlers (still supported for fallback/testing)
from domains.finance.handler import FinanceDomainHandler
from domains.general.handler import GeneralDomainHandler
from skills.gateway import SkillGateway
from models.selector import ModelSelector

logger = logging.getLogger(__name__)
GENERAL_DEFAULT_MODEL = os.getenv("GENERAL_MODEL_NAME", "qwen2.5-coder:32b").strip() or "qwen2.5-coder:32b"

class RegistryLoader:
    """Loads domains from DB into Runtime Registry."""

    def __init__(self, db: RegistryDB, runtime_registry: HandlerRegistry):
        self.db = db
        self.registry = runtime_registry

    def load_all(self, context: dict[str, Any]) -> None:
        """
        Load all enabled domains from DB.
        'context' provides dependencies for local handlers (e.g. skill_gateway).
        """
        domains = self.db.list_domains()
        logger.info("RegistryLoader.load_all: found %d enabled domains in DB", len(domains))
        
        for domain_conf in domains:
            name = domain_conf["name"]
            dtype = domain_conf["type"]
            config = json.loads(domain_conf["config"])
            
            logger.info("Loading domain '%s' (type=%s)", name, dtype)
            
            handler = None
            
            try:
                if dtype == "remote_http":
                    handler = HttpDomainHandler(
                        base_url=config["url"],
                        auth_token=config.get("auth_token"),
                        timeout=config.get("timeout", 60.0)
                    )
                elif dtype == "local":
                    # Factory for local handlers
                    if name == "finance":
                        handler = FinanceDomainHandler(
                            skill_gateway=context["skill_gateway"],
                            registry=self.registry
                        )
                    elif name == "general":
                        handler = GeneralDomainHandler(
                            model_selector=context["model_selector"],
                            model_name=config.get("model", GENERAL_DEFAULT_MODEL),
                            capability_catalog_provider=context.get("capability_catalog_provider"),
                        )
                
                if handler:
                    self.registry.register_domain(name, handler)
                    
                    # Register capabilities
                    caps = self.db.list_capabilities(name)
                    if dtype == "remote_http" and not caps:
                        logger.info("No cached capabilities for '%s'. Trying live sync...", name)
                        if self.sync_capabilities(name):
                            caps = self.db.list_capabilities(name)
                    logger.info("Domain '%s' has %d capabilities in DB", name, len(caps))
                    for cap in caps:
                        try:
                            raw_metadata = json.loads(cap["metadata"]) if cap.get("metadata") else {}
                        except Exception:
                            raw_metadata = {}
                        try:
                            raw_schema = json.loads(cap["input_schema"]) if cap.get("input_schema") else {}
                        except Exception:
                            raw_schema = {}
                        cap_metadata = self._normalize_capability_metadata(
                            domain_name=name,
                            capability_name=str(cap.get("capability", "")).strip(),
                            description=str(cap.get("description", "")).strip(),
                            schema=raw_schema if isinstance(raw_schema, dict) else {},
                            metadata=raw_metadata if isinstance(raw_metadata, dict) else {},
                        )
                        
                        self.registry.register_capability(
                            cap["capability"], 
                            handler, 
                            metadata=cap_metadata
                        )
                else:
                    logger.warning("No handler factory for domain '%s' (type=%s)", name, dtype)
                        
            except Exception as e:
                logger.error("Failed to load domain '%s': %s", name, e, exc_info=True)

    def sync_core_defaults(self) -> None:
        """
        Seed only core local capabilities required by the orchestrator runtime.
        Domain-specific integrations must be injected via RegistryDB/bootstrapping.
        """
        logger.info("RegistryLoader: syncing core defaults...")
        self.db.register_domain("general", "local", {"model": GENERAL_DEFAULT_MODEL})
        self.db.register_capability(
            domain_name="general",
            capability="chat",
            description="General conversation and help",
            schema={"message": "string"},
            metadata={
                "explanation_template": "{result[response]}",
                "planner_available": False,
            },
        )
        self.db.register_capability(
            domain_name="general",
            capability="list_capabilities",
            description="List available domains and capabilities grouped by domain",
            schema={"message": "string"},
            metadata={
                "explanation_template": "{result[response]}",
                "planner_available": False,
            },
        )

    def sync_local_to_db(self) -> None:
        """
        Backward-compatible alias.
        Kept to avoid breaking old call sites; now seeds only core defaults.
        """
        self.sync_core_defaults()

    def bootstrap_domains(self, domains: list[dict[str, Any]], sync_remote_capabilities: bool = True) -> None:
        """
        Upsert domains from external configuration into RegistryDB.
        This avoids hardcoded domain definitions in code.

        Expected item format:
        {
          "name": "finance",
          "type": "remote_http" | "local",
          "config": {"url": "..."}
        }
        """
        if not domains:
            return

        logger.info("Bootstrapping %d domains into RegistryDB", len(domains))
        for item in domains:
            name = item.get("name")
            domain_type = item.get("type")
            config = item.get("config", {})
            should_sync_caps = bool(item.get("sync_capabilities", sync_remote_capabilities))

            if not name or not domain_type:
                logger.warning("Skipping invalid bootstrap domain entry: %s", item)
                continue

            try:
                self.db.register_domain(name=name, domain_type=domain_type, config=config)
                logger.info("Bootstrapped domain: %s (%s)", name, domain_type)
            except Exception as e:
                logger.error("Failed to bootstrap domain '%s': %s", name, e, exc_info=True)
                continue

            if should_sync_caps and domain_type == "remote_http":
                try:
                    synced = self.sync_capabilities(name)
                    if not synced:
                        logger.warning("Manifest sync failed for remote domain '%s'", name)
                except Exception as e:
                    logger.warning("Manifest sync error for '%s': %s", name, e)

    def sync_capabilities(self, domain_name: str) -> bool:
        """
        Connect to remote domain, fetch manifest, and update DB.
        Only works for 'remote_http' domains.
        """
        domain_conf = self.db.get_domain(domain_name)
        if not domain_conf:
            logger.error("Domain not found: %s", domain_name)
            return False

        if domain_conf["type"] != "remote_http":
            logger.warning("Cannot sync capabilities for non-remote domain: %s", domain_name)
            return False

        config = json.loads(domain_conf["config"])
        try:
            handler = HttpDomainHandler(
                base_url=config["url"],
                auth_token=config.get("auth_token"),
                timeout=config.get("timeout", 10.0)
            )
            manifest = handler.fetch_manifest()
            
            # Validate domain name match? Optional.
            # remote_domain = manifest.get("domain")
            
            capabilities = manifest.get("capabilities", [])
            for cap in capabilities:
                cap_name = str(cap.get("name") or cap.get("capability") or "").strip()
                if not cap_name:
                    logger.warning("Skipping manifest capability without name: %s", cap)
                    continue
                raw_schema = cap.get("schema")
                if not isinstance(raw_schema, dict):
                    raw_schema = {}
                raw_metadata = cap.get("metadata")
                if not isinstance(raw_metadata, dict):
                    raw_metadata = {}
                normalized_metadata = self._normalize_capability_metadata(
                    domain_name=domain_name,
                    capability_name=cap_name,
                    description=str(cap.get("description", "")).strip(),
                    schema=raw_schema,
                    metadata=raw_metadata,
                )
                self.db.register_capability(
                    domain_name=domain_name,
                    capability=cap_name,
                    description=cap.get("description", ""),
                    schema=raw_schema,
                    metadata=normalized_metadata,
                )
            
            logger.info("Synced %d capabilities for domain %s (including metadata)", len(capabilities), domain_name)
            return True
            
        except Exception as e:
            logger.error("Failed to sync capabilities for %s: %s", domain_name, e)
            return False

    def _normalize_capability_metadata(
        self,
        *,
        domain_name: str,
        capability_name: str,
        description: str,
        schema: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize capability metadata to include a canonical method contract."""
        normalized = dict(metadata or {})
        normalized.setdefault("domain", domain_name)
        normalized.setdefault("description", description)
        normalized["schema"] = schema if isinstance(schema, dict) else {}
        normalized["parameter_specs"] = self._normalize_parameter_specs(
            schema=schema if isinstance(schema, dict) else {},
            metadata=normalized,
        )

        workflow = normalized.get("workflow")
        if isinstance(workflow, str):
            try:
                workflow = json.loads(workflow)
            except Exception:
                workflow = None
        if not isinstance(workflow, dict):
            workflow = None

        policy = normalized.get("policy")
        if isinstance(policy, str):
            try:
                policy = json.loads(policy)
            except Exception:
                policy = {}
        if not isinstance(policy, dict):
            policy = {}
        normalized["policy"] = policy

        method_spec = normalized.get("method_spec")
        if isinstance(method_spec, str):
            try:
                method_spec = json.loads(method_spec)
            except Exception:
                method_spec = None
        if not isinstance(method_spec, dict):
            method_spec = None

        if method_spec is None and workflow is not None:
            method_spec = {
                "domain": domain_name,
                "method": capability_name,
                "version": str(normalized.get("version", "1.0.0")).strip() or "1.0.0",
                "description": description,
                "input_schema": schema,
                "output_schema": normalized.get("output_schema", {})
                if isinstance(normalized.get("output_schema"), dict)
                else {},
                "workflow": workflow,
                "policy": policy,
                "tags": normalized.get("tags", []) if isinstance(normalized.get("tags"), list) else [],
                "metadata": normalized.get("contract_metadata", {})
                if isinstance(normalized.get("contract_metadata"), dict)
                else {},
            }

        if isinstance(method_spec, dict):
            method_spec.setdefault("domain", domain_name)
            method_spec.setdefault("method", capability_name)
            method_spec.setdefault("description", description)
            if not isinstance(method_spec.get("input_schema"), dict):
                method_spec["input_schema"] = schema
            if not isinstance(method_spec.get("policy"), dict):
                method_spec["policy"] = policy
            if not isinstance(method_spec.get("workflow"), dict) and workflow is not None:
                method_spec["workflow"] = workflow
            normalized["method_spec"] = method_spec

        return normalized

    def _normalize_parameter_specs(self, *, schema: dict[str, Any], metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Build a canonical parameter spec map from metadata overrides + JSON schema.

        Output format:
        {
          "param_name": {
             "type": "string",
             "required": true,
             "description": "...",
             "examples": ["..."],
             "default": "...",
             "enum": [...],
             "pattern": "...",
             "format": "..."
          }
        }
        """
        raw_specs = metadata.get("parameter_specs")
        specs: dict[str, dict[str, Any]] = {}
        if isinstance(raw_specs, dict):
            for name, value in raw_specs.items():
                param = str(name).strip()
                if not param or not isinstance(value, dict):
                    continue
                specs[param] = dict(value)
        elif isinstance(raw_specs, list):
            for item in raw_specs:
                if not isinstance(item, dict):
                    continue
                param = str(item.get("name", "")).strip()
                if not param:
                    continue
                candidate = {k: v for k, v in item.items() if k != "name"}
                specs[param] = candidate if isinstance(candidate, dict) else {}

        properties = schema.get("properties")
        raw_required = schema.get("required")
        required_items = raw_required if isinstance(raw_required, list) else []
        required_set = {str(item).strip() for item in required_items if str(item).strip()}
        if isinstance(properties, dict):
            for param_name, prop in properties.items():
                if not isinstance(prop, dict):
                    continue
                key = str(param_name).strip()
                if not key:
                    continue
                spec = specs.setdefault(key, {})
                if isinstance(prop.get("type"), str) and "type" not in spec:
                    spec["type"] = prop["type"]
                if key in required_set and "required" not in spec:
                    spec["required"] = True
                if isinstance(prop.get("description"), str) and prop["description"].strip() and "description" not in spec:
                    spec["description"] = prop["description"].strip()
                if isinstance(prop.get("examples"), list) and "examples" not in spec:
                    spec["examples"] = [item for item in prop["examples"] if isinstance(item, (str, int, float, bool))]
                if "default" in prop and "default" not in spec:
                    spec["default"] = prop["default"]
                if isinstance(prop.get("enum"), list) and "enum" not in spec:
                    spec["enum"] = prop["enum"]
                if isinstance(prop.get("pattern"), str) and prop["pattern"].strip() and "pattern" not in spec:
                    spec["pattern"] = prop["pattern"].strip()
                if isinstance(prop.get("format"), str) and prop["format"].strip() and "format" not in spec:
                    spec["format"] = prop["format"].strip()

        return specs
