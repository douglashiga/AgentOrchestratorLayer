"""
Registry Loader — Syncs DB Configuration with Runtime Registry.

Responsibility:
- Read active domains from RegistryDB
- Instantiate appropriate handlers (Mobile/Local)
- Register to HandlerRegistry
"""

import logging
import json
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
                        timeout=config.get("timeout", 30.0)
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
                            model_name=config.get("model", "qwen2.5-coder:32b")
                        )
                
                if handler:
                    self.registry.register_domain(name, handler)
                    
                    # Register capabilities
                    caps = self.db.list_capabilities(name)
                    logger.info("Domain '%s' has %d capabilities in DB", name, len(caps))
                    for cap in caps:
                        # Extract metadata from JSON string
                        cap_metadata = json.loads(cap["metadata"]) if cap.get("metadata") else {}
                        # Backwards compatibility: ensure schema is in metadata
                        cap_metadata["schema"] = json.loads(cap["input_schema"]) if cap.get("input_schema") else {}
                        
                        self.registry.register_capability(
                            cap["capability"], 
                            handler, 
                            metadata=cap_metadata
                        )
                else:
                    logger.warning("No handler factory for domain '%s' (type=%s)", name, dtype)
                        
            except Exception as e:
                logger.error("Failed to load domain '%s': %s", name, e, exc_info=True)

    def sync_local_to_db(self) -> None:
        """Helper to seed DB with default local config (for migration)."""
        logger.info("RegistryLoader: Seeding local domain defaults...")
        # General Domain
        self.db.register_domain("general", "local", {"model": "qwen2.5-coder:32b"})
        self.db.register_capability(
            domain_name="general",
            capability="chat",
            description="General conversation and help",
            schema={"message": "string"},
            metadata={"explanation_template": "{result[response]}"}
        )

        # Finance Domain (Optional local fallback, usually remote now)
        self.db.register_domain("finance", "local", {})
        
        # Apply metadata overrides from code to DB
        try:
             from domains.finance.server import METADATA_OVERRIDES
             
             # Fetch existing capabilities to preserve schema/description
             existing_rows = self.db.list_capabilities("finance")
             existing_caps = {row["capability"]: row for row in existing_rows}
             
             for cap_name, overrides in METADATA_OVERRIDES.items():
                 existing = existing_caps.get(cap_name)
                 
                 if existing:
                     # Merge metadata
                     current_meta = json.loads(existing["metadata"]) if existing.get("metadata") else {}
                     current_meta.update(overrides)
                     
                     # Extract schema properly from DB row (key is input_schema)
                     current_schema = json.loads(existing["input_schema"]) if existing.get("input_schema") else {}
                     
                     self.db.register_capability(
                         domain_name="finance",
                         capability=cap_name,
                         description=existing.get("description", ""),
                         schema=current_schema,
                         metadata=current_meta
                     )
                     logger.debug("Updated metadata for %s", cap_name)
                 else:
                     # Optional: Register if strictly local and missing? 
                     # For now, we only enrich existing ones to avoid schema-less ghosts.
                     pass

        except Exception as e:
             logger.warning("Failed to sync finance metadata overrides: %s", e)

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
                self.db.register_capability(
                    domain_name=domain_name,
                    capability=cap["name"],
                    description=cap.get("description", ""),
                    schema=cap.get("schema"),
                    metadata=cap.get("metadata", {})
                )
            
            logger.info("Synced %d capabilities for domain %s (including metadata)", len(capabilities), domain_name)
            return True
            
        except Exception as e:
            logger.error("Failed to sync capabilities for %s: %s", domain_name, e)
            return False
