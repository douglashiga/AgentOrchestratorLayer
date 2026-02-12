"""
Agent Orchestrator — Main CLI Entrypoint.

Wires all layers and runs the interactive CLI loop.
"""

import argparse
import json
import logging
import sys
import os
import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from entry.cli import CLIAdapter
from conversation.manager import ConversationManager
from intent.adapter import IntentAdapter
from planner.service import PlannerService
from execution.engine import ExecutionEngine
from orchestrator.orchestrator import Orchestrator
from registry.domain_registry import HandlerRegistry
from registry.db import RegistryDB
from registry.loader import RegistryLoader
from skills.gateway import SkillGateway
from skills.registry import SkillRegistry
from skills.implementations.mcp_adapter import MCPAdapter
from models.selector import ModelSelector

# ─── Configuration ──────────────────────────────────────────────

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_INTENT_MODEL = "llama3.1:8b"
OLLAMA_CHAT_MODEL = "qwen2.5-coder:32b"
MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/sse")
DB_PATH = os.getenv("DB_PATH", "agent.db")
REGISTRY_DB_PATH = os.getenv("REGISTRY_DB_PATH", "registry.db")
LOG_LEVEL = logging.INFO

# ─── Rich Console ───────────────────────────────────────────────

console = Console()


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def preload_models() -> None:
    """Warm up Ollama models so first query is fast."""
    import httpx
    for model in (OLLAMA_INTENT_MODEL, OLLAMA_CHAT_MODEL):
        try:
            httpx.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": model, "messages": [], "keep_alive": "10m"},
                timeout=30.0,
            )
            logger.info("Preloaded model: %s", model)
        except Exception:
            logger.warning("Could not preload model: %s", model)


def build_pipeline() -> tuple[CLIAdapter, ConversationManager, IntentAdapter, PlannerService, ExecutionEngine]:
    """Wire all layers together using Dynamic Registry."""
    # Shared
    model_selector = ModelSelector(ollama_url=OLLAMA_URL)

    # Skills (still needed for local domains)
    mcp_adapter = MCPAdapter(mcp_url=MCP_URL)
    skill_registry = SkillRegistry()
    skill_registry.register("mcp_finance", mcp_adapter)
    skill_gateway = SkillGateway(skill_registry)

    # Context for Local Handlers
    loader_context = {
        "skill_gateway": skill_gateway,
        "model_selector": model_selector
    }

    # Dynamic Registry
    db = RegistryDB(db_path=REGISTRY_DB_PATH)
    runtime_registry = HandlerRegistry()
    loader = RegistryLoader(db, runtime_registry)

    # Ensure local defaults exist (Seed/Update local domains and capabilities)
    logger.info("Syncing local domain defaults...")
    loader.sync_local_to_db()
        # Initial capability sync for local is manual/hardcoded in loader for now,
        # or we rely on intent adapter to know them.
        # Ideally, local handlers should describe themselves too, but for now we assume they are safe.

    # Load All Domains
    loader.load_all(loader_context)

    # Core
    orchestrator = Orchestrator(domain_registry=runtime_registry, model_selector=model_selector)
    
    # Get all registered capabilities for the Intent Adapter
    registered_capabilities = runtime_registry.registered_capabilities
    intent_adapter = IntentAdapter(model_selector=model_selector, initial_capabilities=registered_capabilities)
    
    planner_service = PlannerService()
    execution_engine = ExecutionEngine(orchestrator=orchestrator)
    conversation_manager = ConversationManager(db_path=DB_PATH)
    cli_adapter = CLIAdapter()

    return cli_adapter, conversation_manager, intent_adapter, planner_service, execution_engine


def render_result(intent, output) -> None:
    """Render DomainOutput to the CLI using Rich."""
    if output.status == "success":
        # Chat responses — just show the text
        if intent.capability == "chat":
            console.print()
            console.print(Panel(
                Text(output.explanation, style="white"),
                title="🤖 Assistant",
                border_style="cyan",
                box=box.ROUNDED,
            ))
            return

        # Header
        console.print()
        console.print(Panel(
            Text(output.explanation, style="bold green"),
            title="✅ Result",
            border_style="green",
            box=box.ROUNDED,
        ))

        # Result data table
        result = output.result
        # Copy to avoid mutating frozen pydantic if meaningful, but result is a dict
        result_view = result.copy()
        market_ctx = result_view.pop("_market_context", None)

        if result_view:
            table = Table(
                title="📊 Data",
                box=box.SIMPLE_HEAVY,
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Field", style="bold white")
            table.add_column("Value", style="white")

            for key, value in result_view.items():
                if isinstance(value, dict):
                    # Check if it's a comparison dictionary (like symbol -> metrics)
                    # Simple heuristic: if value is dict, maybe flatten or JSON dump
                    table.add_row(key, json.dumps(value, indent=2))
                elif isinstance(value, list):
                    # Check if list of dicts (like gainers) to show specialized table?
                    # For now just summary
                     table.add_row(key, f"[{len(value)} items] " + str(value)[:50] + "...")
                else:
                    table.add_row(key, str(value))

            console.print(table)

        # Market context
        if market_ctx:
            ctx_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            ctx_table.add_column("", style="bold dim")
            ctx_table.add_column("", style="dim")
            ctx_table.add_row("🏦 Exchange", f"{market_ctx.get('exchange', '')} ({market_ctx.get('country', market_ctx.get('market', ''))})")
            ctx_table.add_row("💱 Currency", f"{market_ctx.get('currency_symbol', '')} {market_ctx.get('currency', '')}")
            ctx_table.add_row("🕐 Hours", f"{market_ctx.get('trading_hours', '')} ({market_ctx.get('timezone', '')})")
            ctx_table.add_row("📦 Lot/Settlement", f"{market_ctx.get('lot_size', '')} | {market_ctx.get('settlement', 'T+2')}")
            ctx_table.add_row("📋 Tax", f"{market_ctx.get('tax_model', '')} ({market_ctx.get('tax_rate_gains', '')})")
            if market_ctx.get("tax_notes"):
                ctx_table.add_row("💡 Notes", market_ctx["tax_notes"])
            console.print(Panel(ctx_table, title="📍 Market Context", border_style="dim", box=box.ROUNDED))

    elif output.status == "clarification":
        # Clarification needed
        console.print()
        console.print(Panel(
            Text(output.explanation, style="bold yellow"),
            title="❓ Clarification Needed",
            border_style="yellow",
            box=box.ROUNDED,
        ))

    else:
        # Failure case
        error_msg = output.metadata.get("error", "Unknown error")
        console.print()
        console.print(Panel(
            Text(f"Error: {error_msg}", style="bold red"),
            title="❌ Failed",
            border_style="red",
            box=box.ROUNDED,
        ))
        if output.explanation:
            console.print(Text(f"  ℹ️  {output.explanation}", style="dim"))


def render_intent_debug(intent) -> None:
    """Show extracted intent in debug panel."""
    table = Table(box=box.MINIMAL, show_header=False, padding=(0, 1))
    table.add_column("Key", style="bold yellow")
    table.add_column("Value", style="white")
    table.add_row("Domain", intent.domain)
    table.add_row("Capability", intent.capability)
    table.add_row("Original Query", intent.original_query)
    table.add_row("Parameters", str(intent.parameters))
    table.add_row("Confidence", f"{intent.confidence:.0%}")

    console.print(Panel(table, title="🧠 Intent (LLM)", border_style="yellow", box=box.ROUNDED))


async def run_agent_loop() -> None:
    """Interactive Agent Loop."""
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]Agent Orchestrator[/bold cyan]\n"
            f"[dim]Intent: {OLLAMA_INTENT_MODEL} • Chat: {OLLAMA_CHAT_MODEL}[/dim]\n"
            "[dim]Type your question or 'exit' to quit[/dim]"
        ),
        title="🤖",
        border_style="cyan",
        box=box.DOUBLE,
    ))

    # Preload models into GPU memory
    with console.status("[yellow]Loading models...[/yellow]", spinner="dots"):
        preload_models()

    try:
        cli, conversation, intent_adapter, planner, engine = build_pipeline()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize pipeline:[/] {e}")
        sys.exit(1)

    console.print(f"[dim]Session: {cli.session_id}[/dim]")
    console.print(f"[dim]Domains: {engine.orchestrator.domain_registry.registered_domains}[/dim]")
    # console.print(f"[dim]Capabilities: {len(engine.orchestrator.domain_registry.registered_capabilities)}[/dim]")
    console.print()

    while True:
        try:
            # Step 1: Entry — read input
            # Use run_in_executor for blocking input if needed, but console.input is robust enough for CLI
            raw_input = console.input("[bold cyan]You → [/]")

            if raw_input.strip().lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye! 👋[/dim]")
                break

            if not raw_input.strip():
                continue

            entry_request = cli.read_input(raw_input)

            # Step 2: Conversation — get history
            history = conversation.get_history(entry_request.session_id)

            # Step 3: Intent Adapter — extract intent (LLM)
            with console.status("[yellow]Thinking...[/yellow]", spinner="dots"):
                try:
                    # Intent extraction is still sync/blocking (HTTP client), which is fine for CLI
                    intent = intent_adapter.extract(entry_request.input_text, history)
                except ValueError as e:
                    console.print(f"\n[bold red]Intent extraction failed:[/] {e}")
                    conversation.save(entry_request.session_id, "user", entry_request.input_text)
                    conversation.save(entry_request.session_id, "assistant", f"Error: {e}")
                    continue

            # Show intent debug
            render_intent_debug(intent)

            # Step 4: Planner (Decomposition)
            with console.status("[blue]Planning...[/blue]", spinner="dots"):
                plan = planner.generate_plan(intent)
                logger.info("Plan generated: %d steps", len(plan.steps))

            # Step 5-8: Execution Engine (Orchestrator + Domain + Model)
            with console.status("[green]Executing...[/green]", spinner="dots"):
                # await the async engine
                output = await engine.execute_plan(plan, original_intent=intent)

            # Step 9: Render output
            render_result(intent, output)

            # Step 9: Persist conversation
            conversation.save(entry_request.session_id, "user", entry_request.input_text)
            response_text = output.explanation if output.status == "success" else f"Error: {output.metadata.get('error', 'unknown')}"
            conversation.save(entry_request.session_id, "assistant", response_text)

            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye! 👋[/dim]")
            break
        except EOFError:
            console.print("\n[dim]Goodbye! 👋[/dim]")
            break


# ─── Admin Commands ─────────────────────────────────────────────

def admin_list_domains():
    db = RegistryDB(db_path=REGISTRY_DB_PATH)
    domains = db.list_domains()
    
    table = Table(title="Registered Domains")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Config", style="dim")
    table.add_column("Enabled", style="green")
    
    for d in domains:
        table.add_row(d["name"], d["type"], d["config"], str(bool(d["enabled"])))
        
    console.print(table)


def admin_add_domain(name, dtype, config_str):
    try:
        config = json.loads(config_str)
    except json.JSONDecodeError:
        console.print("[bold red]Error:[/] Config must be valid JSON.")
        return

    db = RegistryDB(db_path=REGISTRY_DB_PATH)
    db.register_domain(name, dtype, config)
    console.print(f"[bold green]Registered domain:[/] {name}")


def admin_sync_domain(name):
    db = RegistryDB(db_path=REGISTRY_DB_PATH)
    runtime_registry = HandlerRegistry() # Dummy for loader
    loader = RegistryLoader(db, runtime_registry)
    
    if loader.sync_capabilities(name):
        console.print(f"[bold green]Successfully synced capabilities for:[/] {name}")
    else:
        console.print(f"[bold red]Failed to sync capabilities for:[/] {name}")


def main() -> None:
    """Entrypoint with CLI args."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Agent Orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run Agent
    subparsers.add_parser("run", help="Run interactive agent")
    
    # Domain Management
    subparsers.add_parser("domain-list", help="List registered domains")
    
    add_parser = subparsers.add_parser("domain-add", help="Register a new domain")
    add_parser.add_argument("name", help="Domain name")
    add_parser.add_argument("type", choices=["local", "remote_http"], help="Domain type")
    add_parser.add_argument("config", help="JSON config string (e.g. '{\"url\": \"...\"}')")
    
    sync_parser = subparsers.add_parser("domain-sync", help="Sync capabilities from remote domain")
    sync_parser.add_argument("name", help="Domain name")
    
    args = parser.parse_args()
    
    if args.command == "domain-list":
        admin_list_domains()
    elif args.command == "domain-add":
        admin_add_domain(args.name, args.type, args.config)
    elif args.command == "domain-sync":
        admin_sync_domain(args.name)
    elif args.command == "run" or args.command is None:
        try:
            asyncio.run(run_agent_loop())
        except KeyboardInterrupt:
            pass
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
