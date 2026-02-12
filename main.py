"""
Agent Orchestrator — Main CLI Entrypoint.

Wires all layers and runs the interactive CLI loop.
"""

import logging
import sys

logger = logging.getLogger(__name__)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from entry.cli import CLIAdapter
from conversation.manager import ConversationManager
from intent.adapter import IntentAdapter
from orchestrator.orchestrator import Orchestrator
from registry.domain_registry import DomainRegistry
from domains.finance.handler import FinanceDomainHandler
from domains.general.handler import GeneralDomainHandler
from skills.gateway import SkillGateway
from skills.registry import SkillRegistry
from skills.implementations.mcp_adapter import MCPAdapter

# ─── Configuration ──────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
OLLAMA_INTENT_MODEL = "llama3.1:8b"       # Fast model for intent extraction (JSON)
OLLAMA_CHAT_MODEL = "qwen2.5-coder:32b"   # Full model for general conversation
MCP_URL = "http://localhost:8000/sse"
DB_PATH = "conversations.db"
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


def build_pipeline() -> tuple[CLIAdapter, ConversationManager, IntentAdapter, Orchestrator]:
    """Wire all layers together."""
    # Skills
    mcp_adapter = MCPAdapter(mcp_url=MCP_URL)
    skill_registry = SkillRegistry()
    skill_registry.register("mcp_finance", mcp_adapter)
    skill_gateway = SkillGateway(skill_registry)

    # Domains
    finance_handler = FinanceDomainHandler(skill_gateway=skill_gateway)
    general_handler = GeneralDomainHandler(ollama_url=OLLAMA_URL, model=OLLAMA_CHAT_MODEL)
    domain_registry = DomainRegistry()
    domain_registry.register("finance", finance_handler)
    domain_registry.register("general", general_handler)

    # Core
    orchestrator = Orchestrator(domain_registry=domain_registry)
    intent_adapter = IntentAdapter(ollama_url=OLLAMA_URL, model=OLLAMA_INTENT_MODEL)
    conversation_manager = ConversationManager(db_path=DB_PATH)
    cli_adapter = CLIAdapter()

    return cli_adapter, conversation_manager, intent_adapter, orchestrator


def render_decision(decision) -> None:
    """Render a Decision to the CLI using Rich."""
    if decision.success:
        # Chat responses — just show the text
        if decision.action == "chat":
            console.print()
            console.print(Panel(
                Text(decision.explanation, style="white"),
                title="🤖 Assistant",
                border_style="cyan",
                box=box.ROUNDED,
            ))
            return

        # Header
        console.print()
        console.print(Panel(
            Text(decision.explanation, style="bold green"),
            title="✅ Result",
            border_style="green",
            box=box.ROUNDED,
        ))

        # Result data table
        result = decision.result
        market_ctx = result.pop("_market_context", None)

        if result:
            table = Table(
                title="📊 Data",
                box=box.SIMPLE_HEAVY,
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Field", style="bold white")
            table.add_column("Value", style="white")

            for key, value in result.items():
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        table.add_row(f"  {sub_key}", str(sub_val))
                elif isinstance(value, list):
                    table.add_row(key, f"[{len(value)} items]")
                else:
                    table.add_row(key, str(value))

            console.print(table)

        # Market context
        if market_ctx:
            ctx_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            ctx_table.add_column("", style="bold dim")
            ctx_table.add_column("", style="dim")
            ctx_table.add_row("🏦 Exchange", f"{market_ctx.get('exchange', '')} ({market_ctx.get('country', market_ctx['market'])})")
            ctx_table.add_row("💱 Currency", f"{market_ctx.get('currency_symbol', '')} {market_ctx['currency']}")
            ctx_table.add_row("🕐 Hours", f"{market_ctx.get('trading_hours', '')} ({market_ctx.get('timezone', '')})")
            ctx_table.add_row("📦 Lot/Settlement", f"{market_ctx.get('lot_size', '')} | {market_ctx.get('settlement', 'T+2')}")
            ctx_table.add_row("📋 Tax", f"{market_ctx.get('tax_model', '')} ({market_ctx.get('tax_rate_gains', '')})")
            if market_ctx.get("tax_notes"):
                ctx_table.add_row("💡 Notes", market_ctx["tax_notes"])
            console.print(Panel(ctx_table, title="📍 Market Context", border_style="dim", box=box.ROUNDED))
    else:
        console.print()
        console.print(Panel(
            Text(f"Error: {decision.error}", style="bold red"),
            title="❌ Failed",
            border_style="red",
            box=box.ROUNDED,
        ))
        if decision.explanation:
            console.print(Text(f"  ℹ️  {decision.explanation}", style="dim"))


def render_intent_debug(intent) -> None:
    """Show extracted intent in debug panel."""
    table = Table(box=box.MINIMAL, show_header=False, padding=(0, 1))
    table.add_column("Key", style="bold yellow")
    table.add_column("Value", style="white")
    table.add_row("Domain", intent.domain)
    table.add_row("Action", intent.action)
    table.add_row("Parameters", str(intent.parameters))
    table.add_row("Confidence", f"{intent.confidence:.0%}")

    console.print(Panel(table, title="🧠 Intent (LLM)", border_style="yellow", box=box.ROUNDED))


def main() -> None:
    """Main CLI loop."""
    setup_logging()

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
        cli, conversation, intent_adapter, orchestrator = build_pipeline()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize pipeline:[/] {e}")
        sys.exit(1)

    console.print(f"[dim]Session: {cli.session_id}[/dim]")
    console.print(f"[dim]Domains: {orchestrator.domain_registry.registered_domains}[/dim]")
    console.print()

    while True:
        try:
            # Step 1: Entry — read input
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
                    intent = intent_adapter.extract(entry_request.input_text, history)
                except ValueError as e:
                    console.print(f"\n[bold red]Intent extraction failed:[/] {e}")
                    conversation.save(entry_request.session_id, "user", entry_request.input_text)
                    conversation.save(entry_request.session_id, "assistant", f"Error: {e}")
                    continue

            # Show intent debug
            render_intent_debug(intent)

            # Step 4-7: Orchestrator → Domain → Skills → Strategy → Decision
            with console.status("[green]Executing...[/green]", spinner="dots"):
                decision = orchestrator.process(intent)

            # Step 8: Render decision
            render_decision(decision)

            # Step 9: Persist conversation
            conversation.save(entry_request.session_id, "user", entry_request.input_text)
            response_text = decision.explanation if decision.success else f"Error: {decision.error}"
            conversation.save(entry_request.session_id, "assistant", response_text)

            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye! 👋[/dim]")
            break
        except EOFError:
            console.print("\n[dim]Goodbye! 👋[/dim]")
            break


if __name__ == "__main__":
    main()
