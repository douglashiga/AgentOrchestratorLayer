"""
Seed Registry DB — Populate registry.db with domains, capabilities, and goals.

Run this script after recreating the Docker container or resetting the database.
Idempotent: safe to run multiple times (uses UPSERT logic in RegistryDB).

Usage:
    python scripts/seed_registry_db.py [--db-path registry.db]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from registry.db import RegistryDB

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Domain definitions
# ──────────────────────────────────────────────────────────────

FINANCE_DOMAIN_URL = os.getenv("FINANCE_DOMAIN_URL", "http://localhost:9100")

DOMAINS = [
    {
        "name": "finance",
        "type": "remote_http",
        "config": {
            "url": FINANCE_DOMAIN_URL,
            "timeout": 60.0,
        },
    },
    {
        "name": "general",
        "type": "local",
        "config": {
            "model": os.getenv("GENERAL_MODEL_NAME", "qwen2.5-coder:32b"),
        },
    },
]

# ──────────────────────────────────────────────────────────────
# General domain capabilities & goals
# ──────────────────────────────────────────────────────────────

GENERAL_CAPABILITIES = [
    {
        "capability": "chat",
        "description": "General conversation and help",
        "schema": {"message": "string"},
        "metadata": {
            "explanation_template": "{result[response]}",
            "planner_available": False,
            "domain_description": "General-purpose assistant interactions.",
            "domain_intent_hints": {
                "keywords": ["oi", "olá", "hello", "ajuda", "help", "conversar"],
                "examples": ["oi", "me ajuda com isso"],
            },
        },
    },
    {
        "capability": "list_capabilities",
        "description": "List available domains and capabilities grouped by domain",
        "schema": {"message": "string"},
        "metadata": {
            "explanation_template": "{result[response]}",
            "planner_available": False,
            "domain_description": "General-purpose assistant interactions.",
            "domain_intent_hints": {
                "keywords": ["o que voce faz", "what can you do", "listar capacidades"],
                "examples": ["quais capacidades voce tem?"],
            },
        },
    },
]

GENERAL_GOALS = [
    {
        "name": "CHAT",
        "description": "General conversation and help",
        "capabilities": ["chat"],
        "requires_domains": [],
        "hints": {
            "keywords": ["oi", "olá", "hello", "ajuda", "help", "conversar"],
            "examples": ["oi", "me ajuda com isso"],
        },
    },
    {
        "name": "LIST_CAPABILITIES",
        "description": "List available system capabilities",
        "capabilities": ["list_capabilities"],
        "requires_domains": [],
        "hints": {
            "keywords": ["o que voce faz", "what can you do", "listar capacidades"],
            "examples": ["quais capacidades voce tem?"],
        },
    },
]

# ──────────────────────────────────────────────────────────────
# Finance domain goals (static definition, mirrors server.py manifest)
# ──────────────────────────────────────────────────────────────

FINANCE_GOALS = [
    {
        "name": "GET_QUOTE",
        "description": "Get current stock price/quote for one or more symbols",
        "capabilities": ["get_stock_price"],
        "hints": {
            "keywords": [
                "qual o preco", "qual o preço", "qual o valor",
                "cotacao", "cotação", "quanto esta", "quanto está",
                "ticker", "stock price", "price of",
            ],
            "examples": ["qual o valor da petr4?", "quanto esta a vale3 hoje?", "price of aapl"],
        },
        "entities_schema": {
            "symbol_text": {"type": "string", "required": True, "description": "Stock symbol or company name"},
            "symbols_text": {"type": "array", "description": "Multiple stock symbols/company names if comparing"},
        },
    },
    {
        "name": "VIEW_HISTORY",
        "description": "Get historical OHLC price data for a symbol and period",
        "capabilities": ["get_historical_data"],
        "hints": {
            "keywords": ["historico", "histórico", "grafico", "gráfico", "serie historica", "dados historicos", "preco nos ultimos"],
            "examples": ["historico da petr4 no ultimo mes", "grafico de vale3 em 1 ano"],
        },
        "entities_schema": {
            "symbol_text": {"type": "string", "required": True, "description": "Stock symbol or company name"},
            "period_text": {"type": "string", "description": "Time period (e.g. 'ultimo mes', '1 ano', 'ytd')"},
            "interval_text": {"type": "string", "description": "Data interval (e.g. 'diário', 'semanal')"},
        },
    },
    {
        "name": "SCREEN_STOCKS",
        "description": "Screen or filter stocks by market criteria",
        "capabilities": ["get_stock_screener"],
        "hints": {
            "keywords": ["screener", "filtrar acoes", "filtrar ações", "ranking de acoes", "acoes por setor"],
            "examples": ["screener de acoes no brasil", "melhores acoes de tecnologia nos eua"],
        },
        "entities_schema": {
            "market_text": {"type": "string", "description": "Market name (e.g. 'brasil', 'eua', 'suécia')"},
            "sector_text": {"type": "string", "description": "Sector (e.g. 'tecnologia', 'financeiro')"},
        },
    },
    {
        "name": "TOP_MOVERS",
        "description": "Find top gaining or losing stocks for a market/period",
        "capabilities": ["get_top_gainers", "get_top_losers"],
        "hints": {
            "keywords": [
                "maiores altas", "maiores baixas", "maiores ganhos", "maiores perdas",
                "top gainers", "top losers", "acoes que mais subiram", "ações que mais subiram",
                "acoes que mais cairam", "ações que mais caíram",
            ],
            "examples": ["quais as maiores altas de hoje do bovespa?", "top losers no brasil hoje"],
        },
        "entities_schema": {
            "direction": {
                "type": "enum", "values": ["GAINERS", "LOSERS", "BOTH"],
                "required": True, "default": "BOTH",
                "description": "Whether to show top gainers, losers, or both",
                "capability_map": {"GAINERS": "get_top_gainers", "LOSERS": "get_top_losers", "BOTH": ["get_top_gainers", "get_top_losers"]},
            },
            "market_text": {"type": "string", "description": "Market name (e.g. 'bovespa', 'brasil', 'suécia')"},
            "period_text": {"type": "string", "description": "Time period (e.g. 'hoje', 'semana', 'mês')"},
        },
    },
    {
        "name": "DIVIDEND_ANALYSIS",
        "description": "Find high dividend yield stocks or dividend history for a symbol",
        "capabilities": ["get_top_dividend_payers", "get_dividend_history"],
        "hints": {
            "keywords": ["dividendos", "dividend yield", "maiores dividendos", "acoes pagadoras", "ações pagadoras", "historico de dividendos"],
            "examples": ["quais acoes pagam mais dividendos no brasil?", "dividendos da petr4"],
        },
        "entities_schema": {
            "focus": {
                "type": "enum", "values": ["RANKING", "HISTORY"],
                "required": True, "default": "RANKING",
                "description": "RANKING for top payers by market, HISTORY for a specific symbol",
                "capability_map": {"RANKING": "get_top_dividend_payers", "HISTORY": "get_dividend_history"},
            },
            "symbol_text": {"type": "string", "description": "Company name or ticker"},
            "market_text": {"type": "string", "description": "Market name"},
        },
    },
    {
        "name": "TECHNICAL_SCAN",
        "description": "Find stocks with technical signals (RSI, MACD) by market",
        "capabilities": ["get_technical_signals"],
        "hints": {
            "keywords": ["sinal tecnico", "sinal técnico", "rsi", "macd", "sobrevendido", "sobrecomprado"],
            "examples": ["acoes com rsi sobrevendido no brasil", "sinais macd no mercado americano"],
        },
    },
    {
        "name": "FUNDAMENTALS",
        "description": "Get fundamental data for a specific stock",
        "capabilities": ["get_fundamentals"],
        "hints": {
            "keywords": ["fundamentos", "fundamental", "valuation", "balanco da empresa", "balanço"],
            "examples": ["fundamentos da vale3", "valuation da petr4"],
        },
    },
    {
        "name": "COMPANY_PROFILE",
        "description": "Get company profile and business information",
        "capabilities": ["get_company_profile"],
        "hints": {
            "keywords": ["sobre a empresa", "company info", "setor da empresa", "perfil da empresa"],
            "examples": ["me fale sobre a empresa vale3", "company info da aapl"],
        },
    },
    {
        "name": "FINANCIAL_STATEMENTS",
        "description": "Get financial statements (DRE, balance sheet)",
        "capabilities": ["get_financial_statements"],
        "hints": {
            "keywords": ["demonstracoes financeiras", "financial statements", "dre", "balanco patrimonial"],
            "examples": ["demonstracoes financeiras da vale3", "financial statements de msft"],
        },
    },
    {
        "name": "OPTIONS_DATA",
        "description": "Get option chain or Greeks for a stock",
        "capabilities": ["get_option_chain", "get_option_greeks"],
        "hints": {
            "keywords": ["option chain", "cadeia de opcoes", "opcoes da acao", "opções da ação", "greeks", "delta gamma"],
            "examples": ["option chain da aapl", "gregas das opcoes de petr4"],
        },
        "entities_schema": {
            "focus": {
                "type": "enum", "values": ["CHAIN", "GREEKS"],
                "required": True, "default": "CHAIN",
                "capability_map": {"CHAIN": "get_option_chain", "GREEKS": "get_option_greeks"},
            },
            "symbol_text": {"type": "string", "description": "Company name or ticker"},
        },
    },
    {
        "name": "SEARCH_SYMBOL",
        "description": "Search ticker symbols by company name",
        "capabilities": ["search_by_name", "yahoo_search"],
        "hints": {
            "keywords": ["qual o ticker", "procurar ticker", "buscar simbolo", "buscar símbolo"],
            "examples": ["qual o ticker da petrobras?", "buscar simbolo da nordea"],
        },
        "entities_schema": {
            "query_text": {"type": "string", "required": True, "description": "Company name or keyword to search"},
        },
    },
    {
        "name": "PIPELINE_STATUS",
        "description": "Check data pipeline jobs and status",
        "capabilities": ["list_jobs", "get_job_status"],
        "hints": {
            "keywords": ["listar jobs", "jobs disponiveis", "jobs disponíveis", "pipeline jobs", "status do job"],
            "examples": ["quais jobs de dados estao disponiveis?", "qual o status do job earnings_sync?"],
        },
        "entities_schema": {
            "focus": {
                "type": "enum", "values": ["LIST", "STATUS"],
                "required": True, "default": "LIST",
                "capability_map": {"LIST": "list_jobs", "STATUS": "get_job_status"},
            },
            "job_name_text": {"type": "string", "description": "Job name (e.g. 'earnings_sync')"},
        },
    },
    {
        "name": "ACTIVE_STOCKS",
        "description": "Find most actively traded stocks by volume for a market",
        "capabilities": ["get_most_active_stocks"],
        "hints": {
            "keywords": ["mais negociadas", "mais ativas", "volume alto", "most active", "mais liquidez"],
            "examples": ["quais as acoes mais negociadas hoje?", "most active stocks in sweden"],
        },
        "entities_schema": {
            "market_text": {"type": "string", "description": "Market name"},
            "period_text": {"type": "string", "description": "Period (e.g. 'hoje', 'semana')"},
        },
    },
    {
        "name": "OVERSOLD_STOCKS",
        "description": "Find oversold stocks (low RSI) for a market",
        "capabilities": ["get_oversold_stocks"],
        "hints": {
            "keywords": ["sobrevendido", "oversold", "rsi baixo", "oportunidade de compra"],
            "examples": ["acoes sobrevendidas no brasil", "oversold stocks in sweden"],
        },
        "entities_schema": {
            "market_text": {"type": "string", "description": "Market name"},
        },
    },
    {
        "name": "OVERBOUGHT_STOCKS",
        "description": "Find overbought stocks (high RSI) for a market",
        "capabilities": ["get_overbought_stocks"],
        "hints": {
            "keywords": ["sobrecomprado", "overbought", "rsi alto", "excessivamente comprado"],
            "examples": ["acoes sobrecompradas no brasil", "overbought stocks in usa"],
        },
        "entities_schema": {
            "market_text": {"type": "string", "description": "Market name"},
        },
    },
    {
        "name": "ANALYST_VIEW",
        "description": "Get analyst consensus recommendations for a stock",
        "capabilities": ["get_analyst_recommendations"],
        "hints": {
            "keywords": ["recomendacao", "recomendação", "analistas", "consenso", "buy sell hold", "analyst recommendation"],
            "examples": ["qual a recomendacao dos analistas para petr4?", "analyst recommendations for aapl"],
        },
        "entities_schema": {
            "symbol_text": {"type": "string", "required": True, "description": "Stock symbol or company name"},
        },
    },
    {
        "name": "TECH_ANALYSIS",
        "description": "Get comprehensive technical analysis indicators for a stock",
        "capabilities": ["get_technical_analysis"],
        "hints": {
            "keywords": ["analise tecnica", "análise técnica", "indicadores tecnicos", "technical analysis"],
            "examples": ["analise tecnica da vale3", "technical analysis of aapl"],
        },
        "entities_schema": {
            "symbol_text": {"type": "string", "required": True, "description": "Stock symbol or company name"},
            "period_text": {"type": "string", "description": "Analysis period (e.g. '1 ano', '6 meses')"},
        },
    },
    {
        "name": "NEWS_SENTIMENT",
        "description": "Get news sentiment analysis for a stock",
        "capabilities": ["get_news_sentiment"],
        "hints": {
            "keywords": ["sentimento", "noticias", "notícias", "news sentiment", "humor do mercado"],
            "examples": ["qual o sentimento das noticias sobre petr4?", "news sentiment for tsla"],
        },
        "entities_schema": {
            "symbol_text": {"type": "string", "required": True, "description": "Stock symbol or company name"},
        },
    },
    {
        "name": "STOCK_DEEP_DIVE",
        "description": "Get comprehensive information about a stock",
        "capabilities": ["get_comprehensive_stock_info"],
        "hints": {
            "keywords": ["tudo sobre", "analise completa", "análise completa", "deep dive", "informacoes completas", "resumo completo"],
            "examples": ["me fale tudo sobre a petr4", "deep dive in aapl"],
        },
        "entities_schema": {
            "symbol_text": {"type": "string", "required": True, "description": "Stock symbol or company name"},
        },
    },
    {
        "name": "WHEEL_PUT_SCAN",
        "description": "Find put option candidates for the wheel strategy",
        "capabilities": ["get_wheel_put_candidates"],
        "hints": {
            "keywords": ["wheel puts", "puts para wheel", "candidatos wheel", "vender puts", "sell puts", "put candidates"],
            "examples": ["quais os melhores puts para wheel na nordea?", "wheel put candidates for aapl"],
        },
        "entities_schema": {
            "symbol_text": {"type": "string", "required": True, "description": "Stock symbol or company name"},
            "market_text": {"type": "string", "description": "Market (e.g. 'suécia', 'brasil')"},
        },
    },
    {
        "name": "WHEEL_PUT_RETURN",
        "description": "Calculate return for a specific wheel put position",
        "capabilities": ["get_wheel_put_return"],
        "hints": {
            "keywords": ["retorno do put", "rendimento put", "quanto vou ganhar", "put return", "wheel return"],
            "examples": ["qual o retorno do put de nordea strike 120?", "wheel put return for aapl strike 180"],
        },
        "entities_schema": {
            "symbol_text": {"type": "string", "required": True, "description": "Stock symbol or company name"},
            "strike_text": {"type": "string", "description": "Strike price"},
            "expiry_text": {"type": "string", "description": "Expiry date"},
            "premium_text": {"type": "string", "description": "Premium received"},
        },
    },
    {
        "name": "WHEEL_CALL_SCAN",
        "description": "Find covered call candidates for the wheel strategy after assignment",
        "capabilities": ["get_wheel_covered_call_candidates"],
        "hints": {
            "keywords": ["covered calls", "calls cobertas", "wheel calls", "vender calls", "call candidates"],
            "examples": ["covered calls para nordea com custo medio de 120?", "wheel call candidates for aapl cost basis 180"],
        },
        "entities_schema": {
            "symbol_text": {"type": "string", "required": True, "description": "Stock symbol or company name"},
            "average_cost_text": {"type": "string", "description": "Average cost basis"},
        },
    },
    {
        "name": "WHEEL_CAPACITY",
        "description": "Calculate how many wheel contracts you can sell given capital",
        "capabilities": ["get_wheel_contract_capacity"],
        "hints": {
            "keywords": ["quantos contratos", "capacidade capital", "quanto posso vender", "contract capacity"],
            "examples": ["quantos contratos posso vender de nordea com 100000 sek?"],
        },
        "entities_schema": {
            "symbol_text": {"type": "string", "required": True, "description": "Stock symbol or company name"},
            "capital_text": {"type": "string", "description": "Capital amount"},
        },
    },
    {
        "name": "WHEEL_MULTI_PLAN",
        "description": "Build a diversified wheel strategy plan across multiple stocks",
        "capabilities": ["build_wheel_multi_stock_plan"],
        "hints": {
            "keywords": ["plano wheel multiplos", "carteira wheel", "distribuir capital", "multi stock plan", "diversificar wheel"],
            "examples": ["monte um plano wheel com 500000 sek", "build wheel plan across multiple stocks"],
        },
        "entities_schema": {
            "capital_text": {"type": "string", "required": True, "description": "Capital amount"},
            "symbols_text": {"type": "array", "description": "Specific symbols to consider (optional)"},
            "market_text": {"type": "string", "description": "Market"},
        },
    },
    {
        "name": "WHEEL_RISK",
        "description": "Analyze risk of a wheel put position (downside scenarios)",
        "capabilities": ["analyze_wheel_put_risk"],
        "hints": {
            "keywords": ["risco do put", "risco wheel", "analisar risco", "put risk", "wheel risk analysis", "downside risk"],
            "examples": ["qual o risco do wheel put na nordea?", "analyze wheel risk for aapl"],
        },
        "entities_schema": {
            "symbol_text": {"type": "string", "required": True, "description": "Stock symbol or company name"},
            "market_text": {"type": "string", "description": "Market"},
        },
    },
]


def seed_registry(db_path: str) -> None:
    """Seed the registry database with all domains, capabilities, and goals."""
    db = RegistryDB(db_path=db_path)
    logger.info("Seeding registry DB at: %s", db_path)

    # 1. Register domains
    for domain_def in DOMAINS:
        db.register_domain(
            name=domain_def["name"],
            domain_type=domain_def["type"],
            config=domain_def["config"],
        )
    logger.info("Registered %d domains", len(DOMAINS))

    # 2. Register general domain capabilities
    for cap in GENERAL_CAPABILITIES:
        db.register_capability(
            domain_name="general",
            capability=cap["capability"],
            description=cap["description"],
            schema=cap.get("schema"),
            metadata=cap.get("metadata"),
        )
    logger.info("Registered %d general capabilities", len(GENERAL_CAPABILITIES))

    # 3. Register general domain goals
    for goal in GENERAL_GOALS:
        db.register_goal(
            domain_name="general",
            name=goal["name"],
            description=goal.get("description", ""),
            capabilities=goal.get("capabilities", []),
            requires_domains=goal.get("requires_domains", []),
            hints=goal.get("hints", {}),
            entities_schema=goal.get("entities_schema", {}),
        )
    logger.info("Registered %d general goals", len(GENERAL_GOALS))

    # 4. Register finance domain goals (static — capabilities come from manifest sync)
    for goal in FINANCE_GOALS:
        db.register_goal(
            domain_name="finance",
            name=goal["name"],
            description=goal.get("description", ""),
            capabilities=goal.get("capabilities", []),
            requires_domains=goal.get("requires_domains", []),
            hints=goal.get("hints", {}),
            entities_schema=goal.get("entities_schema", {}),
        )
    logger.info("Registered %d finance goals", len(FINANCE_GOALS))

    # Summary
    domains = db.list_domains()
    all_caps = db.list_capabilities()
    all_goals = db.list_goals()
    logger.info(
        "Seed complete: %d domains, %d capabilities, %d goals",
        len(domains), len(all_caps), len(all_goals),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the registry database")
    parser.add_argument("--db-path", default="registry.db", help="Path to registry.db")
    args = parser.parse_args()
    seed_registry(args.db_path)
