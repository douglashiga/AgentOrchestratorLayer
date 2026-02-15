"""
End-to-end test simulating the full query processing pipeline.
"""

from domains.finance.handler import FinanceDomainHandler
from domains.finance.symbol_normalizer import SymbolNormalizer

# Standard symbol aliases for testing
TEST_SYMBOL_ALIASES = {
    "PETRO": "PETR4.SA",
    "PETRO4": "PETR4.SA",
    "PETROBRAS": "PETR4.SA",
    "VALE": "VALE3.SA",
    "ITAU": "ITUB4.SA",
    "ITAU UNIBANCO": "ITUB4.SA",
    "BBAS": "BBAS3.SA",
    "BANCO DO BRASIL": "BBAS3.SA",
    "MGLU": "MGLU3.SA",
    "MAGAZINE LUIZA": "MGLU3.SA",
    "BBDC": "BBDC4.SA",
    "BRADESCO": "BBDC4.SA",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "MSFT": "MSFT",
    "NVDA": "NVDA",
    "NORDEA": "NDA-SE.ST",
    "TELIA": "TELIA.ST",
}


class MockGateway:
    """Mock gateway that returns realistic stock price data."""
    
    def __init__(self):
        self.call_count = 0
        self.calls = []
    
    def execute(self, skill_name, parameters):
        self.call_count += 1
        self.calls.append({"skill": skill_name, "params": parameters})
        
        # Return mock data based on symbol
        symbol = parameters.get("symbol", "")
        
        if symbol == "PETR4.SA":
            return {"success": True, "data": {"symbol": "PETR4.SA", "price": 36.89, "currency": "BRL"}}
        elif symbol == "TELIA.ST":
            return {"success": True, "data": {"symbol": "TELIA.ST", "price": 45.25, "currency": "SEK"}}
        elif symbol == "VALE3.SA":
            return {"success": True, "data": {"symbol": "VALE3.SA", "price": 87.03, "currency": "BRL"}}
        else:
            return {"success": False, "error": f"Symbol {symbol} not found"}


def test_end_to_end_with_comma():
    """Test full pipeline with comma."""
    gateway = MockGateway()
    normalizer = SymbolNormalizer(aliases=TEST_SYMBOL_ALIASES)
    handler = FinanceDomainHandler(skill_gateway=gateway, registry=None, symbol_normalizer=normalizer)
    
    query = "qual o valor da petr4, TELIA.ST e vale3?"
    symbols = handler._infer_symbols_from_query_text(query)
    
    print(f"\n✓ WITH COMMA: '{query}'")
    print(f"  Extracted symbols: {symbols}")
    print(f"  Expected: ['PETR4.SA', 'TELIA.ST', 'VALE3.SA']")
    
    assert len(symbols) == 3, f"Expected 3 symbols, got {len(symbols)}: {symbols}"
    assert symbols == ['PETR4.SA', 'TELIA.ST', 'VALE3.SA'] or \
           set(symbols) == {'PETR4.SA', 'TELIA.ST', 'VALE3.SA'}, \
           f"Got wrong symbols: {symbols}"


def test_end_to_end_without_comma():
    """Test full pipeline without comma."""
    gateway = MockGateway()
    normalizer = SymbolNormalizer(aliases=TEST_SYMBOL_ALIASES)
    handler = FinanceDomainHandler(skill_gateway=gateway, registry=None, symbol_normalizer=normalizer)
    
    query = "qual o valor da petr4  TELIA.ST e vale3?"
    symbols = handler._infer_symbols_from_query_text(query)
    
    print(f"\n✓ WITHOUT COMMA: '{query}'")
    print(f"  Extracted symbols: {symbols}")
    print(f"  Expected: ['PETR4.SA', 'TELIA.ST', 'VALE3.SA']")
    
    assert len(symbols) == 3, f"Expected 3 symbols, got {len(symbols)}: {symbols}"
    assert symbols == ['PETR4.SA', 'TELIA.ST', 'VALE3.SA'] or \
           set(symbols) == {'PETR4.SA', 'TELIA.ST', 'VALE3.SA'}, \
           f"Got wrong symbols: {symbols}"


def test_flow_resolve_with_comma():
    """Test symbol resolution flow with comma."""
    gateway = MockGateway()
    normalizer = SymbolNormalizer(aliases=TEST_SYMBOL_ALIASES)
    handler = FinanceDomainHandler(skill_gateway=gateway, registry=None, symbol_normalizer=normalizer)
    
    result = handler._flow_resolve_symbol_list(
        step={"param": "symbols", "search_capability": "search_symbol"},
        params={},
        original_query="qual o valor da petr4, TELIA.ST e vale3?"
    )
    
    print(f"\n✓ FLOW WITH COMMA:")
    print(f"  Result: {result}")
    print(f"  Symbols: {result.get('symbols', [])}")
    
    assert isinstance(result, dict)
    assert "symbols" in result


def test_flow_resolve_without_comma():
    """Test symbol resolution flow without comma."""
    gateway = MockGateway()
    normalizer = SymbolNormalizer(aliases=TEST_SYMBOL_ALIASES)
    handler = FinanceDomainHandler(skill_gateway=gateway, registry=None, symbol_normalizer=normalizer)
    
    result = handler._flow_resolve_symbol_list(
        step={"param": "symbols", "search_capability": "search_symbol"},
        params={},
        original_query="qual o valor da petr4  TELIA.ST e vale3?"
    )
    
    print(f"\n✓ FLOW WITHOUT COMMA:")
    print(f"  Result: {result}")
    print(f"  Symbols: {result.get('symbols', [])}")
    
    assert isinstance(result, dict)
    assert "symbols" in result


if __name__ == "__main__":
    test_end_to_end_with_comma()
    test_end_to_end_without_comma()
    test_flow_resolve_with_comma()
    test_flow_resolve_without_comma()
    print("\n✅ All end-to-end tests passed!")
