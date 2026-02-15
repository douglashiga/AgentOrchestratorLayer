"""
Test to verify symbol extraction with and without commas.

User reported:
- "qual o valor da petr4, TELIA.ST e vale3?" → Works (3 symbols)
- "qual o valor da petr4  TELIA.ST e vale3?" → Only returns TELIA

This test investigates the root cause.
"""

from domains.finance.handler import FinanceDomainHandler


class DummyGateway:
    def execute(self, skill_name, parameters):
        return {"success": True, "data": {}}


def test_symbol_extraction_with_comma():
    """Test extraction with comma separator."""
    handler = FinanceDomainHandler(skill_gateway=DummyGateway(), registry=None)
    
    query = "qual o valor da petr4, TELIA.ST e vale3?"
    symbols = handler._infer_symbols_from_query_text(query)
    
    print(f"\nCOM VÍRGULA: '{query}'")
    print(f"  Extracted: {symbols}")
    
    # Should contain all 3 symbols (PETR4.SA from B3, TELIA.ST from explicit, VALE3.SA from B3)
    assert "PETR4.SA" in symbols
    assert "TELIA.ST" in symbols
    assert "VALE3.SA" in symbols


def test_symbol_extraction_without_comma():
    """Test extraction without comma separator."""
    handler = FinanceDomainHandler(skill_gateway=DummyGateway(), registry=None)
    
    query = "qual o valor da petr4  TELIA.ST e vale3?"
    symbols = handler._infer_symbols_from_query_text(query)
    
    print(f"\nSEM VÍRGULA: '{query}'")
    print(f"  Extracted: {symbols}")
    
    # Should contain all 3 symbols
    assert "PETR4.SA" in symbols
    assert "TELIA.ST" in symbols
    assert "VALE3.SA" in symbols


def test_symbol_extraction_without_comma_single_space():
    """Test extraction without comma and single space."""
    handler = FinanceDomainHandler(skill_gateway=DummyGateway(), registry=None)
    
    query = "qual o valor da petr4 TELIA.ST e vale3?"
    symbols = handler._infer_symbols_from_query_text(query)
    
    print(f"\nSEM VÍRGULA, ESPAÇO SIMPLES: '{query}'")
    print(f"  Extracted: {symbols}")
    
    # Should contain all 3 symbols
    assert "PETR4.SA" in symbols
    assert "TELIA.ST" in symbols
    assert "VALE3.SA" in symbols


if __name__ == "__main__":
    test_symbol_extraction_with_comma()
    test_symbol_extraction_without_comma()
    test_symbol_extraction_without_comma_single_space()
    print("\n✅ All tests passed!")
