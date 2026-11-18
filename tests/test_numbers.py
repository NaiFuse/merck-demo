import pytest

from src.extract.numbers import find_numbers, quote_contains_value, unit_multiplier


@pytest.mark.parametrize("quote,value,unit", [
    ("Revenue was $10,898 million", 10898, "USD_m"),
    ("Revenue of $10.9 billion", 10898, "USD_m"),                 # rounded display
    ("Umsatzerlöse 1.234.567,89 EUR", 1234567.89, "EUR"),
    ("Umsatzerlöse T€ 123.456", 123456, "EUR_k"),                  # DE thousands separator
    ("Umsatzerlöse von 2,3 Mrd. €", 2300, "EUR_m"),
    ("Europe | 3,456.7", 3456.7, "USD_m"),
    ("Net loss (123)", -123, "USD_m"),
    ("organic growth of 3%", 3, "%"),
    ("Academic and government accounted for 14 percent", 14, "%"),
    ("sales of 1,234", 1.234, "EUR_m"),                           # ambiguous: DE decimal reading
])
def test_quote_contains_value(quote, value, unit):
    assert quote_contains_value(quote, value, unit)


@pytest.mark.parametrize("quote,value,unit", [
    ("Revenue was $10,898 million", 10889, "USD_m"),
    ("Revenue of $10.9 billion", 10700, "USD_m"),
    ("organic growth of 3%", 30, "%"),
    ("growth of 3 billion", 3, "%"),                               # scaled number is not a percentage
    ("no numbers here", 1, "USD_m"),
])
def test_quote_does_not_contain_value(quote, value, unit):
    assert not quote_contains_value(quote, value, unit)


def test_find_numbers_handles_mixed_separators():
    vals = {r.value for r in find_numbers("1,234.5 and 1.234,5 and 7")}
    assert {1234.5, 7.0} <= vals


def test_unit_multiplier():
    assert unit_multiplier("USD_m") == 1e6 and unit_multiplier("EUR_k") == 1e3 and unit_multiplier("%") == 1
    with pytest.raises(ValueError):
        unit_multiplier("USD_x")
