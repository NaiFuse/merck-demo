from src.models.taxonomy import load_taxonomy
from src.settings import Settings


def test_taxonomy_matches_claude_md():
    tax = load_taxonomy()
    assert set(tax["clusters"]) == {"dach", "france_benelux", "uk_ireland_nordics"}
    assert tax["clusters"]["uk_ireland_nordics"]["label"] == "UK, Ireland & Nordics"
    assert "IE" in tax["clusters"]["uk_ireland_nordics"]["countries"]
    all_countries = [c for cl in tax["clusters"].values() for c in cl["countries"]]
    assert all(isinstance(c, str) and len(c) == 2 for c in all_countries)  # YAML 'NO' -> False trap
    assert "NO" in all_countries
    assert len(tax["categories"]) == 5
    assert [k for k, v in tax["segments"].items() if v["in_scope"]] == ["academia"]
    assert set(tax["peers"]) == {"thermo_fisher", "avantor"}


def test_secrets_are_hidden_in_repr():
    s = Settings(anthropic_api_key="sk-secret-123", anthropic_model="m")
    assert "sk-secret-123" not in repr(s)
    assert "sk-secret-123" not in str(s.configured())
    assert s.configured()["anthropic_api_key"] is True
