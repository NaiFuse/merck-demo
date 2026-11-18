"""PDF extraction: quote/page/number verification, caching and request shape, with a stub Claude client (offline)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.extract import pdf

PAGE1 = "Thermo Fisher Third Quarter 2026 Results"
PAGE2 = "Revenue grew 5% to $11.12 billion in the quarter."


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Minimal valid PDF with one Helvetica text line per page (pypdf can extract it)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        content_id = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def doc(tmp_path):
    return make_pdf(tmp_path / "us_thermo-fisher_2026q3_earnings-release.pdf", [PAGE1, PAGE2])


def item(**kw):
    base = {"metric": "revenue", "period": "2026Q3", "value": 11.12, "unit": "USD_bn", "region": None,
            "segment": None, "page": 2, "quote": "Revenue grew 5% to $11.12 billion in the quarter."}
    return base | kw


class StubClient:
    def __init__(self, items, stop_reason="end_turn"):
        self.items, self.stop_reason, self.requests = items, stop_reason, []
        self.messages = self

    def stream(self, **req):
        self.requests.append(req)
        msg = SimpleNamespace(stop_reason=self.stop_reason, stop_details=None,
                              content=[SimpleNamespace(type="text", text=json.dumps({"facts": self.items}))],
                              usage=SimpleNamespace(input_tokens=100, output_tokens=20))
        return _Ctx(msg)


class _Ctx:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return SimpleNamespace(get_final_message=lambda: self.msg)

    def __exit__(self, *a):
        return False


def test_parse_filename_and_entity(doc):
    d = pdf.parse_filename(doc)
    assert (d.country, d.company, d.fiscal_year, d.doc_type, d.entity) == \
        ("us", "thermo-fisher", "2026q3", "earnings-release", "thermo_fisher")
    with pytest.raises(ValueError):
        pdf.parse_filename(Path("bad_name.pdf"))


def test_normalise_handles_ligatures_spacing_hyphenation():
    assert pdf.normalise("Scientiﬁc  Re-\nports") == pdf.normalise("Scientific Reports")
    assert pdf.normalise("2,5 – 3,0") == pdf.normalise("2,5 - 3,0")
    assert pdf.normalise("re\x00ecting +0.75%") == pdf.normalise("reecting +0.75%")  # unmapped ligature glyph


@pytest.mark.parametrize("it,ok,reason", [
    (item(), True, "ok"),
    (item(page=1), True, "page corrected 1 -> 2"),
    (item(value=11.2), False, "quote does not contain the value"),
    (item(quote="Revenue grew 6% to $11.12 billion in the quarter."), False, "quote not found"),
    (item(unit="GBP"), False, "unknown unit"),
    (item(metric="organic_growth", value=5, unit="%"), True, "ok"),
])
def test_verify_item(doc, it, ok, reason):
    c = pdf.verify_item(it, pdf.page_texts(doc))
    assert c.ok is ok and c.reason.startswith(reason)


def test_extract_caches_and_verifies(doc, tmp_path, monkeypatch):
    monkeypatch.setattr(pdf, "get_settings", lambda: SimpleNamespace(anthropic_model="test-model", anthropic_api_key=None))
    client = StubClient([item(), item(value=99, quote="Revenue grew 5% to $11.12 billion in the quarter.")])
    cache = tmp_path / "cache"
    rec = pdf.extract(doc, client=client, cache_dir=cache)
    assert rec["model"] == "test-model" and rec["sha256"] == pdf.sha256(doc)
    # request: PDF document block, structured output, model from settings, no arithmetic asked
    req = client.requests[0]
    assert req["model"] == "test-model"
    assert req["messages"][0]["content"][0]["type"] == "document"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert "Never calculate" in req["system"]
    # second call uses the cache, no new request
    pdf.extract(doc, client=client, cache_dir=cache)
    assert len(client.requests) == 1
    accepted, rejected = pdf.verify_all_cached([doc], cache)
    assert len(accepted) == 1 and len(rejected) == 1
    r = accepted[0]
    assert (r.entity, r.period, r.value, r.unit, r.grade, r.source) == ("thermo_fisher", "2026Q3", 11.12, "USD_bn", "A", "manual_pdf")
    assert r.extra["page"] == 2 and r.url_or_path.endswith(doc.name)


def test_refusal_and_truncation_raise(doc, monkeypatch, tmp_path):
    monkeypatch.setattr(pdf, "get_settings", lambda: SimpleNamespace(anthropic_model="m", anthropic_api_key=None))
    for stop in ("refusal", "max_tokens"):
        with pytest.raises(RuntimeError):
            pdf.extract(doc, client=StubClient([], stop_reason=stop), cache_dir=tmp_path / stop)
        assert not (tmp_path / stop).exists() or not list((tmp_path / stop).glob("*.json"))


def test_missing_model_raises(doc, monkeypatch, tmp_path):
    monkeypatch.setattr(pdf, "get_settings", lambda: SimpleNamespace(anthropic_model=None, anthropic_api_key=None))
    with pytest.raises(RuntimeError, match="ANTHROPIC_MODEL"):
        pdf.extract(doc, client=StubClient([]), cache_dir=tmp_path)



def test_dedupe_keeps_precise_figure_and_records_duplicate():
    from datetime import datetime
    from src.facts_store import FactRow

    def r(value, unit, page, segment=None):
        return FactRow("manual_pdf", "x.pdf", datetime(2026, 1, 1), "carl_roth", "FY2024", "revenue", value, unit,
                       None, segment, "q", "A", {"page": page})
    rows = [r(103115.0, "EUR_k", 1), r(103114540.40, "EUR", 4), r(75984418.56, "EUR", 10, "Inland"),
            r(103200.0, "EUR_k", 2)]                         # differs by more than rounding: kept
    kept, dropped = pdf.dedupe_facts(rows)
    assert [(x.value, x.unit) for x in dropped] == [(103115.0, "EUR_k")]
    assert len(kept) == 3
    assert next(x for x in kept if x.unit == "EUR" and x.segment is None).extra["duplicates"] == ["p.1: 103115.0 EUR_k"]
