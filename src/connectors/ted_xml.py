"""Lot-level winners and values from a TED eForms notice XML (https://ted.europa.eu/en/notice/<id>/xml).

Used only for a small set of high-value multi-winner notices whose search-API values cannot be
attributed reliably. Chain: LotResult -> LotTender -> TenderingParty -> Tenderer -> Organization.
Value per winning tender: LotTender PayableAmount (BT-720) if present and > 0, else the lot's framework
maximum (BT-709) when the lot has exactly one winner, else the lot maximum split equally (grade C).
The caller must reconcile the XML total against the notice value before using the rows (see ted.py).
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

XML_URL = "https://ted.europa.eu/en/notice/{id}/xml"


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
        el.attrib = {k.split("}", 1)[-1]: v for k, v in el.attrib.items()}
    return root


def _txt(el: ET.Element | None, path: str) -> str | None:
    if el is None:
        return None
    f = el.find(path)
    return f.text.strip() if f is not None and f.text else None


def _positive(s: str | None) -> bool:
    """Amounts of 0 are placeholders some buyers enter for frameworks; treat them as missing."""
    try:
        return s is not None and float(s) > 0
    except ValueError:
        return False


@dataclass
class XmlAward:
    lot_id: str | None
    tender_id: str
    winner_name: str | None
    winner_id: str | None
    value: float | None
    currency: str | None
    value_kind: str          # 'payable' | 'framework_max' | 'framework_max_split' | 'none'
    grade: str
    evidence: str


@dataclass
class XmlResult:
    awards: list[XmlAward]
    lot_texts: dict[str, str] = field(default_factory=dict)  # lot id -> title + description

    @property
    def fully_resolved(self) -> bool:
        return bool(self.awards) and all(a.grade == "A" and a.winner_name for a in self.awards)

    def total(self) -> float:
        return sum(a.value or 0 for a in self.awards)


def parse_notice_xml(xml_text: str) -> XmlResult:
    root = _strip_ns(ET.fromstring(xml_text.encode("utf-8")))

    orgs: dict[str, tuple[str | None, str | None]] = {}
    for org in root.iter("Organization"):
        comp = org.find("Company")
        oid = _txt(comp, "PartyIdentification/ID")
        if oid:
            orgs[oid] = (_txt(comp, "PartyName/Name"), _txt(comp, "PartyLegalEntity/CompanyID"))

    parties: dict[str, str] = {}  # TPA -> lead ORG
    for tp in root.iter("TenderingParty"):
        tid = _txt(tp, "ID")
        tenderers = tp.findall("Tenderer")
        if not tid or not tenderers:
            continue
        lead = next((t for t in tenderers if _txt(t, "GroupLeadIndicator") == "true"), tenderers[0])
        parties[tid] = _txt(lead, "ID")

    tenders: dict[str, dict] = {}
    for lt in root.iter("LotTender"):
        tid_el = lt.find("ID")
        # Definitions carry a TenderingParty; references (inside LotResult / SettledContract) don't,
        # and some notices give references schemeName="tender" too.
        if tid_el is None or not tid_el.text or lt.find("TenderingParty") is None:
            continue
        amt = lt.find("LegalMonetaryTotal/PayableAmount")
        tenders[tid_el.text.strip()] = {
            "tpa": _txt(lt, "TenderingParty/ID"),
            "lot": _txt(lt, "TenderLot/ID"),
            "amount": amt.text.strip() if amt is not None and amt.text else None,
            "currency": amt.get("currencyID") if amt is not None else None,
        }

    lot_texts: dict[str, str] = {}
    for pl in root.iter("ProcurementProjectLot"):
        lid = _txt(pl, "ID")
        if lid:
            parts = [_txt(pl, "ProcurementProject/Name"), _txt(pl, "ProcurementProject/Description")]
            lot_texts[lid] = " | ".join(p for p in parts if p)

    awards: list[XmlAward] = []
    for lr in root.iter("LotResult"):
        if _txt(lr, "TenderResultCode") != "selec-w":
            continue
        lot = _txt(lr, "TenderLot/ID")
        refs = [r.text.strip() for r in lr.findall("LotTender/ID") if r.text]
        mx = lr.find("FrameworkAgreementValues/MaximumValueAmount")
        mx_val = mx.text.strip() if mx is not None and mx.text else None
        mx_cur = mx.get("currencyID") if mx is not None else None
        for ref in refs:
            t = tenders.get(ref, {})
            org_id = parties.get(t.get("tpa") or "")
            name, company_id = orgs.get(org_id or "", (None, None))
            if _positive(t.get("amount")):
                val, cur, kind, grade = float(t["amount"]), t["currency"], "payable", "A"
                ev = f"{lot}/{ref} PayableAmount={t['amount']} {cur}"
            elif _positive(mx_val) and len(refs) == 1:
                val, cur, kind, grade = float(mx_val), mx_cur, "framework_max", "A"
                ev = f"{lot}/{ref} MaximumValueAmount={mx_val} {cur}"
            elif _positive(mx_val):
                val, cur, kind, grade = float(mx_val) / len(refs), mx_cur, "framework_max_split", "C"
                ev = f"{lot}/{ref} MaximumValueAmount={mx_val} {cur} split over {len(refs)} winners"
            else:
                val, cur, kind, grade = None, None, "none", "C"
                ev = f"{lot}/{ref} no amount"
            awards.append(XmlAward(lot or t.get("lot"), ref, name, company_id, val, cur, kind, grade,
                                   f"{ev}; winner {org_id}={name}"))
    return XmlResult(awards, lot_texts)


def fetch_xml(notice_ids: list[str], client: httpx.Client | None = None, min_interval: float = 1.0) -> dict[str, tuple[str, str]]:
    """Network only. Returns {notice_id: (url, xml_text)}; failures are skipped and reported by the caller."""
    own = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True,
                                    headers={"User-Agent": "merck-ds-demo/0.1 (research)"})
    out: dict[str, tuple[str, str]] = {}
    try:
        for nid in notice_ids:
            url = XML_URL.format(id=nid)
            for attempt in range(3):
                try:
                    r = client.get(url)
                    if r.status_code == 200:
                        out[nid] = (url, r.text)
                        break
                    if r.status_code < 500 and r.status_code != 429:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(2 * (attempt + 1))
            time.sleep(min_interval)
    finally:
        if own:
            client.close()
    return out
