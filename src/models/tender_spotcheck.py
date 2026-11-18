"""Spot-check UK (FTS) awards in the DB against the live FTS notice (official API, one call per notice).

    uv run python -m src.models.tender_spotcheck [--n 3]

Picks deterministic samples from the default view: (1) the largest UK award by value, (2) the largest UK
non-framework award, (3) the largest UK award won by a peer group (Merck / Thermo Fisher / Avantor), falling back
to the next-largest notice. For each it fetches GET /ocdsReleasePackages/{noticeId} and compares winner, value,
currency, buyer, title, publication date and framework flag with what is stored. Prints the FTS notice page
(source_url) for a manual look. Read-only DB access, closed before any network call.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from src.connectors import uk_fts
from src.db import connect_readonly
from src.models.tender_report import in_scope, load_awards
from src.models.winners import PEER_GROUPS, OTHER, IVD

PEERS = [g for g in PEER_GROUPS if g not in (OTHER, IVD)]


def load_uk(con) -> pd.DataFrame:
    df = load_awards(con)
    extra = con.execute("""
        SELECT a.notice_id, a.seq, a.value, a.currency AS award_currency, t.source_url, t.is_framework AS fw_raw
        FROM tender_awards a JOIN tenders t USING (notice_id) WHERE t.source = 'uk_fts'
    """).df()
    d = in_scope(df)
    return d[d["source"] == "uk_fts"].merge(extra, on=["notice_id", "seq"])


def pick(d: pd.DataFrame, n: int = 3) -> list[tuple[str, pd.Series]]:
    """[(reason, award row)] with distinct notices."""
    d = d[d["value_eur"].notna()].sort_values("value_eur", ascending=False)
    rules = [("largest UK award", d),
             ("largest UK non-framework award", d[~d["is_framework"]]),
             ("largest UK award won by a peer", d[d["group"].isin(PEERS)])]
    out, used = [], set()
    for reason, cand in rules:
        cand = cand[~cand["notice_id"].isin(used)]
        if cand.empty:
            cand, reason = d[~d["notice_id"].isin(used)], f"{reason} – none, next-largest UK award instead"
        if not cand.empty:
            r = cand.iloc[0]
            out.append((reason, r))
            used.add(r["notice_id"])
    while len(out) < n:
        rest = d[~d["notice_id"].isin(used)]
        if rest.empty:
            break
        out.append(("next-largest UK award", rest.iloc[0]))
        used.add(rest.iloc[0]["notice_id"])
    return out[:n]


@dataclass
class Check:
    field: str
    stored: object
    live: object
    ok: bool


def compare(row: pd.Series, pkg: dict) -> list[Check]:
    rid = row["notice_id"].removeprefix(uk_fts.NOTICE_PREFIX)
    rels = [r for r in pkg.get("releases") or [] if r.get("id") == rid] or (pkg.get("releases") or [])[-1:]
    if not rels:
        return [Check("release found", rid, None, False)]
    r = rels[-1]
    award_id = str(row["evidence"]).split(":", 1)[0]
    award = next((a for a in r.get("awards") or [] if a.get("id") == award_id), None)
    checks = [Check("award id in live notice", award_id, award_id if award else None, award is not None)]
    if award is None:
        return checks
    sups = [s.get("name") for s in award.get("suppliers") or []]
    checks.append(Check("winner", row["winner_name"], sups, row["winner_name"] in sups))
    amount = uk_fts.award_values(r).get(award_id)
    if amount is None:
        checks.append(Check("value", row["value"], None, pd.isna(row["value"])))
    else:
        expect = amount[0] / max(len(sups), 1) if row["alignment"] == "split_equal" else amount[0]
        checks.append(Check("value (orig. currency)", row["value"], expect, abs((row["value"] or 0) - expect) < 0.01))
        checks.append(Check("currency", row["award_currency"], amount[1], row["award_currency"] == amount[1]))
    buyer = (r.get("buyer") or {}).get("name") or next(
        (p.get("name") for p in r.get("parties") or [] if "buyer" in (p.get("roles") or [])), None)
    checks.append(Check("buyer", row["buyer"], buyer, row["buyer"] == buyer))
    title = (r.get("tender") or {}).get("title")
    checks.append(Check("title", row["title"], title, row["title"] == title))
    pub = str(r.get("date") or "")[:10]
    checks.append(Check("publication date", str(row["publication_date"]), pub, str(row["publication_date"]) == pub))
    fw = bool(((r.get("tender") or {}).get("techniques") or {}).get("hasFrameworkAgreement"))
    checks.append(Check("framework flag", bool(row["is_framework"]), fw, bool(row["is_framework"]) == fw))
    return checks


def run(n: int = 3, client: uk_fts.FtsClient | None = None) -> list[tuple[str, pd.Series, list[Check]]]:
    with connect_readonly() as con:      # read, then close before the network calls
        picks = pick(load_uk(con), n)
    own = client is None
    client = client or uk_fts.FtsClient()
    try:
        out = []
        for reason, row in picks:
            rid = row["notice_id"].removeprefix(uk_fts.NOTICE_PREFIX)
            _, pkg = client.get(f"{uk_fts.API_URL}/{rid}")
            out.append((reason, row, compare(row, pkg)))
        return out
    finally:
        if own:
            client.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args(argv)
    results = run(args.n)
    for i, (reason, row, checks) in enumerate(results, 1):
        status = "PASS" if all(c.ok for c in checks) else "FAIL"
        print(f"\n[{i}] {status} – {reason}: {row['notice_id']}  EUR {row['value_eur']/1e6:,.2f}m  "
              f"group={row['group']}  framework={bool(row['is_framework'])}  grade={row['grade']}/{row['alignment']}")
        print(f"    notice page: {row['source_url']}")
        for c in checks:
            s, l = str(c.stored)[:70], str(c.live)[:70]
            print(f"    {'ok ' if c.ok else 'XX '} {c.field:<24} stored={s!r}" + ("" if c.ok else f"  live={l!r}"))
    print(f"\n{sum(all(c.ok for c in ch) for _, _, ch in results)} of {len(results)} spot-checks passed")


if __name__ == "__main__":
    main()
