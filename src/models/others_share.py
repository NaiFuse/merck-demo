"""Others' share for method A from tender win shares (TED + UK FTS) instead of a free assumption.

    uv run python -m src.models.others_share

others = 1 - (Thermo Fisher + Avantor) share of Academia tender awards, default tender view (core CPV, Academia
buyers, excl. IVD, clinical consumables, analyser bundles, Merck lab water), periods 2024 .. latest pooled (the
5-week 2023 stub is left out). Eight views span the range:
    measure        value | award count
    frameworks     included (ceilings) | excluded
    distributors   counted as others | allocated pro rata (removed from the base)
low = min, high = max, base = median of the eight views; grade B (two registers, eight views).
Caveat: above-threshold tenders only. Small catalogue orders (web shop, below threshold) are not in the data.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import pandas as pd

from src.models.peer_restatement import Assumption

PEER_GROUPS = ("thermo_fisher", "avantor")


@dataclass
class View:
    measure: str          # 'value' | 'count'
    frameworks: bool
    distributors: str     # 'others' | 'pro rata'
    n_awards: int
    total: float          # EUR m or award count
    peers: float          # share of the base
    distributor_share: float
    others: float


def views(awards: pd.DataFrame) -> list[View]:
    """awards = tender_report.load_awards() output (all rows); the default in-scope filter is applied here."""
    from src.models.tender_report import in_scope

    d = in_scope(awards)
    d = d[~d["period"].astype(str).str.startswith("2023")]
    out = []
    for fw in (True, False):
        y = d if fw else d[~d["is_framework"]]
        for measure in ("value", "count"):
            w = y.groupby("group")["value_eur"].sum() if measure == "value" else y.groupby("group").size()
            tot = float(w.sum())
            if tot <= 0:
                continue
            peers = float(sum(w.get(g, 0.0) for g in PEER_GROUPS))
            dist = float(w.get("distributor", 0.0))
            for mode, base in (("others", tot), ("pro rata", tot - dist)):
                out.append(View(measure, fw, mode, len(y), tot / 1e6 if measure == "value" else tot,
                                peers / base, dist / tot, 1 - peers / base))
    return out


def derive(awards: pd.DataFrame, prior: Assumption | None = None) -> tuple[Assumption, list[View]]:
    vs = views(awards)
    if not vs:
        raise ValueError("no in-scope tender awards – load TED/FTS first")
    vals = [v.others for v in vs]
    periods = sorted(awards["period"].astype(str).unique())
    lo, base, hi = min(vals), statistics.median(vals), max(vals)
    detail = "; ".join(f"{v.measure}/{'with' if v.frameworks else 'no'} frameworks/distributors {v.distributors}: "
                       f"{v.others:.1%}" for v in vs)
    a = Assumption(
        key="others_share", low=round(lo, 4), base=round(base, 4), high=round(hi, 4), grade="B",
        source=f"derived:tenders – 1 - (Thermo Fisher + Avantor) share of Academia tender awards (TED + UK FTS, "
               f"default view, {periods[1] if len(periods) > 1 else periods[0]} .. {periods[-1]} pooled); "
               f"min / median / max of {len(vs)} views: {detail}",
        note="Above-threshold public tenders only (below-threshold catalogue orders not covered)."
             + (f" Replaces the prior estimate {prior.low:.0%}-{prior.high:.0%} (base {prior.base:.0%})." if prior else ""))
    return a, vs


def load_awards() -> pd.DataFrame:
    from src.db import connect_readonly
    from src.models.tender_report import load_awards as _load

    with connect_readonly() as con:     # read-only, closed immediately (rule 7)
        return _load(con)


def apply_to_market(market, awards: pd.DataFrame | None = None):
    """Market with supply.others_share derived from tenders when market.yaml sets others_share_from: tenders.
    The prior estimate is kept as supply.others_share_prior."""
    s = dict(market.supply)
    if s.get("others_share_from") != "tenders":
        return market
    prior = Assumption.model_validate(s["others_share"])
    derived, _ = derive(load_awards() if awards is None else awards, prior)
    s["others_share_prior"] = s["others_share"]
    s["others_share"] = derived.model_dump(exclude_none=True)
    return market.model_copy(update={"supply": s})


if __name__ == "__main__":
    a, vs = derive(load_awards())
    for v in vs:
        print(f"  {v.measure:<6} frameworks {'in ' if v.frameworks else 'out'}  distributors {v.distributors:<9} "
              f"awards {v.n_awards:5d}  base {v.total:9,.1f}  peers {v.peers:5.1%}  distributors {v.distributor_share:5.1%}"
              f"  -> others {v.others:5.1%}")
    print(f"others_share = {a.low:.1%} / {a.base:.1%} / {a.high:.1%} [{a.grade}]")
