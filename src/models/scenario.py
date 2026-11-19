"""Scenario runner for the dashboard: verify sources once, then recalculate everything live for slider overrides.

    base = prepare()                 # slow (~15 s): loads caches, verifies every assumption quote on its PDF page
    s = run(base, {"market.others_share": 0.6, "merck_ds.ds_emea_revenue": 900})   # fast: Monte Carlo only

A slider moves the base (mode) of a triangular input. 'shift' knobs move low/base/high by the same amount (shares),
'scale' knobs multiply them (intensities, EUR amounts), so the input keeps its uncertainty width. Overridden inputs
stay grade C. No network, no DB writes, no LLM: pure recomputation from cached facts and config/assumptions/*.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from src.facts_store import FactRow
from src.models import market_model, merck_ds, reconciliation, share_bridge
from src.models.peer_restatement import (
    PEERS, Assumption, Market, PeerConfig, RunResult, check_source_quotes, dach_crosscheck, fx_units_per_eur,
    load_market, load_peer, restate,
)
from src.models.research_base import Panel

EMEA_REVENUE = "merck_ds.ds_emea_revenue"


@dataclass(frozen=True)
class Knob:
    uid: str            # '<owner>.<assumption key>'; owner = peer | market | merck_ds
    label: str
    group: str
    mode: str           # 'shift' | 'scale' | 'value' (EUR m target, DS EMEA revenue only)
    lo: float
    hi: float
    step: float
    pct: bool = True    # display as %
    help: str = ""


KNOBS: list[Knob] = [
    Knob(EMEA_REVENUE, "DS EMEA revenue FY2025 (EUR m, all customers)", "Merck DS", "value", 500, 1500, 10, False,
         "Public-data estimate: DS ~EUR 2.9bn x Europe share. Move it to your internal number."),
    Knob("merck_ds.ds_academia_share", "DS Academia share of EMEA revenue", "Merck DS", "shift", 0.15, 0.65, 0.01),
    Knob("merck_ds.ds_emea_organic_growth", "DS EMEA organic growth (bridge)", "Merck DS", "shift", -0.03, 0.08, 0.001),
    Knob("market.ds_consumables_intensity", "Consumables intensity per EUR of R&D (method B)", "Market", "scale",
         0.005, 0.040, 0.001),
    Knob("market.non_rd_uplift", "Non-R&D uplift: teaching & hospital labs (method B)", "Market", "shift", 0.0, 0.30, 0.01),
    Knob("market.others_share", "Others' share of market, excl. TMO + Avantor (method A; default from tenders)",
         "Market", "shift", 0.40, 0.95, 0.01),
    Knob("thermo_fisher.ds_share", "Thermo Fisher: DS-equivalent product share", "Peers", "shift", 0.25, 0.85, 0.01),
    Knob("avantor.ds_share", "Avantor: DS-equivalent product share", "Peers", "shift", 0.25, 0.85, 0.01),
    Knob("merck_ds.market_w_academia", "Academia weight of the DS-type market (bridge)", "Market", "shift",
         0.10, 0.50, 0.01),
]
KNOB_BY_UID = {k.uid: k for k in KNOBS}


# ---------------------------------------------------------------- overrides

def moved(a: Assumption, target: float, mode: str) -> Assumption:
    """Copy of a with base = target, low/high moved alongside (shift or scale), clamped for shares; grade C."""
    if mode == "scale":
        f = target / a.base if a.base else 1.0
        lo, hi = a.low * f, a.high * f
    else:
        d = target - a.base
        lo, hi = a.low + d, a.high + d
        if a.unit == "share" and 0 <= a.low and a.high <= 1:
            lo, hi = max(lo, 0.0), min(hi, 0.99)
    lo, hi = min(lo, target), max(hi, target)
    if lo == hi:                     # grade C must stay a range (rule 3)
        lo, hi = target * 0.95, target * 1.05
    return Assumption(key=a.key, low=lo, base=target, high=hi, grade="C",
                      source=f"slider override (default {a.base:g}; {a.low:g}-{a.high:g}) – {a.source}", note=a.note,
                      unit=a.unit)


def replace_assumption(obj, key: str, fn):
    """Return obj with every Assumption (or assumption dict) whose key == key replaced by fn(assumption)."""
    if isinstance(obj, Assumption):
        return fn(obj) if obj.key == key else obj
    if isinstance(obj, BaseModel):
        return obj.model_copy(update={f: replace_assumption(getattr(obj, f), key, fn) for f in type(obj).model_fields})
    if isinstance(obj, dict):
        if obj.get("key") == key and "base" in obj:
            return fn(Assumption.model_validate(obj)).model_dump(exclude_none=True)
        return {k: replace_assumption(v, key, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_assumption(v, key, fn) for v in obj]
    return obj


def find_assumption(obj, key: str) -> Assumption | None:
    found: list[Assumption] = []
    replace_assumption(obj, key, lambda a: found.append(a) or a)
    return found[0] if found else None


# ---------------------------------------------------------------- base (slow, once) and scenario (fast)

@dataclass
class Base:
    facts: list[FactRow] = field(repr=False)
    panel: Panel = field(repr=False)
    market: Market                          # with derived cluster shares
    configs: list[PeerConfig]
    merck: merck_ds.MerckConfig
    fx: dict[str, tuple[float, str]]        # peer -> (USD per EUR, source URL)

    def owner(self, owner: str):
        if owner == "market":
            return self.market
        if owner == "merck_ds":
            return self.merck
        return next(c for c in self.configs if c.peer == owner)

    def default(self, uid: str) -> float:
        if uid == EMEA_REVENUE:
            return self.merck.default_emea_base()
        owner, key = uid.split(".", 1)
        a = find_assumption(self.owner(owner), key)
        if a is None:
            raise KeyError(f"no assumption {uid}")
        return a.base


def prepare(facts: list[FactRow] | None = None, online_fx: bool = False) -> Base:
    from src.facts_store import base_facts
    from src.models import others_share, research_base

    facts = base_facts() if facts is None else facts
    market = others_share.apply_to_market(research_base.apply_to_market(load_market(), facts, online_fx))
    configs = [load_peer(p) for p in PEERS]
    merck = merck_ds.load_config()
    problems = check_source_quotes([a for c in configs for a in c.assumptions()] + merck.assumptions())
    if problems:
        raise ValueError("assumption quotes failed verification: " + "; ".join(problems))
    fx = {c.peer: fx_units_per_eur(c.currency, int(c.period[2:6]), online=online_fx) for c in configs}
    return Base(facts, research_base.build_panel(facts, online_fx), market, configs, merck, fx)


@dataclass
class Scenario:
    peers: RunResult
    model: market_model.ModelResult
    ds: merck_ds.DSEstimate
    bridge: share_bridge.Bridge
    overrides: dict[str, float]
    year: int
    checks: dict = field(default_factory=dict)          # cluster -> [reconciliation.Check]
    check_info: dict = field(default_factory=dict)      # anchor / floor details
    share_checks: list = field(default_factory=list)    # implied DS share per method + plausibility flags
    gap: object = None                                  # reconciliation.GapResult (base-case decomposition)

    def cell(self, cluster: str, method: str, year: int | None = None) -> market_model.MarketCell:
        y = self.year if year is None else year
        return next(c for c in self.model.cells if c.cluster == cluster and c.method == method and c.year == y)

    def assumptions(self) -> list[tuple[str, Assumption]]:
        """(owner, assumption) for every input used in this scenario, overrides applied."""
        out = [(c.peer, a) for c in self.peers.configs for a in c.assumptions()]
        m = self.model.market
        out += [("market", a) for a in m.clusters.values()]
        d, s = m.demand, m.supply
        out += [("market", Assumption.model_validate(x)) for x in
                [d["intensity"], d["non_rd_uplift"], d["growth_offset"], *d["sector_weights"].values(), s["others_share"]]]
        out += [("market", a) for a in m.dach_anchor.values()]
        return out


def apply_overrides(base: Base, overrides: dict[str, float]):
    market, configs, merck = base.market, list(base.configs), base.merck
    for uid, target in overrides.items():
        if uid == EMEA_REVENUE:
            continue
        knob = KNOB_BY_UID.get(uid)
        mode = knob.mode if knob else "shift"
        owner, key = uid.split(".", 1)
        fn = lambda a, t=target, m=mode: moved(a, t, m)  # noqa: E731
        if owner == "market":
            market = replace_assumption(market, key, fn)
        elif owner == "merck_ds":
            merck = replace_assumption(merck, key, fn)
        else:
            configs = [replace_assumption(c, key, fn) if c.peer == owner else c for c in configs]
    return market, configs, merck


def run(base: Base, overrides: dict[str, float]) -> Scenario:
    overrides = {k: v for k, v in overrides.items() if v is not None and abs(v - base.default(k)) > 1e-12}
    market, configs, merck = apply_overrides(base, overrides)
    mc = market.monte_carlo
    n, seed = int(mc["draws"]), int(mc["seed"])
    results = [restate(cfg, base.facts, market, *base.fx[cfg.peer], n_draws=n, seed=seed + i)
               for i, cfg in enumerate(configs)]
    try:
        cross = dach_crosscheck(results, base.facts, market, seed=seed)
    except KeyError:
        cross = []
    peers = RunResult(results, configs, market, cross)
    checks, info = reconciliation.checks(peers, base.facts, merck, n, seed)
    model = market_model.run(peers, base.panel, checks)
    year = int(market.supply["year"])
    methods = {"A": {a.cluster: a.draws for a in model.supply},
               "B": {b.cluster: b.draws for b in model.demand if b.year == year}}
    anchor = {k: c.draws for k, cs in checks.items() for c in cs if c.method == "merck_anchor"}
    if anchor:
        methods["merck_anchor"] = anchor
    rec = {c.cluster: (c.low, c.base, c.high, c.status) for c in model.cells if c.method == "reconciled"}
    ds = merck_ds.estimate(merck, market, methods, rec, n, seed + 300, overrides.get(EMEA_REVENUE))
    b = {(c.cluster, c.year): c.draws for c in model.demand}
    acad_growth = b[(market_model.TOTAL, year)] / b[(market_model.TOTAL, year - 1)] - 1
    lsd = merck.anchor.get("market_cagr")
    bridge = share_bridge.build(merck.bridge, ds.academia_share, acad_growth, n, seed + 400,
                                Assumption.model_validate(lsd) if lsd else None)
    band = tuple(market.reconciliation.get("ds_share_plausible", (0.05, 0.30)))
    share_checks = reconciliation.implied_shares(ds.academia, methods, peers, band)
    gap = reconciliation.gap_decomposition(peers, base.panel, base.facts)
    return Scenario(peers, model, ds, bridge, overrides, year, checks, info, share_checks, gap)


def merck_assumptions(s: Scenario, base: Base) -> list[Assumption]:
    _, _, merck = apply_overrides(base, s.overrides)
    return merck.assumptions()


def as_float(x) -> float:
    return float(np.asarray(x))
