"""Merck DS EMEA Academia revenue (public-data estimate, grade C) and its share of the Academia market.

    uv run python -m src.models.merck_ds show

    DS EMEA revenue    = DS revenue FY2025 x DS Europe share            (config/assumptions/merck_ds.yaml)
                         x (slider target / default base) when the dashboard slider moves it
    DS Academia[k]     = DS EMEA revenue x DS Academia share x cluster share k (research base, grade B)
    share vs method   = DS Academia[k] / market draws of A, B and the Merck anchor (FY2025)
    share (reconciled) = pooled DS/A + DS/B draws if published, else DS P10 / market high .. DS P90 / market low

Monte Carlo: independent triangular draws (own seed); all arithmetic numpy. Output grade = C (estimate inputs).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel

from src.models.peer_restatement import ASSUMPTIONS_DIR, PCTS, Assumption, Market, _grade, triangular

TOTAL = "total"


class MerckConfig(BaseModel):
    entity: str
    label: str
    period: str
    disclosed: dict[str, Assumption]
    estimate: dict[str, Assumption]
    bridge: dict
    anchor: dict = {}

    def assumptions(self) -> list[Assumption]:
        """Every Assumption in the config (disclosed, estimates, bridge inputs)."""
        b = self.bridge
        out = list(self.disclosed.values()) + list(self.estimate.values())
        out += [Assumption.model_validate(b[k]) for k in ("ds_growth", "ds_pharma_of_rest", "market_academia_weight",
                                                         "market_pharma_of_rest")]
        out += [Assumption.model_validate(v) for v in b["market_growth"].values() if isinstance(v, dict)]
        out += [Assumption.model_validate(v) for v in self.anchor.values() if isinstance(v, dict) and "key" in v]
        return out

    def default_emea_base(self) -> float:
        e = self.estimate
        return e["ds_revenue_fy2025"].base * e["ds_europe_share"].base


def load_config(directory: Path = ASSUMPTIONS_DIR) -> MerckConfig:
    with open(directory / "merck_ds.yaml", encoding="utf-8") as f:
        return MerckConfig.model_validate(yaml.safe_load(f))


def pct(x: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.percentile(x, PCTS))  # type: ignore[return-value]


@dataclass
class ShareCell:
    cluster: str
    vs: dict[str, tuple[float, float, float]]  # method -> P10/P50/P90 of DS / that method's market draws
    low: float                                 # vs the reconciled market
    base: float | None                         # None when the market cell is a range
    high: float
    grade: str = "C"

    @property
    def vs_a(self) -> tuple[float, float, float]:
        return self.vs["A"]

    @property
    def vs_b(self) -> tuple[float, float, float]:
        return self.vs["B"]


@dataclass
class DSEstimate:
    emea_target: float                                     # EUR m, base of DS EMEA revenue (slider or default)
    emea_default: float
    emea: np.ndarray = field(repr=False)                   # DS EMEA revenue draws, all customers
    academia_share: np.ndarray = field(repr=False)
    academia: dict[str, np.ndarray] = field(repr=False, default_factory=dict)   # cluster|total -> EUR m draws
    shares: dict[str, ShareCell] = field(default_factory=dict)
    grade: str = "C"

    def academia_pct(self, k: str = TOTAL) -> tuple[float, float, float]:
        return pct(self.academia[k])


def estimate(cfg: MerckConfig, market: Market, methods: dict[str, dict[str, np.ndarray]],
             reconciled: dict[str, tuple[float, float | None, float, str]], n: int, seed: int,
             emea_target: float | None = None) -> DSEstimate:
    """methods: name ('A', 'B', 'merck_anchor', ...) -> cluster|total -> market draws (EUR m), share year.
    reconciled: cluster|total -> (low, base, high, status) of the reconciled market cell.
    Share vs the reconciled market: pooled DS/A and DS/B draws if published; else DS P10 / high .. DS P90 / low."""
    rng = np.random.default_rng(seed)
    e = cfg.estimate
    default = cfg.default_emea_base()
    target = default if emea_target is None else float(emea_target)
    emea = triangular(e["ds_revenue_fy2025"], n, rng) * triangular(e["ds_europe_share"], n, rng) * (target / default)
    acad = triangular(e["ds_academia_share"], n, rng)
    academia = {k: emea * acad * triangular(a, n, rng) for k, a in market.clusters.items()}
    academia[TOTAL] = sum(academia.values())
    grade = _grade("C", *(a.grade for a in market.clusters.values()))
    shares = {}
    for k, ds in academia.items():
        vs = {name: pct(ds / cells[k]) for name, cells in methods.items() if k in cells}
        lo_m, _, hi_m, status = reconciled[k]
        if status == "published":
            lo, mid, hi = pct(np.concatenate([ds / methods["A"][k], ds / methods["B"][k]]))
        else:
            d10, _, d90 = pct(ds)
            lo, mid, hi = d10 / hi_m, None, d90 / lo_m
        shares[k] = ShareCell(k, vs, lo, mid, hi, grade)
    return DSEstimate(target, default, emea, acad, academia, shares, grade)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["show"])
    ap.parse_args(argv)
    from src.models import scenario

    s = scenario.run(scenario.prepare(), {})
    d = s.ds
    print(f"DS EMEA revenue (all customers) P10/P50/P90: {' / '.join(f'{v:,.0f}' for v in pct(d.emea))} EUR m "
          f"(default base {d.emea_default:,.0f})")
    for k, sc in d.shares.items():
        a = d.academia_pct(k)
        print(f"  {k:<20} DS Academia {a[0]:5,.0f} – {a[2]:5,.0f} EUR m   "
              + "  ".join(f"vs {m} {v[0]:.1%}–{v[2]:.1%}" for m, v in sc.vs.items())
              + f"  reconciled {sc.low:.1%}–{sc.high:.1%}")


if __name__ == "__main__":
    main()
