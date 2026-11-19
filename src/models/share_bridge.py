"""Share bridge: why DS grows faster or slower than its market (CLAUDE.md "Share bridge").

    g_DS - g_M = sum_s (w_s - v_s) * g_s   [mix]   +   sum_s w_s * (g_DS_s - g_s)   [share]

w_s = DS revenue weight of customer segment s, v_s = market weight, g_s = market growth of s, g_DS_s = DS growth in s.
With public data DS growth by segment is unknown, so the share term is the residual (g_DS - g_M) - mix; the identity
holds either way. Inputs may be scalars or numpy arrays of Monte Carlo draws (element-wise arithmetic only).

Two market-growth views (same DS growth and weights):
  model      segment growth from the market model (Academia = method B nowcast) and estimates
  merck_lsd  Merck's own view of the Discovery market, '~LSD' (CMD 2025 p.14, 1-3%, grade C): each segment's growth is
             scaled per draw so that sum_s v_s g_s equals the LSD draw (the segment pattern is kept, the level is Merck's)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.models.peer_restatement import Assumption, triangular


def decompose(w: dict, v: dict, g: dict, g_ds=None, g_ds_s: dict | None = None) -> dict:
    """Returns g_m, g_ds, mix, share, mix_s (per segment) and share_s (per segment, only when g_ds_s is given).
    Give either g_ds (total DS growth -> share is the residual) or g_ds_s (DS growth per segment)."""
    if (g_ds is None) == (g_ds_s is None):
        raise ValueError("give exactly one of g_ds / g_ds_s")
    segs = list(g)
    g_m = sum(v[s] * g[s] for s in segs)
    mix_s = {s: (w[s] - v[s]) * g[s] for s in segs}
    mix = sum(mix_s.values())
    out = {"g_m": g_m, "mix": mix, "mix_s": mix_s}
    if g_ds_s is not None:
        share_s = {s: w[s] * (g_ds_s[s] - g[s]) for s in segs}
        out.update(share_s=share_s, share=sum(share_s.values()), g_ds=sum(w[s] * g_ds_s[s] for s in segs))
    else:
        out.update(share=g_ds - g_m - mix, g_ds=g_ds)
    return out


@dataclass
class Bridge:
    segments: dict[str, str]                     # key -> label
    draws: dict = field(repr=False)              # decompose() output on draws
    w: dict = field(repr=False)
    v: dict = field(repr=False)
    g: dict = field(repr=False)
    grade: str = "C"
    views: dict = field(default_factory=dict, repr=False)   # view key -> decompose() output ('model', 'merck_lsd')
    view_labels: dict = field(default_factory=dict)

    def pct(self, x) -> tuple[float, float, float]:
        return tuple(float(q) for q in np.percentile(x, (10, 50, 90)))  # type: ignore[return-value]


VIEW_LABELS = {"model": "Our model: 2025 nowcast", "merck_lsd": "Merck's view: Discovery market ~LSD"}


def build(bridge_cfg: dict, ds_academia_share: np.ndarray, academia_growth: np.ndarray, n: int, seed: int,
          lsd: Assumption | None = None) -> Bridge:
    """ds_academia_share: DS Academia weight draws (same draws as the DS estimate);
    academia_growth: market growth draws of the Academia segment (method B, latest year)."""
    rng = np.random.default_rng(seed)

    def t(key):
        return triangular(Assumption.model_validate(bridge_cfg[key]), n, rng)

    segs = {k: v["label"] for k, v in bridge_cfg["segments"].items()}
    a, p, i = "academia", "pharma_biotech", "industrial_testing"
    if set(segs) != {a, p, i}:
        raise ValueError(f"bridge segments must be {a}, {p}, {i}")
    ds_p = t("ds_pharma_of_rest")
    w = {a: ds_academia_share, p: (1 - ds_academia_share) * ds_p, i: (1 - ds_academia_share) * (1 - ds_p)}
    va, mp = t("market_academia_weight"), t("market_pharma_of_rest")
    v = {a: va, p: (1 - va) * mp, i: (1 - va) * (1 - mp)}
    mg = bridge_cfg["market_growth"]
    g = {a: academia_growth,
         p: triangular(Assumption.model_validate(mg[p]), n, rng),
         i: triangular(Assumption.model_validate(mg[i]), n, rng)}
    g_ds = t("ds_growth")
    model = decompose(w, v, g, g_ds=g_ds)
    views = {"model": model}
    if lsd is not None:
        g_lsd = triangular(lsd, n, rng)
        scale = g_lsd / model["g_m"]
        views["merck_lsd"] = decompose(w, v, {s: g[s] * scale for s in g}, g_ds=g_ds)
    return Bridge(segs, model, w, v, g, views=views, view_labels={k: VIEW_LABELS[k] for k in views})
