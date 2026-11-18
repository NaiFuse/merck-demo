"""Map raw tender winner names to corporate groups (Thermo Fisher, Avantor, Merck, multi-brand distributors,
IVD, other).

Patterns cover the parent name, legal entities and the main lab brands each group trades under.
Names are matched case- and accent-insensitively. Order matters: exclusions run first, then the three
peers, then multi-brand lab distributors, then IVD (so e.g. 'Thermo Fisher Diagnostics' stays Thermo Fisher).

Multi-brand distributors = independent lab distributors carrying many suppliers' brands. This separates them
from manufacturers only; it makes no claim about which brands a distributor carries.

IVD = clinical diagnostics vendors, out of scope for DS. Bio-Rad sells both life-science and clinical
products under the same legal entities, so only Bio-Rad names that say 'clinical'/'diagnostic' go to IVD.
"""

from __future__ import annotations

import re
import unicodedata

THERMO = "thermo_fisher"
AVANTOR = "avantor"
MERCK = "merck"
DISTRIBUTOR = "distributor"
IVD = "ivd"
OTHER = "other"
GROUPS = [THERMO, AVANTOR, MERCK, DISTRIBUTOR, IVD, OTHER]
PEER_GROUPS = [THERMO, AVANTOR, MERCK, DISTRIBUTOR, OTHER]  # default report (IVD shown separately)
GROUP_LABELS = {THERMO: "Thermo Fisher", AVANTOR: "Avantor", MERCK: "Merck",
                DISTRIBUTOR: "Multi-brand distributors", OTHER: "Others",
                IVD: "IVD / diagnostics (out of scope)"}

# Names that contain a group keyword but are a different company.
_EXCLUDE = [
    # US Merck & Co. (MSD) is unrelated to Merck KGaA
    r"\bmerck sharp\b", r"\bsharp\s*&?\s*dohme\b", r"\bmsd\b",
]

_PATTERNS: dict[str, list[str]] = {
    THERMO: [
        r"\bthermo\s*fisc?her\b", r"^fisher$", r"\bfisher emergo\b", r"\bthermo electron\b", r"\bthermo scientific\b",
        r"\bfisher scientific\b", r"\bfisher bioblock\b", r"\bfisher clinical\b",
        r"\blife technologies\b", r"\binvitrogen\b", r"\bapplied biosystems\b",
        r"\balfa aesar\b", r"\bacros organics\b", r"\boxoid\b", r"\bperbio\b",
        r"\bunity lab services\b", r"\bpatheon\b",
    ],
    AVANTOR: [
        r"\bavantor\b", r"\bvwr\b", r"\bj\.?\s?t\.?\s?baker\b", r"\bmacron fine chemicals\b",
        r"\bnusil\b", r"\bmasterflex\b", r"\bbdh prolabo\b", r"\bamresco\b",
    ],
    MERCK: [
        r"\bmerck\b", r"\bsigma[\s-]*aldrich\b", r"\bmillipore\b", r"\bmilliporesigma\b",
        r"\bsupelco\b", r"\bsafc\b", r"\bbioreliance\b",
    ],
    DISTRIBUTOR: [
        r"\bscientific laboratory supplies\b", r"\bth\.?\s?geyer\b", r"\bcamlab\b", r"\bappleton woods\b",
        r"\bthistle scientific\b", r"\bdutscher\b", r"\blabo moderne\b", r"\bpremier scientific\b",
        r"\btotal lab supplies\b", r"\blondon laboratory equipment\b", r"\btriolab\b", r"\bnordic biosite\b",
        r"\bclinisciences\b", r"\blabelians\b", r"\bgrosseron\b", r"\bhounisen\b", r"\bbie\s*(&|og|and)\s*berntsen\b",
        r"\bomnilab\b", r"\bbiozym\b",
    ],
    IVD: [
        r"\broche\b", r"\bsysmex\b", r"\bsiemens healthineers\b", r"\bsiemens healthcare\b",
        r"\babbott\b", r"\bbeckman coulter\b", r"\bcepheid\b", r"\bradiometer\b", r"\bleica biosystems\b",
        r"\bbiomerieux\b", r"\bbio merieux\b", r"\bwerfen\b", r"\binstrumentation laboratory\b",
        r"\bbio-?rad\b.*\b(clinical|diagnost)", r"\bortho[- ]clinical\b", r"\bquidelortho\b",
        r"\bdiasorin\b", r"\bhologic\b", r"\bstago\b", r"\bimmucor\b", r"\bfujirebio\b", r"\bsebia\b",
        r"\bmindray\b", r"\bsnibe\b",
        r"\bdiagnost",  # generic: '... Diagnostics', 'Diagnostica ...', '... Diagnostic SAS'
    ],
}

_COMPILED = {g: [re.compile(p) for p in ps] for g, ps in _PATTERNS.items()}
_EXCLUDE_RE = [re.compile(p) for p in _EXCLUDE]


def _fold(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower()).strip()


_MULTI_WINNER_RE = re.compile(r"\blots?\b")


def is_multi_winner_string(name: str | None) -> bool:
    """Buyer typed several winners with their lots into one field ('X (lot 4) - Y (lots 2-5) - ...')."""
    return isinstance(name, str) and len(_MULTI_WINNER_RE.findall(_fold(name))) >= 3


def winner_group(name: str | None) -> str:
    if not isinstance(name, str) or not name:  # None or pandas NaN
        return OTHER
    if is_multi_winner_string(name):           # cannot be attributed to one group
        return OTHER
    s = _fold(name)
    if any(r.search(s) for r in _EXCLUDE_RE):
        return OTHER
    for group, regexes in _COMPILED.items():
        if any(r.search(s) for r in regexes):
            return group
    return OTHER


_LEGAL_SUFFIX = re.compile(
    r"[,.]?\s*\b(gmbh|ag|sas|sasu|sa|sarl|s\.a\.s\.?|s\.a\.?|bv|b\.v\.|nv|n\.v\.|ltd|limited|plc|inc|ab|as|a/s|oy|oyj|"
    r"aps|kg|co\.? kg|& co\.? kg|e\.k\.|srl|spa|ehf|llc|corp)\b\.?"
)


def clean_name(name: str | None) -> str:
    """Folded name without legal suffixes, for grouping the 'other' bucket by company."""
    if not isinstance(name, str) or not name:  # None or pandas NaN
        return ""
    s = _fold(name)
    prev = None
    while prev != s:
        prev, s = s, _LEGAL_SUFFIX.sub("", s).strip(" ,.-&+")
    return s
