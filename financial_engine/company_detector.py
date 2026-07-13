import re
from .text_utils import normalize_pdf_text, normalize_key

KNOWN_PATTERNS: list[tuple[str, list[str]]] = [
    ("BANQUE NATIONALE AGRICOLE - BNA BANK", [
        "banque nationale agricole bna bank", "banque nationale agricole bna", "bna bank"
    ]),
    ("AMEN BANK", ["amen bank"]),
    ("BH BANK", ["bh bank"]),
    ("BANQUE INTERNATIONALE ARABE DE TUNISIE - BIAT", [
        "banque internationale arabe de tunisie", "biat"
    ]),
    ("ATTIJARI BANK", ["attijari bank"]),
    ("BANQUE DE TUNISIE", ["banque de tunisie"]),
    ("STB BANK", ["stb bank", "societe tunisienne de banque"]),
    ("UIB", ["union internationale de banques", "uib"]),
    ("UBCI", ["union bancaire pour le commerce et l industrie", "ubci"]),
    ("BANQUE ZITOUNA", ["banque zitouna", "zitouna bank"]),
]

IGNORED = {
    "avis des societes", "etats financiers", "etats financiers consolides", "bilan",
    "etat de resultat", "etat des engagements hors bilan", "etat de flux de tresorerie",
    "notes aux etats financiers", "rapport general"
}


def detect_company_name(first_pages_text: str, filename: str | None = None) -> tuple[str | None, float, str]:
    text = normalize_pdf_text(first_pages_text)
    header = "\n".join(text.splitlines()[:120])
    key = normalize_key(header)

    for canonical, aliases in KNOWN_PATTERNS:
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", key):
                return canonical, 0.99, "first_pages_known_alias"

    # Filename is a useful but weaker fallback.
    filename_key = normalize_key(filename or "")
    for canonical, aliases in KNOWN_PATTERNS:
        if any(alias in filename_key for alias in aliases):
            return canonical, 0.82, "filename_alias"

    for line in header.splitlines():
        candidate = line.strip()
        line_key = normalize_key(candidate)
        if not candidate or len(candidate) < 4 or line_key in IGNORED:
            continue
        if any(x in line_key for x in ["siege social", "arrete au", "unite", "commissaire", "publie ci dessous"]):
            continue
        if re.fullmatch(r"[A-ZÀ-ÖØ-Ý0-9][A-ZÀ-ÖØ-Ý0-9 &'’.\-]{3,}", candidate):
            return candidate, 0.72, "first_pages_uppercase_line"
    return None, 0.0, "missing"
