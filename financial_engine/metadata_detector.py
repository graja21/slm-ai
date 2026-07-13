import re
from .text_utils import normalize_pdf_text, normalize_key


def detect_period(text: str) -> tuple[str | None, float, str]:
    clean = normalize_pdf_text(text)
    patterns = [
        r"31\s*/\s*12\s*/\s*(20\d{2})",
        r"31\s+d[ée]cembre\s+(20\d{2})",
        r"exercice\s+clos\s+au\s+31\s+d[ée]cembre\s+(20\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, re.I)
        if match:
            return f"31/12/{match.group(1)}", 0.97, "date_regex"
    years = re.findall(r"\b20\d{2}\b", clean[:10000])
    if years:
        year = max(set(years), key=years.count)
        return year, 0.65, "dominant_year"
    return None, 0.0, "missing"


def detect_currency(text: str) -> tuple[str | None, float, str]:
    key = normalize_key(normalize_pdf_text(text)[:20000])
    if any(x in key for x in ["unite en 1 000 dt", "unite en mille dinars", "millier de dinars", "mille dinars", "mdt", "kdt"]):
        return "KDT", 0.98, "unit_header"
    if any(x in key for x in ["dinars tunisiens", "dinar tunisien", "tnd"]):
        return "TND", 0.90, "currency_text"
    if "eur" in key or "euro" in key:
        return "EUR", 0.90, "currency_text"
    if "usd" in key or "dollar" in key:
        return "USD", 0.90, "currency_text"
    return None, 0.0, "missing"
