from dataclasses import dataclass
from .text_utils import normalize_key


@dataclass(frozen=True)
class FieldSpec:
    labels: tuple[str, ...]
    min_value: int
    absolute: bool

FIELD_SPECS = {
    "total_assets": FieldSpec(("total des actifs", "total actifs", "total actif", "total assets"), 10_000, True),
    "net_assets": FieldSpec(("total des capitaux propres", "total capitaux propres", "total equity", "shareholders equity"), 1_000, True),
    "revenue": FieldSpec(("total produit net bancaire", "produit net bancaire", "net banking income", "total revenue", "revenue"), 1_000, True),
    "expenses": FieldSpec(("total charges d exploitation bancaire", "total des charges d exploitation bancaire", "total operating expenses", "total expenses"), 1_000, True),
    "net_profit": FieldSpec(("resultat net de l exercice", "resultat de l exercice", "net profit for the year", "profit for the year", "net income"), 1_000, False),
}


def best_label_match(line: str, labels: tuple[str, ...]) -> tuple[str | None, int, float]:
    key = normalize_key(line)
    best = (None, -1, 0.0)
    for label in labels:
        idx = key.find(label)
        if idx < 0:
            continue
        score = 0.80
        if key.startswith(label):
            score += 0.08
        if "total" in label:
            score += 0.06
        score += min(len(label) / 100.0, 0.05)
        if score > best[2]:
            # Approximate raw-line end using normalized-label occurrence; enough to avoid numbers in codes before label.
            raw_lower = line.lower()
            raw_idx = raw_lower.find(label.split()[0])
            label_end = max(0, raw_idx) + len(label)
            best = (label, label_end, min(score, 0.99))
    return best
