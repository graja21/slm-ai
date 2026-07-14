from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any


SPACE_CHARS = "\u00a0\u2007\u202f\u2009\u200a\u200b\u2060\ufeff"


def normalize_spaces(text: str) -> str:
    for char in SPACE_CHARS:
        text = text.replace(char, " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize(text: str) -> str:
    text = normalize_spaces(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[’'`´-]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9%/.,() ]+", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def split_lines(text: str) -> list[str]:
    return [normalize_spaces(line) for line in text.splitlines() if normalize_spaces(line)]


@dataclass
class Evidence:
    value: Any
    confidence: float
    source: str
    line_index: int | None = None
    line: str | None = None
    matched_label: str | None = None

    def dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class Result:
    company_name: str | None = None
    document_type: str = "Etats financiers"
    report_profile: str = "generic"
    period: str | None = None
    total_assets: int | None = None
    net_assets: int | None = None
    revenue: int | None = None
    net_profit: int | None = None
    expenses: int | None = None
    growth_rate: float | None = None
    currency: str | None = None
    important_dates: list[str] = field(default_factory=list)
    financial_indicators: list[dict[str, Any]] = field(default_factory=list)
    risks_or_observations: list[Any] = field(default_factory=list)
    summary: str | None = None
    extraction_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    validation_warnings: list[str] = field(default_factory=list)
    validation_status: str = "valid"

    def dict(self) -> dict[str, Any]:
        return asdict(self)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\(?-?\d+\)?", normalize_spaces(text))


def _group_valid(group: list[str]) -> bool:
    if not 1 <= len(group) <= 4:
        return False

    clean = [re.sub(r"\D", "", token) for token in group]

    if not clean[0] or len(clean[0]) > 3:
        return False

    return all(len(item) == 3 for item in clean[1:])


def _group_to_int(group: list[str]) -> int:
    raw = " ".join(group)
    value = int(re.sub(r"\D", "", raw))
    return -value if "-" in raw or ("(" in raw and ")" in raw) else value


def all_partitions(text: str, columns: int) -> list[list[int]]:
    """
    Return every valid partition into the requested number of accounting columns.
    Each amount may contain between 1 and 4 digit groups.
    """
    tokens = _tokens(text)
    count = len(tokens)

    if columns <= 0 or count < columns:
        return []

    results: list[list[int]] = []

    for cuts in combinations(range(1, count), columns - 1):
        positions = (0,) + cuts + (count,)
        groups = [
            tokens[positions[index]:positions[index + 1]]
            for index in range(columns)
        ]

        if all(_group_valid(group) for group in groups):
            results.append([_group_to_int(group) for group in groups])

    return results


def choose_partition(
    text: str,
    columns: int,
    *,
    accounting_identity: bool = False,
) -> list[int]:
    candidates = all_partitions(text, columns)

    if not candidates:
        return []

    if accounting_identity and columns >= 3:
        # Insurance total assets:
        # gross assets - amortisation/provisions ~= net current-year assets.
        def score(values: list[int]) -> tuple[int, int]:
            difference = abs(abs(values[0]) - abs(values[1]) - abs(values[2]))
            # Prefer sensible column magnitudes if two candidates tie.
            magnitude_penalty = 0
            if abs(values[0]) < abs(values[2]):
                magnitude_penalty += abs(values[2]) - abs(values[0])
            return difference, magnitude_penalty

        return min(candidates, key=score)

    # For ordinary comparative rows, prefer balanced group sizes and descending
    # current/prior magnitudes only as a weak tie breaker.
    def generic_score(values: list[int]) -> tuple[int, int]:
        implausible = sum(1 for value in values if abs(value) < 1000)
        spread = max(len(str(abs(value))) for value in values) - min(
            len(str(abs(value))) for value in values
        )
        return implausible, spread

    return min(candidates, key=generic_score)


def numeric_tail(line: str, label: str) -> str:
    normalized_line = normalize(line)
    normalized_label = normalize(label)
    position = normalized_line.find(normalized_label)

    if position == -1:
        return ""

    tail = normalized_line[position + len(normalized_label):]
    tail = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " ", tail)
    tail = re.sub(r"\b\d+(?:[.,]\d+)?\s*%", " ", tail)

    return normalize_spaces(tail)


def detect_profile(full_text: str, head: str) -> Evidence:
    h = normalize(head)
    f = normalize(full_text)

    terms = {
        "insurance": [
            "assurances",
            "reassurance",
            "primes acquises",
            "provisions techniques",
            "charges de sinistres",
        ],
        "bank": [
            "produit net bancaire",
            "charges d exploitation bancaire",
            "depots de la clientele",
        ],
        "corporate": [
            "chiffre d affaires",
            "resultat d exploitation",
            "cout des ventes",
        ],
    }

    scores: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}

    for profile, labels in terms.items():
        scores[profile] = 0
        evidence[profile] = []

        for label in labels:
            item = normalize(label)

            if item in h:
                scores[profile] += 3
                evidence[profile].append(f"header:{label}")
            elif item in f:
                scores[profile] += 1
                evidence[profile].append(f"document:{label}")

    profile = max(scores, key=scores.get)

    if scores[profile] == 0:
        return Evidence("generic", 0.40, "profile_detector")

    return Evidence(
        profile,
        min(0.99, 0.65 + scores[profile] * 0.04),
        "profile_detector",
        matched_label=", ".join(evidence[profile]),
    )


def detect_company(head: str, filename: str) -> Evidence:
    h = normalize(head)

    aliases = [
        (["assurances biat"], "ASSURANCES BIAT"),
        (
            ["banque nationale agricole", "bna bank"],
            "BANQUE NATIONALE AGRICOLE - BNA BANK",
        ),
        (["bh bank"], "BH BANK"),
        (["amen bank"], "AMEN BANK"),
        (
            ["banque internationale arabe de tunisie"],
            "BANQUE INTERNATIONALE ARABE DE TUNISIE - BIAT",
        ),
    ]

    for names, canonical in aliases:
        for name in names:
            if normalize(name) in h:
                return Evidence(
                    canonical,
                    0.99,
                    "first_pages_known_alias",
                    matched_label=name,
                )

    filename_normalized = normalize(filename)

    if "assurances" in filename_normalized and "biat" in filename_normalized:
        return Evidence("ASSURANCES BIAT", 0.75, "filename_alias")

    return Evidence(None, 0.0, "missing")


def detect_period(head: str) -> Evidence:
    h = normalize(head)
    years: list[str] = []

    for pattern in (
        r"31[ /.-]*12[ /.-]*(20\d{2})",
        r"31 decembre (20\d{2})",
    ):
        years.extend(re.findall(pattern, h))

    if years:
        return Evidence(f"31/12/{max(years)}", 0.98, "first_pages_period")

    return Evidence(None, 0.0, "missing")


def detect_currency(head: str, full_text: str) -> Evidence:
    h = normalize(head)

    for label in (
        "unite : en dinars",
        "unite: en dinars",
        "chiffres sont exprimes en dt",
        "dinars tunisiens",
    ):
        if normalize(label) in h:
            return Evidence("TND", 0.99, "unit_header", matched_label=label)

    if "milliers de dinars" in h or "kdt" in h:
        return Evidence("KDT", 0.98, "unit_header")

    if re.search(r"\beur\b|euros?", h):
        return Evidence("EUR", 0.90, "unit_header")

    if "dinars" in normalize(full_text):
        return Evidence("TND", 0.80, "currency_text")

    return Evidence(None, 0.0, "missing")


def find_amount(
    all_lines: list[str],
    labels: tuple[str, ...],
    *,
    value_index: int,
    expected_columns: int,
    confidence: float,
    reject: tuple[str, ...] = (),
    accounting_identity: bool = False,
    before: int = 0,
    after: int = 2,
) -> Evidence:
    for index, line in enumerate(all_lines):
        normalized_line = normalize(line)

        if any(normalize(item) in normalized_line for item in reject):
            continue

        matched = next(
            (label for label in labels if normalize(label) in normalized_line),
            None,
        )

        if matched is None:
            continue

        start = max(0, index - before)
        end = min(len(all_lines), index + after + 1)

        candidate_lines = [line]

        for nearby_index in range(start, end):
            if nearby_index != index:
                candidate_lines.append(f"{line} {all_lines[nearby_index]}")

        for candidate_line in candidate_lines:
            tail = numeric_tail(candidate_line, matched)

            values = choose_partition(
                tail,
                expected_columns,
                accounting_identity=accounting_identity,
            )

            if value_index < len(values):
                return Evidence(
                    value=values[value_index],
                    confidence=confidence,
                    source="deterministic_column_parser",
                    line_index=index,
                    line=candidate_line,
                    matched_label=matched,
                )

    return Evidence(None, 0.0, "missing")


def grouped_amounts(text: str) -> list[int]:
    cleaned = re.sub(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        " ",
        normalize_spaces(text),
    )

    groups = re.findall(
        r"(?<!\d)(?:\d{1,3}(?: \d{3})+)(?!\d)",
        cleaned,
    )

    return [int(group.replace(" ", "")) for group in groups]


def find_insurance_premiums(all_lines: list[str]) -> list[Evidence]:
    results: list[Evidence] = []

    for index, line in enumerate(all_lines):
        normalized_line = normalize(line)

        if "primes acquises" not in normalized_line:
            continue

        if not (
            "totalise" in normalized_line
            or "s eleve" in normalized_line
        ):
            continue

        values = grouped_amounts(line)

        # First grouped amount after removing the date is the current-year total.
        if values:
            value = values[0]

            if value not in [item.value for item in results]:
                results.append(
                    Evidence(
                        value=value,
                        confidence=0.97,
                        source="insurance_narrative_total",
                        line_index=index,
                        line=line,
                        matched_label="primes acquises",
                    )
                )

    return results[:2]


class FinancialExtractor:
    def extract(
        self,
        *,
        full_text: str,
        page_texts: list[str],
        filename: str = "",
    ) -> Result:
        head = "\n".join(page_texts[:3])

        profile = detect_profile(full_text, head)
        company = detect_company(head, filename)
        period = detect_period(head)
        currency = detect_currency(head, full_text)

        result = Result(
            company_name=company.value,
            report_profile=profile.value,
            document_type={
                "bank": "Etats financiers - banque",
                "insurance": "Etats financiers - assurance",
                "corporate": "Etats financiers - entreprise",
            }.get(profile.value, "Etats financiers"),
            period=period.value,
            currency=currency.value,
            extraction_details={
                "company_name": company.dict(),
                "document_type": {
                    "value": profile.value,
                    "confidence": profile.confidence,
                    "source": profile.source,
                    "evidence": profile.matched_label,
                },
                "period": period.dict(),
                "currency": currency.dict(),
            },
        )

        all_lines = split_lines(full_text)

        if profile.value == "insurance":
            self._insurance(all_lines, result)
        elif profile.value == "bank":
            self._bank(all_lines, result)
        else:
            self._corporate(all_lines, result)

        self._validate(result)

        return result

    def _bank(self, lines: list[str], result: Result) -> None:
        fields = {
            "total_assets": find_amount(
                lines,
                ("total des actifs", "total actifs"),
                value_index=0,
                expected_columns=3,
                confidence=0.99,
            ),
            "net_assets": find_amount(
                lines,
                ("total des capitaux propres", "total capitaux propres"),
                value_index=0,
                expected_columns=3,
                confidence=0.99,
            ),
            "revenue": find_amount(
                lines,
                ("total produit net bancaire", "produit net bancaire"),
                value_index=0,
                expected_columns=3,
                confidence=0.98,
            ),
            "expenses": find_amount(
                lines,
                (
                    "total charges d exploitation bancaire",
                    "charges d exploitation bancaire",
                ),
                value_index=0,
                expected_columns=3,
                confidence=0.98,
            ),
            "net_profit": find_amount(
                lines,
                (
                    "resultat net de l exercice",
                    "resultat de l exercice",
                ),
                value_index=0,
                expected_columns=3,
                confidence=0.95,
            ),
        }

        self._apply(fields, result)

    def _insurance(self, lines: list[str], result: Result) -> None:
        total_assets = find_amount(
            lines,
            ("total actifs",),
            value_index=2,
            expected_columns=4,
            confidence=0.99,
            accounting_identity=True,
        )

        net_assets = find_amount(
            lines,
            (
                "total capitaux propres avant affectation",
                "solde au 31 12 2025 avant affectation",
            ),
            value_index=0,
            expected_columns=2,
            confidence=0.99,
            reject=("avant resultat de l exercice",),
        )

        net_profit = find_amount(
            lines,
            (
                "resultat net de l exercice",
                "resultat de l exercice",
            ),
            value_index=0,
            expected_columns=2,
            confidence=0.98,
            reject=(
                "capitaux propres",
                "avant resultat",
            ),
            before=8,
            after=3,
        )

        self._apply(
            {
                "total_assets": total_assets,
                "net_assets": net_assets,
                "net_profit": net_profit,
            },
            result,
        )

        premiums = find_insurance_premiums(lines)

        if len(premiums) >= 2:
            result.revenue = abs(premiums[0].value) + abs(premiums[1].value)

        result.extraction_details["revenue"] = {
            "value": result.revenue,
            "confidence": 0.96 if result.revenue is not None else 0.0,
            "source": "insurance_earned_premiums_sum",
            "components": [item.value for item in premiums],
        }

        result.financial_indicators = [
            {
                "label": "life_earned_premiums",
                "value": premiums[0].value if premiums else None,
            },
            {
                "label": "non_life_earned_premiums",
                "value": premiums[1].value if len(premiums) > 1 else None,
            },
        ]

    def _corporate(self, lines: list[str], result: Result) -> None:
        fields = {
            "total_assets": find_amount(
                lines,
                ("total actif", "total des actifs"),
                value_index=0,
                expected_columns=2,
                confidence=0.92,
            ),
            "net_assets": find_amount(
                lines,
                ("total capitaux propres",),
                value_index=0,
                expected_columns=2,
                confidence=0.90,
            ),
            "revenue": find_amount(
                lines,
                ("chiffre d affaires", "produits d exploitation"),
                value_index=0,
                expected_columns=2,
                confidence=0.90,
            ),
            "expenses": find_amount(
                lines,
                ("charges d exploitation", "cout des ventes"),
                value_index=0,
                expected_columns=2,
                confidence=0.88,
            ),
            "net_profit": find_amount(
                lines,
                (
                    "resultat net de l exercice",
                    "resultat de l exercice",
                ),
                value_index=0,
                expected_columns=2,
                confidence=0.92,
            ),
        }

        self._apply(fields, result)

    @staticmethod
    def _apply(fields: dict[str, Evidence], result: Result) -> None:
        for name, evidence in fields.items():
            setattr(result, name, evidence.value)
            result.extraction_details[name] = evidence.dict()

    @staticmethod
    def _validate(result: Result) -> None:
        warnings: list[str] = []

        if result.total_assets is None:
            warnings.append("Total assets were not detected.")
        elif result.total_assets <= 0:
            warnings.append("Total assets must be positive.")

        if (
            result.total_assets is not None
            and result.net_assets is not None
            and result.net_assets > result.total_assets
        ):
            warnings.append("Net assets cannot be greater than total assets.")

        if result.period is None:
            warnings.append("Reporting period was not detected.")

        if result.company_name is None:
            warnings.append("Company name was not detected.")

        if result.net_profit is None:
            warnings.append("Net profit was not detected.")

        result.validation_warnings = warnings
        result.validation_status = "valid" if not warnings else "needs_review"
