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
    page: int | None = None
    line_index: int | None = None
    line: str | None = None
    matched_label: str | None = None
    def dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

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

def detect_company(first_pages: str, filename: str) -> Evidence:
    head = normalize(first_pages)
    aliases = [
        (("assurances biat",), "ASSURANCES BIAT", "insurance"),
        (("banque nationale agricole", "bna bank"), "BANQUE NATIONALE AGRICOLE - BNA BANK", "bank"),
        (("bh bank",), "BH BANK", "bank"),
        (("amen bank",), "AMEN BANK", "bank"),
        (("banque internationale arabe de tunisie",), "BANQUE INTERNATIONALE ARABE DE TUNISIE - BIAT", "bank"),
    ]
    for names, canonical, profile in aliases:
        for name in names:
            if normalize(name) in head:
                return Evidence({"name": canonical, "profile": profile}, 0.99, "first_pages_known_alias", matched_label=name)
    fn = normalize(filename)
    if "assurances" in fn and "biat" in fn:
        return Evidence({"name": "ASSURANCES BIAT", "profile": "insurance"}, 0.80, "filename_alias")
    return Evidence({"name": None, "profile": None}, 0.0, "missing")

def detect_profile(full_text: str, first_pages: str, company_profile: str | None) -> Evidence:
    if company_profile in {"bank", "insurance"}:
        return Evidence(company_profile, 0.99, "known_company_profile")
    head, full = normalize(first_pages), normalize(full_text)
    terms = {
        "insurance": ("etat de resultat technique de l assurance", "provisions techniques", "primes acquises", "charges de sinistres"),
        "bank": ("produit net bancaire", "charges d exploitation bancaire", "depots et avoirs de la clientele", "creances sur la clientele"),
        "corporate": ("chiffre d affaires", "cout des ventes", "resultat d exploitation"),
    }
    scores = {p: 0 for p in terms}
    evidence = {p: [] for p in terms}
    for profile, labels in terms.items():
        for label in labels:
            item = normalize(label)
            if item in head:
                scores[profile] += 5; evidence[profile].append(f"header:{label}")
            elif item in full:
                scores[profile] += 2; evidence[profile].append(f"document:{label}")
    profile = max(scores, key=scores.get)
    if scores[profile] == 0:
        return Evidence("generic", 0.40, "profile_detector")
    return Evidence(profile, min(0.98, 0.65 + scores[profile]*0.03), "profile_detector", matched_label=", ".join(evidence[profile]))

def detect_period(first_pages: str) -> Evidence:
    head = normalize(first_pages); years = []
    for pattern in (r"31[ /.-]*12[ /.-]*(20\d{2})", r"31 decembre (20\d{2})"):
        years.extend(re.findall(pattern, head))
    return Evidence(f"31/12/{max(years)}", 0.98, "first_pages_period") if years else Evidence(None, 0.0, "missing")

def detect_currency(first_pages: str, full_text: str) -> Evidence:
    head, full = normalize(first_pages), normalize(full_text)
    for label in ("unite en millier de dinars tunisiens", "unite en milliers de dinars tunisiens", "unite en mille dinars", "unite en milliers de dinars", "les chiffres sont exprimes en mille dinars tunisiens", "les chiffres sont exprimes en milliers de dinars tunisiens", "kdt", "mdt"):
        if normalize(label) in head or normalize(label) in full[:15000]:
            return Evidence("KDT", 0.99, "unit_header", matched_label=label)
    for label in ("unite en dinars tunisiens", "unite en dinars", "dinars tunisiens"):
        if normalize(label) in head:
            return Evidence("TND", 0.98, "unit_header", matched_label=label)
    if re.search(r"\beur\b|euros?", head):
        return Evidence("EUR", 0.90, "unit_header")
    if "dinar" in full:
        return Evidence("TND", 0.75, "currency_text")
    return Evidence(None, 0.0, "missing")

def detect_page_columns(page_text: str) -> int:
    maximum = 0
    for line in split_lines(page_text[:3500]):
        count = len(re.findall(r"31[ /.-]*12[ /.-]*20\d{2}", normalize(line)))
        maximum = max(maximum, count)
    if maximum >= 3:
        return 3
    if maximum >= 2:
        return 2
    head = normalize(page_text[:2500])
    if "2024 publie" in head and "2024 retraite" in head:
        return 3
    return 2

def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"\(?-?\d+\)?", normalize_spaces(text))

def valid_amount_group(group: list[str]) -> bool:
    if not 1 <= len(group) <= 4: return False
    clean = [re.sub(r"\D", "", token) for token in group]
    if not clean[0] or len(clean[0]) > 3: return False
    return all(len(token) == 3 for token in clean[1:])

def group_to_int(group: list[str]) -> int:
    raw = " ".join(group); value = int(re.sub(r"\D", "", raw))
    return -value if "-" in raw or ("(" in raw and ")" in raw) else value

def all_partitions(tokens: list[str], columns: int) -> list[list[int]]:
    if columns <= 0 or len(tokens) < columns: return []
    results = []
    for cuts in combinations(range(1, len(tokens)), columns - 1):
        positions = (0,) + cuts + (len(tokens),)
        groups = [tokens[positions[i]:positions[i+1]] for i in range(columns)]
        if all(valid_amount_group(group) for group in groups):
            results.append([group_to_int(group) for group in groups])
    return results

def partition_score(values: list[int]) -> tuple:
    lengths = [len(str(abs(v))) for v in values]
    tiny = sum(1 for v in values if abs(v) < 1000)
    spread = max(lengths) - min(lengths)
    positive = [abs(v) for v in values if v != 0]
    ratio = max(positive)/max(1,min(positive)) if positive else 1
    return tiny, spread, ratio

def choose_partition(tokens: list[str], columns: int) -> list[int]:
    candidates = all_partitions(tokens, columns)
    return min(candidates, key=partition_score) if candidates else []

def parse_comparative_amounts(line: str, label: str, columns: int, accounting_identity: bool=False) -> list[int]:
    nl, nlabel = normalize(line), normalize(label)
    pos = nl.find(nlabel)
    if pos == -1:
        return []

    tail = nl[pos + len(nlabel):]
    has_percentage = bool(re.search(r"\b\d+(?:[.,]\d+)?\s*%", tail))
    tail = re.sub(r"\b\d+(?:[.,]\d+)?\s*%", " ", tail)
    tail = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " ", tail)
    tokens = numeric_tokens(tail)

    # Rows with a trailing variation percentage normally contain:
    # current year | prior year | variation amount.
    preferred_columns = 3 if has_percentage and columns < 3 else columns

    token_sets: list[tuple[list[str], int]] = [(tokens, 0)]

    # Remove only an explicit parenthesized note marker without penalty.
    if tokens and re.fullmatch(r"\(\d{1,2}\)", tokens[0]):
        token_sets.insert(0, (tokens[1:], 0))

    # Some PDFs expose an unparenthesized note number before the values
    # (for example BH Bank: "Total capitaux propres 12 1 373 273 ...").
    # Keep this as a fallback with a penalty so a real leading amount such as
    # BNA's "2 372 903" is never removed when the full row partitions cleanly.
    if tokens and len(re.sub(r"\D", "", tokens[0])) <= 2:
        token_sets.append((tokens[1:], 2))

    candidates: list[tuple[list[int], int, int]] = []
    for token_set, removal_penalty in token_sets:
        for candidate_columns in dict.fromkeys((preferred_columns, columns, 3, 2)):
            values = choose_partition(token_set, candidate_columns)
            if values:
                column_penalty = 0 if candidate_columns == preferred_columns else 1
                candidates.append((values, removal_penalty, column_penalty))

    if not candidates:
        return []

    if accounting_identity:
        identity = [item for item in candidates if len(item[0]) >= 3]
        if identity:
            return min(
                identity,
                key=lambda item: (
                    abs(abs(item[0][0]) - abs(item[0][1]) - abs(item[0][2])),
                    item[1],
                    item[2],
                ),
            )[0]

    return min(
        candidates,
        key=lambda item: (
            item[1],
            item[2],
            partition_score(item[0]),
        ),
    )[0]

def find_amount(page_texts: list[str], labels: tuple[str,...], value_index:int=0, confidence:float=0.95, reject:tuple[str,...]=(), exact_total:bool=False, accounting_identity:bool=False, columns_override:int|None=None) -> Evidence:
    matches = []
    for page_number, page_text in enumerate(page_texts, 1):
        columns = columns_override or detect_page_columns(page_text)
        for line_index, line in enumerate(split_lines(page_text)):
            nl = normalize(line)
            if any(normalize(x) in nl for x in reject): continue
            matched = next((label for label in labels if normalize(label) in nl), None)
            if matched is None: continue
            if exact_total and "total" not in nl: continue
            values = parse_comparative_amounts(line, matched, columns, accounting_identity)
            if value_index >= len(values): continue
            score = (5 if nl.startswith(normalize(matched)) else 0) + (5 if "total" in nl else 0) + (5 if exact_total else 0) - page_number*0.01
            matches.append((score, Evidence(values[value_index], confidence, "deterministic_page_column_parser", page_number, line_index, line, matched)))
    return max(matches, key=lambda x:x[0])[1] if matches else Evidence(None,0.0,"missing")

def find_insurance_premiums(page_texts: list[str]) -> list[Evidence]:
    results=[]
    for pno, page in enumerate(page_texts,1):
        for lno,line in enumerate(split_lines(page)):
            nl=normalize(line)
            if "primes acquises" not in nl or ("totalise" not in nl and "s eleve" not in nl): continue
            vals=[int(g.replace(" ","")) for g in re.findall(r"\d{1,3}(?: \d{3})+", normalize_spaces(line))]
            if vals and vals[0] not in [x.value for x in results]:
                results.append(Evidence(vals[0],0.97,"insurance_narrative_total",pno,lno,line,"primes acquises"))
    return results[:2]

class FinancialExtractor:
    def extract(self, *, full_text:str, page_texts:list[str], filename:str="") -> Result:
        first_pages="\n".join(page_texts[:3])
        company=detect_company(first_pages, filename); data=company.value or {"name":None,"profile":None}
        profile=detect_profile(full_text, first_pages, data.get("profile")); period=detect_period(first_pages); currency=detect_currency(first_pages,full_text)
        result=Result(company_name=data.get("name"), report_profile=profile.value, document_type={"bank":"Etats financiers - banque","insurance":"Etats financiers - assurance","corporate":"Etats financiers - entreprise"}.get(profile.value,"Etats financiers"), period=period.value, currency=currency.value, extraction_details={"company_name":{**company.dict(),"value":data.get("name")},"document_type":{"value":profile.value,"confidence":profile.confidence,"source":profile.source,"evidence":profile.matched_label},"period":period.dict(),"currency":currency.dict()})
        if profile.value=="insurance": self._insurance(page_texts,result)
        elif profile.value=="bank": self._bank(page_texts,result)
        else: self._corporate(page_texts,result)
        self._validate(result); return result
    def _bank(self,pages,result):
        fields={
            "total_assets":find_amount(pages,("total des actifs","total actifs"),confidence=0.99,exact_total=True),
            "net_assets":find_amount(pages,("total des capitaux propres","total capitaux propres"),confidence=0.99,exact_total=True,reject=("et passifs",)),
            "revenue":find_amount(pages,("total produit net bancaire","produit net bancaire"),confidence=0.98),
            "expenses":find_amount(pages,("total charges d exploitation bancaire",),confidence=0.98,exact_total=True),
            "net_profit":find_amount(pages,("resultat net de l exercice","resultat de l exercice"),confidence=0.98,reject=("capitaux propres","resultat de base par action","resultat dilue par action")),
        }; self._apply(fields,result)
        if result.expenses is not None:
            result.expenses = abs(result.expenses)
            result.extraction_details["expenses"]["value"] = result.expenses
    def _insurance(self,pages,result):
        fields={
            "total_assets":find_amount(pages,("total actifs",),value_index=2,confidence=0.99,exact_total=True,accounting_identity=True,columns_override=4),
            "net_assets":find_amount(pages,("total capitaux propres avant affectation","solde au 31 12 2025 avant affectation"),confidence=0.99,reject=("avant resultat de l exercice",),columns_override=2),
            "net_profit":find_amount(pages,("resultat net de l exercice","resultat de l exercice"),confidence=0.98,reject=("capitaux propres","avant resultat"),columns_override=2),
        }; self._apply(fields,result)
        premiums=find_insurance_premiums(pages)
        if len(premiums)>=2: result.revenue=abs(premiums[0].value)+abs(premiums[1].value)
        result.extraction_details["revenue"]={"value":result.revenue,"confidence":0.96 if result.revenue is not None else 0.0,"source":"insurance_earned_premiums_sum","components":[x.value for x in premiums]}
        result.financial_indicators=[{"label":"life_earned_premiums","value":premiums[0].value if premiums else None},{"label":"non_life_earned_premiums","value":premiums[1].value if len(premiums)>1 else None}]
    def _corporate(self,pages,result):
        fields={
            "total_assets":find_amount(pages,("total actif","total des actifs"),confidence=0.92,exact_total=True),
            "net_assets":find_amount(pages,("total capitaux propres",),confidence=0.90,exact_total=True),
            "revenue":find_amount(pages,("chiffre d affaires","produits d exploitation"),confidence=0.90),
            "expenses":find_amount(pages,("charges d exploitation","cout des ventes"),confidence=0.88),
            "net_profit":find_amount(pages,("resultat net de l exercice","resultat de l exercice"),confidence=0.92),
        }; self._apply(fields,result)
    @staticmethod
    def _apply(fields,result):
        for name,e in fields.items(): setattr(result,name,e.value); result.extraction_details[name]=e.dict()
    @staticmethod
    def _validate(result):
        warnings=[]
        for name,value in (("total_assets",result.total_assets),("net_assets",result.net_assets),("net_profit",result.net_profit)):
            if value is None: warnings.append(f"{name} was not detected.")
            elif not isinstance(value,int) or value<=0: warnings.append(f"{name} must be a positive financial amount.")
            elif value<1000: warnings.append(f"{name} is implausibly small.")
        if result.report_profile=="bank":
            for name,value in (("revenue",result.revenue),("expenses",result.expenses)):
                if value is None: warnings.append(f"{name} was not detected.")
                elif abs(value)<1000: warnings.append(f"{name} is implausibly small.")
        if result.total_assets is not None and result.net_assets is not None and result.net_assets > result.total_assets:
            warnings.append("Net assets cannot be greater than total assets.")
        if result.report_profile == "bank" and result.revenue is not None and result.net_profit is not None and result.net_profit > result.revenue:
            warnings.append("Net profit cannot be greater than banking revenue.")
        if result.report_profile == "bank" and result.total_assets and result.net_assets and result.net_assets < result.total_assets * 0.01:
            warnings.append("Net assets are implausibly small compared with total assets.")
        if result.period is None: warnings.append("Reporting period was not detected.")
        if result.company_name is None: warnings.append("Company name was not detected.")
        result.validation_warnings=warnings; result.validation_status="valid" if not warnings else "needs_review"
