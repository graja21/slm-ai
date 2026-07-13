from .amount_parser import select_current_period_amount
from .company_detector import detect_company_name
from .label_detector import FIELD_SPECS, best_label_match
from .metadata_detector import detect_currency, detect_period
from .text_utils import normalize_pdf_text
from .validator import validate_result


def _extract_field(lines: list[str], field: str) -> dict:
    spec = FIELD_SPECS[field]
    candidates: list[dict] = []
    for idx, line in enumerate(lines):
        label, label_end, base_score = best_label_match(line, spec.labels)
        if not label:
            continue
        value = select_current_period_amount(line, label_end=label_end, min_value=spec.min_value, absolute=spec.absolute)
        if value is None:
            continue
        # Official statements are typically near the beginning; notes later receive a small penalty.
        confidence = max(0.50, min(0.99, base_score - min(idx, 3000) / 20000))
        candidates.append({
            "value": value,
            "confidence": round(confidence, 2),
            "source": "deterministic_line_parser",
            "line_index": idx,
            "line": line.strip(),
            "matched_label": label,
        })
    if not candidates:
        return {"value": None, "confidence": 0.0, "source": "missing"}
    candidates.sort(key=lambda x: (x["confidence"], len(x["matched_label"])), reverse=True)
    return candidates[0]


def extract_financial_data(full_text: str, first_pages_text: str | None = None, filename: str | None = None) -> dict:
    text = normalize_pdf_text(full_text)
    lines = text.splitlines()
    first_pages_text = normalize_pdf_text(first_pages_text or "\n".join(lines[:250]))

    company, company_conf, company_source = detect_company_name(first_pages_text, filename)
    period, period_conf, period_source = detect_period(first_pages_text)
    currency, currency_conf, currency_source = detect_currency(first_pages_text)

    details = {field: _extract_field(lines, field) for field in FIELD_SPECS}
    extraction_details = {
        "company_name": {"value": company, "confidence": company_conf, "source": company_source},
        "document_type": {"value": "Etats financiers", "confidence": 0.90, "source": "deterministic_default"},
        "period": {"value": period, "confidence": period_conf, "source": period_source},
        "currency": {"value": currency, "confidence": currency_conf, "source": currency_source},
        **details,
    }

    result = {
        "company_name": company,
        "document_type": "Etats financiers",
        "period": period,
        "total_assets": details["total_assets"]["value"],
        "net_assets": details["net_assets"]["value"],
        "revenue": details["revenue"]["value"],
        "net_profit": details["net_profit"]["value"],
        "expenses": abs(details["expenses"]["value"]) if isinstance(details["expenses"]["value"], int) else None,
        "growth_rate": None,
        "currency": currency,
        "important_dates": [],
        "financial_indicators": [],
        "risks_or_observations": [],
        "summary": None,
        "extraction_details": extraction_details,
    }
    if isinstance(result["expenses"], int):
        result["extraction_details"]["expenses"]["value"] = result["expenses"]
    return validate_result(result)
