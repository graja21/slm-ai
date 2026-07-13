import re
from dataclasses import dataclass

YEAR_VALUES = set(range(1990, 2101))
DATE_RE = re.compile(r"\b\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}\b")
PERCENT_RE = re.compile(r"\(?\s*\d+(?:[.,]\d+)?\s*%\s*\)?")
PAREN_RE = re.compile(r"\(\s*-?[\d \u00a0\u202f]+\s*\)")
DIGIT_GROUP_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class AmountCandidate:
    value: int
    raw: str
    start: int
    end: int
    negative: bool = False


def parse_amount(raw: str | int | float | None) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"na", "n/a", "null", "none", "-", "non claire"}:
        return None
    negative = text.startswith("-") or ("(" in text and ")" in text)
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    value = int(digits)
    return -value if negative else value


def _consume_plain_groups(groups: list[re.Match[str]]) -> list[AmountCandidate]:
    """Convert digit groups into individual financial amounts.

    Thousand-formatted amounts use a leading group of 1-3 digits followed by
    one or two groups of exactly three digits. We consume:
      1 087 093 -> 1087093
      15 244 878 -> 15244878
      590 069 -> 590069
      274 544 254 557 -> 274544, 254557
    """
    out: list[AmountCandidate] = []
    i = 0
    while i < len(groups):
        current = groups[i]
        token = current.group(0)
        value = int(token)

        # Ignore years as standalone values.
        if value in YEAR_VALUES:
            i += 1
            continue

        # Determine grouped-thousands length from the leading group.
        # 1-2 leading digits usually means millions (3 groups total),
        # 3 leading digits usually means thousands (2 groups total).
        desired = 3 if len(token) <= 2 else 2
        parts = [token]
        j = i + 1
        while j < len(groups) and len(parts) < desired:
            nxt = groups[j]
            if len(nxt.group(0)) != 3:
                break
            # Require whitespace separation so unrelated numbers are not glued.
            if nxt.start() > groups[j - 1].end() and not groups[j - 1].string[groups[j - 1].end():nxt.start()].isspace():
                break
            parts.append(nxt.group(0))
            j += 1

        if len(parts) >= 2:
            raw = " ".join(parts)
            out.append(AmountCandidate(int("".join(parts)), raw, current.start(), groups[j - 1].end()))
            i = j
        else:
            out.append(AmountCandidate(value, token, current.start(), current.end()))
            i += 1
    return out


def extract_amount_candidates(line: str) -> list[AmountCandidate]:
    if not line:
        return []
    clean = line.replace("\xa0", " ").replace("\u202f", " ")
    clean = DATE_RE.sub(" ", clean)
    clean = PERCENT_RE.sub(" ", clean)

    candidates: list[AmountCandidate] = []

    # Capture accounting negatives as complete units first.
    spans: list[tuple[int, int]] = []
    for match in PAREN_RE.finditer(clean):
        value = parse_amount(match.group(0))
        if value is not None:
            candidates.append(AmountCandidate(value, match.group(0), match.start(), match.end(), True))
            spans.append((match.start(), match.end()))

    chars = list(clean)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    remaining = "".join(chars)

    groups: list[re.Match[str]] = []
    for match in DIGIT_GROUP_RE.finditer(remaining):
        before = remaining[match.start() - 1] if match.start() > 0 else " "
        after = remaining[match.end()] if match.end() < len(remaining) else " "
        if before.isalpha() or after.isalpha():
            continue
        groups.append(match)

    candidates.extend(_consume_plain_groups(groups))
    candidates.sort(key=lambda c: c.start)
    return candidates


def select_current_period_amount(line: str, label_end: int = 0, min_value: int = 0, absolute: bool = False) -> int | None:
    candidates = [c for c in extract_amount_candidates(line) if c.start >= label_end and abs(c.value) >= min_value]
    if not candidates:
        return None

    # Remove a note reference such as 12 or (3) before the actual amount.
    if len(candidates) >= 2:
        first, second = candidates[0], candidates[1]
        if abs(first.value) < 100 and len(re.sub(r"\D", "", first.raw)) <= 2 and abs(second.value) >= 1000:
            candidates = candidates[1:]

    if not candidates:
        return None
    value = candidates[0].value
    return abs(value) if absolute else value
