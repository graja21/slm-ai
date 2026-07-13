def validate_result(result: dict) -> dict:
    warnings: list[str] = []
    ta = result.get("total_assets")
    eq = result.get("net_assets")
    rev = result.get("revenue")
    prof = result.get("net_profit")
    exp = result.get("expenses")

    if not result.get("company_name"):
        warnings.append("Company name is missing.")
    if not result.get("period"):
        warnings.append("Financial period is missing.")
    if ta is None:
        warnings.append("Total assets value is missing.")
    if prof is None:
        warnings.append("Net profit value is missing.")
    if isinstance(ta, int) and isinstance(eq, int) and eq > ta:
        warnings.append("Net assets cannot be greater than total assets.")
    if isinstance(rev, int) and isinstance(prof, int) and abs(prof) > abs(rev):
        warnings.append("Net profit is greater than revenue; review extraction.")
    if isinstance(ta, int) and isinstance(exp, int) and abs(exp) > abs(ta):
        warnings.append("Expenses are greater than total assets; review extraction.")

    result["validation_warnings"] = warnings
    result["validation_status"] = "valid" if not warnings else "needs_review"
    return result
