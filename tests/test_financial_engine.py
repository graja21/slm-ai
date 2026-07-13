from pathlib import Path
from financial_engine.extractor import extract_financial_data
from financial_engine.pdf_reader import extract_pdf_text


def run(pdf_path: str):
    data = Path(pdf_path).read_bytes()
    full, first, _ = extract_pdf_text(data)
    return extract_financial_data(full, first, Path(pdf_path).name)


def test_amen():
    result = run("/mnt/data/amen_bank_efd311225(5).pdf")
    assert result["company_name"] == "AMEN BANK"
    assert result["total_assets"] == 12569041
    assert result["net_assets"] == 1707363
    assert result["revenue"] == 590069
    assert result["expenses"] == 644693
    assert result["net_profit"] == 248652


def test_bna():
    result = run("/mnt/data/bna_bank_efd311225.pdf")
    assert result["company_name"] == "BANQUE NATIONALE AGRICOLE - BNA BANK"
    assert result["total_assets"] == 25039508
    assert result["net_assets"] == 2372903
    assert result["revenue"] == 1087093
    assert result["expenses"] == 1410604
    assert result["net_profit"] == 274544
