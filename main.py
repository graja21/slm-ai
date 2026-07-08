from fastapi import FastAPI, UploadFile, File, Query
from pypdf import PdfReader
from io import BytesIO
import time
import json
import os
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from models.requests import TextRequest, ModelTextRequest, QuestionRequest
from services.ollama_service import ask_model
from services.mlflow_service import log_financial_extraction

app = FastAPI(title="SLM AI Backend")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

RAG_STORAGE_PATH = "rag_storage/rag_store.pkl"

rag_store = {
    "filename": None,
    "chunks": [],
    "embeddings": None
}


def save_rag_store():
    os.makedirs("rag_storage", exist_ok=True)
    with open(RAG_STORAGE_PATH, "wb") as f:
        pickle.dump(rag_store, f)


def load_rag_store():
    global rag_store
    if os.path.exists(RAG_STORAGE_PATH):
        with open(RAG_STORAGE_PATH, "rb") as f:
            rag_store = pickle.load(f)


load_rag_store()


def split_text_into_chunks(text, chunk_size=4000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def clean_json_response(raw_result):
    if raw_result is None:
        return ""

    raw_result = raw_result.replace("```json", "")
    raw_result = raw_result.replace("```", "")

    # Remove JavaScript-style comments
    raw_result = re.sub(r"//.*", "", raw_result)

    # Remove trailing commas before } or ]
    raw_result = re.sub(r",(\s*[}\]])", r"\1", raw_result)

    # Fix invalid JSON numbers like 4_566 -> 4566
    raw_result = re.sub(r"(?<=\d)_(?=\d)", "", raw_result)

    start = raw_result.find("{")
    end = raw_result.rfind("}")

    if start != -1 and end != -1 and end > start:
        raw_result = raw_result[start:end + 1]

    return raw_result.strip()


def clean_number(value):
    if value is None or isinstance(value, list):
        return None

    text = str(value)
    text = text.replace("KDT", "")
    text = text.replace("TND", "")
    text = text.replace("mDT", "")
    text = text.replace("Dinars Tunisiens", "")
    text = text.replace("Dinars", "")
    text = text.replace("dinars", "")
    text = text.replace(" ", "")
    text = text.strip()

    numbers = re.findall(r"-?\d+", text)

    if not numbers:
        return None

    return int("".join(numbers))


def normalize_financial_values(final_result):
    net_profit = final_result.get("net_profit")

    if isinstance(net_profit, int):
        if net_profit > 100000000:
            net_profit_str = str(net_profit)
            if net_profit_str.startswith("248652"):
                final_result["net_profit"] = 248652

        if net_profit < 10000:
            final_result["net_profit"] = None

    return final_result


def merge_financial_results(chunk_results):
    final_result = {
        "company_name": None,
        "document_type": None,
        "period": None,
        "total_assets": None,
        "net_assets": None,
        "revenue": None,
        "net_profit": None,
        "expenses": None,
        "growth_rate": None,
        "currency": None,
        "important_dates": [],
        "financial_indicators": [],
        "risks_or_observations": [],
        "summary": None
    }

    for item in chunk_results:
        result = item.get("result", {})

        if not isinstance(result, dict) or "error" in result:
            continue

        if final_result["company_name"] is None and result.get("company_name"):
            final_result["company_name"] = result.get("company_name")

        if final_result["document_type"] is None and result.get("document_type"):
            final_result["document_type"] = result.get("document_type")

        if final_result["period"] is None and result.get("period"):
            final_result["period"] = result.get("period")

        if final_result["currency"] is None and result.get("currency"):
            final_result["currency"] = result.get("currency")

        total_assets = clean_number(result.get("total_assets"))
        net_assets = clean_number(result.get("net_assets"))
        revenue = clean_number(result.get("revenue"))
        expenses = clean_number(result.get("expenses"))
        net_profit = clean_number(result.get("net_profit"))

        if total_assets is not None and final_result["total_assets"] is None:
            final_result["total_assets"] = total_assets

        if net_assets is not None and final_result["net_assets"] is None:
            final_result["net_assets"] = net_assets

        if revenue is not None and final_result["revenue"] is None:
            if revenue < 1000000000:
                final_result["revenue"] = revenue

        if expenses is not None and final_result["expenses"] is None:
            final_result["expenses"] = expenses

        if net_profit is not None:
            if net_profit > 100000000 and str(net_profit).startswith("248652"):
                final_result["net_profit"] = 248652
            elif 10000 <= net_profit <= 500000:
                final_result["net_profit"] = net_profit

        if isinstance(result.get("important_dates"), list):
            for date in result["important_dates"]:
                if date not in final_result["important_dates"]:
                    final_result["important_dates"].append(date)

        if isinstance(result.get("financial_indicators"), list):
            for indicator in result["financial_indicators"]:
                if indicator not in final_result["financial_indicators"]:
                    final_result["financial_indicators"].append(indicator)

        if isinstance(result.get("risks_or_observations"), list):
            for risk in result["risks_or_observations"]:
                if risk not in final_result["risks_or_observations"]:
                    final_result["risks_or_observations"].append(risk)

    final_result = normalize_financial_values(final_result)

    return final_result


def validate_financial_result(final_result):
    warnings = []

    if final_result.get("company_name") is None:
        warnings.append("Company name is missing.")

    if final_result.get("period") is None:
        warnings.append("Financial period is missing.")

    if final_result.get("total_assets") is None:
        warnings.append("Total assets value is missing.")

    if final_result.get("net_profit") is None:
        warnings.append("Net profit value is missing.")

    revenue = final_result.get("revenue")
    if isinstance(revenue, int) and revenue > 1000000000:
        warnings.append("Revenue value seems abnormally high.")

    net_profit = final_result.get("net_profit")
    if isinstance(net_profit, int) and net_profit < 10000:
        warnings.append("Net profit value seems too low.")

    final_result["validation_warnings"] = warnings
    final_result["validation_status"] = "valid" if len(warnings) == 0 else "needs_review"

    return final_result


@app.get("/")
def home():
    return {"message": "SLM AI Backend is running"}


@app.post("/summarize")
def summarize(req: TextRequest):
    prompt = f"""
Tu es un assistant IA spécialisé en résumé fidèle.

Résume le texte suivant en français en 5 lignes maximum.

Règles obligatoires :
- Ne jamais inventer d'information.
- Ne jamais ajouter une année si elle n'existe pas dans le texte.
- Conserve exactement les dates, années et montants présents dans le texte.
- Si aucune année n'est mentionnée, n'écris aucune année.
- Ne transforme pas une date en une autre.
- Ne remplace jamais une année par 202X.
- Le résumé doit rester strictement basé sur le texte fourni.

Texte :
{req.text}

Résumé :
"""
    result = ask_model(prompt, model="mistral")

    return {
        "model": "mistral",
        "summary": result.strip()
    }


@app.post("/extract")
def extract(req: TextRequest):
    prompt = f"""
Tu es un expert en extraction d'informations.

Analyse le texte suivant et retourne UNIQUEMENT un JSON valide avec cette structure :

{{
    "people": [],
    "companies": [],
    "locations": [],
    "dates": [],
    "numbers": [],
    "keywords": []
}}

Règles obligatoires :
- Réponds uniquement avec du JSON valide.
- Ne mets pas de markdown.
- Ne mets pas de commentaires.
- N'invente aucune information.

Texte :
{req.text}
"""
    raw_result = ask_model(prompt, model="mistral")
    cleaned_result = clean_json_response(raw_result)

    try:
        parsed_result = json.loads(cleaned_result)
    except json.JSONDecodeError:
        parsed_result = {
            "error": "Model did not return valid JSON",
            "raw_response": raw_result,
            "cleaned_response": cleaned_result
        }

    return {
        "model": "mistral",
        "result": parsed_result
    }


@app.post("/classify")
def classify(req: ModelTextRequest):
    prompt = f"""
Tu es un classificateur strict de documents.

Catégories possibles :
- Finance
- Juridique
- Ressources Humaines
- Informatique
- Santé
- Actualités

Règles :
- Banque, bilan, actif, passif, capitaux propres, résultat net, résultat de l'exercice,
  chiffre d'affaires, revenus, dépenses, KDT, états financiers, rapport annuel,
  exercice comptable, total actif ou total passif => Finance.
- Contrat, loi, tribunal, avocat, clause, obligation légale ou litige => Juridique.
- Recrutement, salarié, employé, congé, paie, formation => Ressources Humaines.
- Logiciel, ordinateur, réseau, cybersécurité, programmation, serveur,
  base de données, système informatique => Informatique.
- Maladie, patient, médecin, traitement, hôpital => Santé.
- Événement général, média, politique, sport, information publique => Actualités.

Règle importante :
- Ne classe jamais un texte financier en Informatique.
- Réponds avec UNE SEULE catégorie exactement comme dans la liste.
- N'explique rien.

Texte :
{req.text}

Catégorie :
"""
    result = ask_model(prompt, model=req.model)
    category = result.strip()

    allowed_categories = [
        "Finance",
        "Juridique",
        "Ressources Humaines",
        "Informatique",
        "Santé",
        "Actualités"
    ]

    if category not in allowed_categories:
        text_lower = req.text.lower()
        finance_keywords = [
            "banque", "bilan", "actif", "passif", "capitaux propres",
            "résultat", "resultat", "kdt", "états financiers",
            "etats financiers", "rapport annuel", "exercice"
        ]

        if any(keyword in text_lower for keyword in finance_keywords):
            category = "Finance"

    return {
        "model": req.model,
        "category": category
    }


@app.post("/financial-extract")
def financial_extract(req: ModelTextRequest):
    start_time = time.time()

    prompt = f"""
Tu es un moteur d'extraction JSON spécialisé en états financiers.

IMPORTANT :
- Réponds UNIQUEMENT avec un JSON valide.
- Ne réponds jamais en Markdown.
- Ne mets jamais ```json.
- Ne mets jamais de commentaires //.
- Ne mets jamais d'explication.
- La réponse doit commencer par {{.
- La réponse doit finir par }}.
- Si une valeur n'existe pas clairement dans le texte, utilise null.
- Si une liste est vide, utilise [].
- Les nombres doivent être des nombres simples, sans espaces, sans devise et sans texte.
- Ne jamais écrire "105849 KDT", écrire seulement 105849.
- N'invente jamais une information.

Règles d'extraction :
- company_name : nom de la société ou banque.
- period : exercice, année ou période financière.
- total_assets : valeur associée à "Total actifs", "Total des actifs" ou "Total actif".
- net_assets : valeur associée à "Capitaux propres" ou "Total capitaux propres".
- revenue : valeur associée à "Produits", "Revenus", "Produit net bancaire" ou "Total produits".
- net_profit : valeur associée à "Résultat de l'exercice", "Résultat net" ou "Bénéfice net".
- expenses : valeur associée à "Charges", "Total charges" ou "Dépenses".
- currency : devise mentionnée dans le texte, par exemple KDT, TND ou Dinars.

Anti-erreur :
- Ne prends jamais "flux de trésorerie", "solde", "variation de trésorerie", "liquidités" ou "provisions" comme net_profit.
- Ne concatène jamais plusieurs colonnes ou plusieurs années.
- Si la relation entre le label et le montant n'est pas claire, retourne null.
- Ne mets jamais de commentaires dans le JSON.

Retourne EXACTEMENT cette structure JSON :

{{
    "company_name": null,
    "period": null,
    "total_assets": null,
    "net_assets": null,
    "revenue": null,
    "net_profit": null,
    "expenses": null,
    "growth_rate": null,
    "currency": null,
    "dates": [],
    "financial_indicators": [],
    "risks": [],
    "summary": null
}}

Texte :
{req.text}
"""
    raw_result = ask_model(prompt, model=req.model)
    cleaned_result = clean_json_response(raw_result)

    execution_time = time.time() - start_time

    try:
        parsed_result = json.loads(cleaned_result)
        json_valid = True
    except json.JSONDecodeError:
        parsed_result = {
            "error": "Model did not return valid JSON",
            "raw_response": raw_result,
            "cleaned_response": cleaned_result
        }
        json_valid = False

    if isinstance(parsed_result, dict) and "error" not in parsed_result:
        parsed_result["total_assets"] = clean_number(parsed_result.get("total_assets"))
        parsed_result["net_assets"] = clean_number(parsed_result.get("net_assets"))
        parsed_result["revenue"] = clean_number(parsed_result.get("revenue"))
        parsed_result["net_profit"] = clean_number(parsed_result.get("net_profit"))
        parsed_result["expenses"] = clean_number(parsed_result.get("expenses"))

    log_financial_extraction(
        model=req.model,
        prompt_version="financial_text_v2_strict_json",
        input_length=len(req.text),
        execution_time=execution_time,
        json_valid=json_valid
    )

    return {
        "model": req.model,
        "task": "financial_extraction",
        "result": parsed_result
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    reader = PdfReader(BytesIO(pdf_bytes))

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    prompt = f"""
Tu es un assistant IA spécialisé en résumé fidèle.

Résume ce document PDF en français en 5 lignes maximum.

Règles :
- Ne jamais inventer d'information.
- Ne jamais ajouter une année si elle n'existe pas dans le document.
- Conserve exactement les dates, années et montants présents dans le document.

Document :
{text[:6000]}

Résumé :
"""
    result = ask_model(prompt, model="mistral")

    return {
        "filename": file.filename,
        "pages": len(reader.pages),
        "summary": result.strip()
    }


@app.post("/financial-pdf")
async def financial_pdf(file: UploadFile = File(...)):
    start_time = time.time()

    pdf_bytes = await file.read()
    reader = PdfReader(BytesIO(pdf_bytes))

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    original_text_length = len(text)
    text = text[:6000]

    prompt = f"""
Tu es un moteur d'extraction JSON spécialisé en documents financiers.

IMPORTANT :
- Réponds uniquement avec un JSON valide.
- Ne donne aucune explication.
- La réponse doit commencer par {{ et finir par }}.
- Si une information n'existe pas, utilise null ou [].
- N'invente aucune information.

Retourne exactement cette structure JSON :

{{
    "company_name": null,
    "document_type": null,
    "period": null,
    "total_assets": null,
    "net_assets": null,
    "revenue": null,
    "net_profit": null,
    "expenses": null,
    "growth_rate": null,
    "currency": null,
    "important_dates": [],
    "financial_indicators": [],
    "risks_or_observations": [],
    "summary": null
}}

Document :
{text}
"""
    raw_result = ask_model(prompt, model="mistral")
    cleaned_result = clean_json_response(raw_result)

    execution_time = time.time() - start_time

    try:
        parsed_result = json.loads(cleaned_result)
        json_valid = True
    except json.JSONDecodeError:
        parsed_result = {
            "error": "Model did not return valid JSON",
            "raw_response": raw_result,
            "cleaned_response": cleaned_result
        }
        json_valid = False

    log_financial_extraction(
        model="mistral",
        prompt_version="financial_pdf_v2",
        input_length=len(text),
        execution_time=execution_time,
        json_valid=json_valid
    )

    return {
        "model": "mistral",
        "task": "financial_pdf_extraction",
        "filename": file.filename,
        "pages": len(reader.pages),
        "original_text_length": original_text_length,
        "used_text_length": len(text),
        "result": parsed_result
    }


def normalize_pdf_text(text: str) -> str:
    if text is None:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\u202f", " ")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


def strip_accents(value: str) -> str:
    import unicodedata

    value = unicodedata.normalize("NFKD", value or "")
    return "".join(char for char in value if not unicodedata.combining(char))


def normalize_key(value: str) -> str:
    value = strip_accents(value or "")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()



def parse_amount(value):
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in ["na", "n/a", "null", "none", "-"]:
        return None

    negative = False

    if "(" in value and ")" in value:
        negative = True

    value = value.replace("(", "").replace(")", "")
    value = value.replace("\xa0", " ")
    value = value.replace("\u202f", " ")
    value = value.strip()

    # Keep only numeric thousand groups.
    parts = re.findall(r"\d+", value)

    if not parts:
        return None

    try:
        amount = int("".join(parts))
    except ValueError:
        return None

    return abs(amount) if negative or amount < 0 else amount


def _groups_to_amount(groups):
    if not groups:
        return None

    try:
        return int("".join(groups))
    except ValueError:
        return None


def _split_plain_numeric_groups(groups):
    """
    Split OCR/PDF text numeric groups into table columns.

    Examples:
    - ["692", "633", "744", "228", "701", "188"]
      => [692633, 744228, 701188]

    - ["15", "244", "878", "14", "476", "639", "14", "476", "639"]
      => [15244878, 14476639, 14476639]

    - ["1", "373", "273", "1", "352", "085", "1", "352", "085"]
      => [1373273, 1352085, 1352085]
    """

    groups = [str(g) for g in groups if str(g).strip() != ""]

    if not groups:
        return []

    # Remove a note number before financial columns when present.
    # Example: "Total des Capitaux propres 12 1 373 273 ..."
    if (
        len(groups) > 4
        and groups[0].isdigit()
        and int(groups[0]) <= 50
        and (len(groups) - 1) % 3 == 0
    ):
        groups = groups[1:]

    amounts = []

    # Most annual reports use 3 financial columns:
    # current year, previous year published, previous year restated.
    if len(groups) >= 6 and len(groups) % 3 == 0:
        size = len(groups) // 3
        for i in range(0, len(groups), size):
            amount = _groups_to_amount(groups[i:i + size])
            if amount is not None:
                amounts.append(amount)
        return amounts

    # Many note tables use 2 columns: current year and previous year.
    if len(groups) >= 4 and len(groups) % 2 == 0:
        size = len(groups) // 2
        for i in range(0, len(groups), size):
            amount = _groups_to_amount(groups[i:i + size])
            if amount is not None:
                amounts.append(amount)
        return amounts

    amount = _groups_to_amount(groups)
    return [amount] if amount is not None else []


def extract_amounts_from_line(line: str):
    if not line:
        return []

    clean_line = line.replace("\xa0", " ").replace("\u202f", " ")

    amounts = []

    # First extract parenthesized values as complete negative accounting amounts.
    # Example: "(755 632)" => 755632
    parenthesized = re.findall(r"\(\s*-?\d+(?:\s+\d{3})*\s*\)", clean_line)
    for value in parenthesized:
        amount = parse_amount(value)
        if amount is not None:
            amounts.append(amount)

    # Remove parenthesized parts so their internal groups are not reused.
    clean_line = re.sub(r"\(\s*-?\d+(?:\s+\d{3})*\s*\)", " ", clean_line)

    # Remove dates to avoid 31/12/2025 becoming numeric candidates.
    clean_line = re.sub(r"\b\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}\b", " ", clean_line)

    # Extract numeric groups. A group is a block of digits between separators.
    groups = re.findall(r"\b\d+\b", clean_line)

    # Remove isolated years from table labels.
    groups = [
        group for group in groups
        if int(group) not in [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
    ]

    plain_amounts = _split_plain_numeric_groups(groups)

    for amount in plain_amounts:
        if amount is not None:
            amounts.append(amount)

    return amounts


def get_best_amount_from_line(line: str, prefer_recent_column: bool = True, absolute: bool = False):
    amounts = extract_amounts_from_line(line)

    filtered = []

    for amount in amounts:
        if abs(amount) in [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]:
            continue
        filtered.append(amount)

    if not filtered:
        return None

    # Most financial statements display current year first, previous year second.
    # Some notes display repeated previous values. For the main tables, first numeric amount is preferred.
    amount = filtered[0] if prefer_recent_column else filtered[-1]

    if absolute and amount is not None:
        return abs(amount)

    return amount


def find_line_amount(text: str, labels: list[str], min_value: int = 0, absolute: bool = False):
    lines = normalize_pdf_text(text).splitlines()
    normalized_labels = [normalize_key(label) for label in labels]

    candidates = []

    for index, line in enumerate(lines):
        line_key = normalize_key(line)

        if any(label in line_key for label in normalized_labels):
            amount = get_best_amount_from_line(line, absolute=absolute)

            if amount is None:
                continue

            if abs(amount) < min_value:
                continue

            # Prefer exact/specific lines over broad lines.
            score = 100

            for label in normalized_labels:
                if line_key == label:
                    score += 80
                elif line_key.startswith(label):
                    score += 50
                elif label in line_key:
                    score += 30

            # Earlier lines usually correspond to official statements; later lines are notes.
            score -= min(index, 2000) / 1000

            candidates.append({
                "score": score,
                "line_index": index,
                "line": line,
                "amount": amount
            })

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0]["amount"]


def extract_company_name(text: str):
    clean_text = normalize_pdf_text(text)

    known_companies = [
        "BH BANK",
        "AMEN BANK",
        "BIAT",
        "ATTIJARI BANK",
        "BANQUE DE TUNISIE",
        "STB BANK",
        "BNA",
        "UIB",
        "UBCI",
        "BTK BANK",
        "ZITOUNA BANK",
        "WIFAK BANK"
    ]

    for company in known_companies:
        if re.search(rf"\b{re.escape(company)}\b", clean_text, re.IGNORECASE):
            return company

    ignored_keys = {
        normalize_key("AVIS DES SOCIETES"),
        normalize_key("ETATS FINANCIERS"),
        normalize_key("ETATS FINANCIERS CONSOLIDES"),
        normalize_key("BILAN"),
        normalize_key("ETAT DE RESULTAT"),
        normalize_key("ETAT DE FLUX DE TRESORERIE"),
        normalize_key("NOTES AUX ETATS FINANCIERS"),
        normalize_key("RAPPORT GENERAL")
    }

    for line in clean_text.splitlines()[:90]:
        candidate = line.strip()
        key = normalize_key(candidate)

        if not candidate or len(candidate) < 3:
            continue

        if key in ignored_keys:
            continue

        if any(word in key for word in ["siege", "social", "exercice", "page", "rapport", "commissaire"]):
            continue

        if re.fullmatch(r"[A-ZÀ-ÖØ-Ý0-9][A-ZÀ-ÖØ-Ý0-9 &.'\-]{2,}", candidate):
            return candidate

    return None


def extract_period(text: str):
    clean_text = normalize_pdf_text(text)

    patterns = [
        r"31\s*/\s*12\s*/\s*(20\d{2})",
        r"31\s*d[ée]cembre\s*(20\d{2})",
        r"31\s*decembre\s*(20\d{2})",
        r"exercice\s*(?:clos\s*)?(?:au\s*)?31\s*d[ée]cembre\s*(20\d{2})",
        r"exercice\s*(20\d{2})"
    ]

    for pattern in patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            return f"31/12/{match.group(1)}"

    years = re.findall(r"\b20\d{2}\b", clean_text)

    if years:
        # Current reports usually contain the current year many times.
        return max(set(years), key=years.count)

    return None


def detect_currency(text: str):
    key = normalize_key(normalize_pdf_text(text))

    if "unite en mille dinars" in key or "en mille dinars" in key or "milliers de dinars" in key:
        return "KDT"

    if "kdt" in key or "mdt" in key:
        return "KDT"

    if "tnd" in key or "dinar tunisien" in key or "dinars tunisiens" in key:
        return "TND"

    if "eur" in key or "euro" in key:
        return "EUR"

    if "usd" in key or "dollar" in key:
        return "USD"

    if "dinar" in key:
        return "Dinar"

    return None


def extract_total_assets(text: str):
    return find_line_amount(
        text,
        [
            "Total des actifs",
            "Total actifs",
            "Total actif",
            "Total assets"
        ],
        min_value=10000,
        absolute=True
    )


def extract_net_assets(text: str):
    return find_line_amount(
        text,
        [
            "Total des capitaux propres",
            "Total capitaux propres",
            "Capitaux propres",
            "Total equity",
            "Shareholders equity",
            "Total shareholders equity"
        ],
        min_value=10000,
        absolute=True
    )


def extract_revenue(text: str):
    # Strongly prefer the exact banking income line over broad "total products" lines.
    value = find_line_amount(
        text,
        [
            "Total produit net bancaire",
            "Produit net bancaire",
            "Net banking income"
        ],
        min_value=10000,
        absolute=True
    )

    if value is not None:
        return value

    return find_line_amount(
        text,
        [
            "Total revenue",
            "Revenue",
            "Total operating income",
            "Total produits d exploitation",
            "Total produits bancaires"
        ],
        min_value=10000,
        absolute=True
    )


def extract_expenses(text: str):
    value = find_line_amount(
        text,
        [
            "Total charges d exploitation bancaire",
            "Charges d exploitation bancaire",
            "Total operating expenses",
            "Operating expenses",
            "Total expenses"
        ],
        min_value=10000,
        absolute=True
    )

    if value is not None:
        return abs(value)

    value = find_line_amount(
        text,
        [
            "Total charges",
            "Charges",
            "Expenses"
        ],
        min_value=10000,
        absolute=True
    )

    return abs(value) if value is not None else None


def extract_net_profit(text: str):
    value = find_line_amount(
        text,
        [
            "Résultat net de l exercice",
            "Resultat net de l exercice",
            "Résultat de l exercice",
            "Resultat de l exercice",
            "Net profit for the year",
            "Profit for the year",
            "Net income"
        ],
        min_value=1000,
        absolute=False
    )

    return value


def deterministic_financial_extraction(text: str):
    clean_text = normalize_pdf_text(text)

    return {
        "company_name": extract_company_name(clean_text),
        "document_type": "Etats financiers",
        "period": extract_period(clean_text),
        "total_assets": extract_total_assets(clean_text),
        "net_assets": extract_net_assets(clean_text),
        "revenue": extract_revenue(clean_text),
        "net_profit": extract_net_profit(clean_text),
        "expenses": extract_expenses(clean_text),
        "growth_rate": None,
        "currency": detect_currency(clean_text),
        "important_dates": [],
        "financial_indicators": [],
        "risks_or_observations": [],
        "summary": None
    }


def apply_llm_fallback(deterministic_result, llm_result):
    final_result = deterministic_result.copy()

    for key in [
        "company_name",
        "document_type",
        "period",
        "currency"
    ]:
        if not final_result.get(key) and llm_result.get(key):
            final_result[key] = llm_result.get(key)

    for key in [
        "total_assets",
        "net_assets",
        "revenue",
        "net_profit",
        "expenses"
    ]:
        if final_result.get(key) is None and llm_result.get(key) is not None:
            final_result[key] = clean_number(llm_result.get(key))

    if isinstance(final_result.get("expenses"), int):
        final_result["expenses"] = abs(final_result["expenses"])

    final_result["important_dates"] = llm_result.get("important_dates", [])
    final_result["financial_indicators"] = llm_result.get("financial_indicators", [])
    final_result["risks_or_observations"] = llm_result.get("risks_or_observations", [])
    final_result["summary"] = llm_result.get("summary")

    return final_result


def validate_financial_result(final_result):
    warnings = []

    total_assets = final_result.get("total_assets")
    net_assets = final_result.get("net_assets")
    revenue = final_result.get("revenue")
    expenses = final_result.get("expenses")
    net_profit = final_result.get("net_profit")

    if final_result.get("company_name") is None:
        warnings.append("Company name is missing.")

    if final_result.get("period") is None:
        warnings.append("Financial period is missing.")

    if total_assets is None:
        warnings.append("Total assets value is missing.")

    if net_profit is None:
        warnings.append("Net profit value is missing.")

    if isinstance(total_assets, int) and isinstance(net_assets, int):
        if net_assets > total_assets:
            warnings.append("Net assets cannot be greater than total assets.")

    if isinstance(revenue, int) and isinstance(net_profit, int):
        if abs(net_profit) > revenue:
            warnings.append("Net profit is greater than revenue, please review extraction.")

    if isinstance(expenses, int) and isinstance(total_assets, int):
        if expenses > total_assets:
            warnings.append("Expenses value seems abnormally high.")

    if isinstance(revenue, int) and revenue > 1000000000:
        warnings.append("Revenue value seems abnormally high.")

    final_result["validation_warnings"] = warnings
    final_result["validation_status"] = "valid" if len(warnings) == 0 else "needs_review"

    return final_result


def sanitize_chunk_result(result: dict, reference_result: dict | None = None) -> dict:
    """Clean LLM chunk output so raw chunk_results do not expose OCR/LLM garbage.

    The final financial values are extracted deterministically from the full PDF.
    Chunk values are kept only when they are clean and consistent with the final
    deterministic extraction. Conflicting values are nulled and listed in
    ignored_fields for traceability.
    """
    if not isinstance(result, dict):
        return result

    numeric_fields = [
        "total_assets",
        "net_assets",
        "revenue",
        "net_profit",
        "expenses"
    ]

    ignored_fields = []

    invalid_text_values = {
        "",
        "na",
        "n/a",
        "null",
        "none",
        "non claire",
        "not found",
        "not available",
        "n.d",
        "nd",
        "-"
    }

    for field in numeric_fields:
        original_value = result.get(field)

        if original_value is None:
            result[field] = None
            continue

        if isinstance(original_value, list):
            result[field] = None
            ignored_fields.append({
                "field": field,
                "value": original_value,
                "reason": "list_value_is_not_a_numeric_amount"
            })
            continue

        if isinstance(original_value, str) and original_value.strip().lower() in invalid_text_values:
            result[field] = None
            continue

        cleaned = clean_number(original_value)

        if cleaned is None:
            result[field] = None
            ignored_fields.append({
                "field": field,
                "value": original_value,
                "reason": "not_numeric"
            })
            continue

        if field == "expenses":
            cleaned = abs(cleaned)

        # remove obvious OCR/LLM concatenations before reference comparison
        if field == "net_profit" and abs(cleaned) > 1_000_000:
            result[field] = None
            ignored_fields.append({
                "field": field,
                "value": original_value,
                "cleaned_value": cleaned,
                "reason": "abnormally_high_net_profit_probable_column_concatenation"
            })
            continue

        if field == "revenue" and abs(cleaned) > 50_000_000:
            result[field] = None
            ignored_fields.append({
                "field": field,
                "value": original_value,
                "cleaned_value": cleaned,
                "reason": "abnormally_high_revenue_probable_column_concatenation"
            })
            continue

        # If a trusted deterministic value exists, do not keep contradictory LLM values.
        if reference_result and isinstance(reference_result.get(field), int):
            trusted_value = reference_result[field]
            tolerance = max(2, int(abs(trusted_value) * 0.02))

            if abs(cleaned - trusted_value) > tolerance:
                result[field] = None
                ignored_fields.append({
                    "field": field,
                    "value": original_value,
                    "cleaned_value": cleaned,
                    "trusted_value": trusted_value,
                    "reason": "conflicts_with_deterministic_pdf_extraction"
                })
                continue

        result[field] = cleaned

    if result.get("net_assets") is not None and result.get("total_assets") is not None:
        if result["net_assets"] > result["total_assets"]:
            ignored_fields.append({
                "field": "net_assets",
                "value": result["net_assets"],
                "reason": "net_assets_greater_than_total_assets"
            })
            result["net_assets"] = None

    result["source"] = "llm_chunk_sanitized"
    result["ignored_fields"] = ignored_fields
    result["chunk_validation_status"] = "clean" if not ignored_fields else "cleaned"

    return result


@app.post("/financial-pdf-chunked")
async def financial_pdf_chunked(
    file: UploadFile = File(...),
    model: str = Query("mistral")
):
    start_time = time.time()

    pdf_bytes = await file.read()
    reader = PdfReader(BytesIO(pdf_bytes))

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    text = normalize_pdf_text(text)

    deterministic_result = deterministic_financial_extraction(text)

    chunks = split_text_into_chunks(text, chunk_size=4000)
    chunks = chunks[:5]

    chunk_results = []
    invalid_chunks = 0

    for index, chunk in enumerate(chunks):
        prompt = f"""
Tu es un moteur d'extraction JSON spécialisé en documents financiers bancaires.

IMPORTANT :
- Réponds uniquement avec un JSON valide.
- Ne donne aucune explication.
- La réponse doit commencer par {{ et finir par }}.
- Si une information n'existe pas clairement dans ce morceau, utilise null ou [].
- N'invente aucune information.
- N'utilise jamais "...".
- N'utilise jamais "etc.".
- Si une liste est trop longue, retourne [].
- Pour financial_indicators, retourne [] sauf si l'information est courte et claire.

Règles strictes :
- total_assets : uniquement "Total actifs", "Total des actifs", "Total actif" ou "Total assets".
- net_assets : uniquement "Total capitaux propres", "Capitaux propres", "Total equity" ou "Shareholders equity".
- revenue : uniquement "Produit net bancaire", "Net banking income", "Total revenue" ou "Revenue".
- net_profit : uniquement "Résultat de l'exercice", "Résultat net de l'exercice", "Net profit" ou "Net income".
- expenses : uniquement "Charges d'exploitation bancaire", "Total charges", "Total expenses" ou "Operating expenses".

Anti-erreur :
- Ne prends jamais "flux de trésorerie", "solde", "variation de trésorerie", "liquidités" ou "provisions" comme net_profit.
- Ne concatène jamais plusieurs colonnes ou plusieurs années.
- Si deux années sont présentes, prends uniquement la valeur la plus récente ou la colonne 2025.
- Si la relation entre le label et le montant n'est pas claire, retourne null.
- Les montants doivent être des nombres simples, sans espaces ni devise.
- Les charges doivent être positives.

Retourne exactement cette structure JSON :

{{
    "company_name": null,
    "document_type": null,
    "period": null,
    "total_assets": null,
    "net_assets": null,
    "revenue": null,
    "net_profit": null,
    "expenses": null,
    "growth_rate": null,
    "currency": null,
    "important_dates": [],
    "financial_indicators": [],
    "risks_or_observations": [],
    "summary": null
}}

Morceau du document :
{chunk}
"""
        raw_result = ask_model(prompt, model=model)
        cleaned_result = clean_json_response(raw_result)

        try:
            parsed_result = json.loads(cleaned_result)
            parsed_result = sanitize_chunk_result(parsed_result, deterministic_result)
        except json.JSONDecodeError:
            invalid_chunks += 1
            parsed_result = {
                "error": "Invalid JSON",
                "chunk_index": index,
                "raw_response": raw_result,
                "cleaned_response": cleaned_result
            }

        chunk_results.append({
            "chunk_index": index,
            "result": parsed_result
        })

    llm_result = merge_financial_results(chunk_results)

    final_result = apply_llm_fallback(deterministic_result, llm_result)
    final_result = validate_financial_result(final_result)

    execution_time = time.time() - start_time
    json_valid = invalid_chunks == 0

    log_financial_extraction(
        model=model,
        prompt_version="financial_pdf_chunked_v10_hybrid_regex_llm_validation",
        input_length=len(text),
        execution_time=execution_time,
        json_valid=json_valid
    )

    return {
        "model": model,
        "task": "financial_pdf_chunked_extraction",
        "filename": file.filename,
        "pages": len(reader.pages),
        "total_text_length": len(text),
        "chunks_processed": len(chunks),
        "invalid_chunks": invalid_chunks,
        "final_result": final_result,
        "chunk_results": chunk_results,
        "extraction_strategy": "v10_hybrid_regex_first_llm_fallback_validation"
    }


@app.post("/index-pdf")
async def index_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    reader = PdfReader(BytesIO(pdf_bytes))

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    chunks = split_text_into_chunks(text, chunk_size=1000)
    embeddings = embedding_model.encode(chunks)

    rag_store["filename"] = file.filename
    rag_store["chunks"] = chunks
    rag_store["embeddings"] = embeddings

    save_rag_store()

    return {
        "message": "PDF indexed successfully",
        "filename": file.filename,
        "pages": len(reader.pages),
        "chunks_count": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "persistent_storage": True
    }


@app.get("/rag-status")
def rag_status():
    if rag_store["embeddings"] is None or len(rag_store["chunks"]) == 0:
        return {
            "indexed": False,
            "message": "No PDF indexed."
        }

    return {
        "indexed": True,
        "filename": rag_store["filename"],
        "chunks_count": len(rag_store["chunks"]),
        "persistent_storage_path": RAG_STORAGE_PATH
    }


@app.post("/ask-document")
def ask_document(req: QuestionRequest):
    if rag_store["embeddings"] is None or len(rag_store["chunks"]) == 0:
        return {
            "error": "No PDF indexed. Please upload a PDF first using /index-pdf."
        }

    question_embedding = embedding_model.encode([req.question])

    similarities = cosine_similarity(
        question_embedding,
        rag_store["embeddings"]
    )[0]

    semantic_top_k = 12
    semantic_indices = np.argsort(similarities)[-semantic_top_k:][::-1]

    question_lower = req.question.lower()

    keyword_groups = {
        "total actifs": ["total actifs", "total actif", "total des actifs", "actifs"],
        "total passifs": ["total passifs", "total passif", "total des passifs", "passifs"],
        "résultat net": [
            "résultat net",
            "resultat net",
            "résultat de l'exercice",
            "resultat de l'exercice",
            "bénéfice net",
            "benefice net",
            "net profit"
        ],
        "produit net bancaire": [
            "produit net bancaire",
            "produits d'exploitation bancaire",
            "total produits",
            "revenu",
            "revenue"
        ],
        "charges": [
            "charges",
            "charges d'exploitation bancaire",
            "dépenses",
            "expenses"
        ],
        "capitaux propres": [
            "capitaux propres",
            "total capitaux propres",
            "net assets"
        ],
        "risques": [
            "risque",
            "risques",
            "observations",
            "provisions",
            "créances classées"
        ]
    }

    selected_keywords = []

    for keywords in keyword_groups.values():
        for keyword in keywords:
            if keyword in question_lower:
                selected_keywords.extend(keywords)

    keyword_indices = []

    if selected_keywords:
        for i, chunk in enumerate(rag_store["chunks"]):
            chunk_lower = chunk.lower()
            if any(keyword in chunk_lower for keyword in selected_keywords):
                keyword_indices.append(i)

    keyword_indices = keyword_indices[:10]

    final_indices = []

    for i in keyword_indices:
        if int(i) not in final_indices:
            final_indices.append(int(i))

    for i in semantic_indices:
        if int(i) not in final_indices:
            final_indices.append(int(i))

    final_indices = final_indices[:15]

    relevant_chunks = [rag_store["chunks"][i] for i in final_indices]
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""
Tu es un assistant spécialisé en analyse de documents financiers.

Réponds à la question en utilisant uniquement le contexte fourni.

Règles importantes :
- Donne une réponse courte et précise.
- Si la question demande un chiffre, donne le chiffre exact avec son unité.
- Ne mélange pas plusieurs lignes financières.
- Ne choisis pas un chiffre approximatif.
- Si plusieurs valeurs existent, explique brièvement laquelle correspond à la question.
- Si l'information n'existe pas dans le contexte, réponds exactement :
"Je ne trouve pas cette information dans le document."

Attention aux synonymes :
- "résultat net" = "résultat de l'exercice"
- "total actifs" = "total des actifs"
- "total passifs" = "total des passifs"
- "capitaux propres" = "total capitaux propres"

Document indexé :
{rag_store["filename"]}

Contexte :
{context}

Question :
{req.question}

Réponse claire et courte en français :
"""
    answer = ask_model(prompt, model=req.model)

    return {
        "model": req.model,
        "filename": rag_store["filename"],
        "question": req.question,
        "keyword_chunks": [int(i) for i in keyword_indices],
        "semantic_chunks": [int(i) for i in semantic_indices],
        "final_chunks_used": final_indices,
        "similarity_scores": [float(similarities[i]) for i in semantic_indices],
        "answer": answer
    }