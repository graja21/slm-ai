from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from io import BytesIO
import time
import json

from models.requests import TextRequest, ModelTextRequest
from services.ollama_service import ask_model
from services.mlflow_service import log_financial_extraction

app = FastAPI(title="SLM AI Backend")


def split_text_into_chunks(text, chunk_size=4000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def clean_json_response(raw_result):
    raw_result = raw_result.replace("```json", "")
    raw_result = raw_result.replace("```", "")
    return raw_result.strip()


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

    simple_fields = [
        "company_name",
        "document_type",
        "period",
        "total_assets",
        "net_assets",
        "revenue",
        "net_profit",
        "expenses",
        "growth_rate",
        "currency",
        "summary"
    ]

    for item in chunk_results:
        result = item.get("result", {})

        if not isinstance(result, dict):
            continue

        if "error" in result:
            continue

        for field in simple_fields:
            value = result.get(field)

            if final_result[field] in [None, "", []] and value not in [None, "", []]:
                final_result[field] = value

        if isinstance(result.get("important_dates"), list):
            for date in result["important_dates"]:
                if date not in final_result["important_dates"]:
                    final_result["important_dates"].append(date)

        if isinstance(result.get("financial_indicators"), list):
            final_result["financial_indicators"].extend(result["financial_indicators"])

        if isinstance(result.get("risks_or_observations"), list):
            final_result["risks_or_observations"].extend(result["risks_or_observations"])

    return final_result


def validate_financial_result(final_result):
    warnings = []

    revenue = final_result.get("revenue")

    try:
        if revenue is not None:
            revenue_number = int(str(revenue).replace(" ", "").replace(",", ""))

            if revenue_number > 1000000000:
                warnings.append(
                    "Revenue value seems abnormally high. It may be caused by table number concatenation."
                )
    except Exception:
        warnings.append("Revenue value could not be converted to a number.")

    if final_result.get("company_name") is None:
        warnings.append("Company name is missing.")

    if final_result.get("period") is None:
        warnings.append("Financial period is missing.")

    if final_result.get("total_assets") is None:
        warnings.append("Total assets value is missing.")

    if final_result.get("net_profit") is None:
        warnings.append("Net profit value is missing.")

    final_result["validation_warnings"] = warnings
    final_result["validation_status"] = "valid" if len(warnings) == 0 else "needs_review"

    return final_result


@app.get("/")
def home():
    return {
        "message": "SLM AI Backend is running"
    }


@app.post("/classify")
def classify(req: ModelTextRequest):
    prompt = f"""
Tu es un expert en classification de documents.

Choisis UNE SEULE catégorie parmi :
- Finance
- Juridique
- Ressources Humaines
- Informatique
- Santé
- Actualités

Réponds uniquement avec le nom de la catégorie.

Texte :
{req.text}
"""

    result = ask_model(prompt, model=req.model)

    return {
        "model": req.model,
        "category": result.strip()
    }


@app.post("/financial-extract")
def financial_extract(req: ModelTextRequest):
    start_time = time.time()

    prompt = f"""
Tu es un expert en analyse financière.

Retourne UNIQUEMENT un JSON valide avec cette structure :

{{
    "company_name": null,
    "period": null,
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

N'invente aucune information.

Texte :
{req.text}
"""

    raw_result = ask_model(prompt, model=req.model)
    raw_result = clean_json_response(raw_result)

    execution_time = time.time() - start_time

    try:
        parsed_result = json.loads(raw_result)
        json_valid = True
    except json.JSONDecodeError:
        parsed_result = {
            "error": "Model did not return valid JSON",
            "raw_response": raw_result
        }
        json_valid = False

    log_financial_extraction(
        model=req.model,
        prompt_version="v1",
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
Résume ce document PDF en français en 5 lignes maximum.

Document :
{text[:6000]}
"""

    result = ask_model(prompt, model="mistral")

    return {
        "filename": file.filename,
        "pages": len(reader.pages),
        "summary": result
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
    raw_result = clean_json_response(raw_result)

    execution_time = time.time() - start_time

    try:
        parsed_result = json.loads(raw_result)
        json_valid = True
    except json.JSONDecodeError:
        parsed_result = {
            "error": "Model did not return valid JSON",
            "raw_response": raw_result
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


@app.post("/financial-pdf-chunked")
async def financial_pdf_chunked(file: UploadFile = File(...)):
    start_time = time.time()

    pdf_bytes = await file.read()
    reader = PdfReader(BytesIO(pdf_bytes))

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    chunks = split_text_into_chunks(text, chunk_size=4000)

    # Limited for speed during demo/testing
    chunks = chunks[:5]

    chunk_results = []
    invalid_chunks = 0

    for index, chunk in enumerate(chunks):
        prompt = f"""
Tu es un moteur d'extraction JSON spécialisé en documents financiers.

IMPORTANT :
- Réponds uniquement avec un JSON valide.
- Ne donne aucune explication.
- La réponse doit commencer par {{ et finir par }}.
- Si une information n'existe pas dans ce morceau, utilise null ou [].
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

Morceau du document :
{chunk}
"""

        raw_result = ask_model(prompt, model="mistral")
        raw_result = clean_json_response(raw_result)

        try:
            parsed_result = json.loads(raw_result)
        except json.JSONDecodeError:
            invalid_chunks += 1
            parsed_result = {
                "error": "Invalid JSON",
                "chunk_index": index,
                "raw_response": raw_result
            }

        chunk_results.append({
            "chunk_index": index,
            "result": parsed_result
        })

    final_result = merge_financial_results(chunk_results)
    final_result = validate_financial_result(final_result)

    execution_time = time.time() - start_time
    json_valid = invalid_chunks == 0

    log_financial_extraction(
        model="mistral",
        prompt_version="financial_pdf_chunked_v3_validated",
        input_length=len(text),
        execution_time=execution_time,
        json_valid=json_valid
    )

    return {
        "model": "mistral",
        "task": "financial_pdf_chunked_extraction",
        "filename": file.filename,
        "pages": len(reader.pages),
        "total_text_length": len(text),
        "chunks_processed": len(chunks),
        "invalid_chunks": invalid_chunks,
        "final_result": final_result,
        "chunk_results": chunk_results
    }