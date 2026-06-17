from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from io import BytesIO
from models.requests import TextRequest
from services.ollama_service import ask_model
import json

app = FastAPI(title="SLM AI Backend")


@app.get("/")
def home():
    return {
        "message": "SLM AI Backend is running"
    }


@app.post("/summarize")
def summarize(req: TextRequest):
    prompt = f"""
Tu es un assistant IA spécialisé en résumé.
Résume le texte suivant en français en 5 lignes maximum.

Texte :
{req.text}
"""

    result = ask_model(prompt)

    return {
        "model": "mistral",
        "summary": result
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

Texte :
{req.text}
"""

    raw_result = ask_model(prompt)

    try:
        parsed_result = json.loads(raw_result)
    except json.JSONDecodeError:
        parsed_result = {
            "error": "Model did not return valid JSON",
            "raw_response": raw_result
        }

    return {
        "model": "mistral",
        "result": parsed_result
    }


@app.post("/classify")
def classify(req: TextRequest):
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

    result = ask_model(prompt)

    return {
        "category": result.strip()
    }


@app.post("/upload-txt")
async def upload_txt(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    prompt = f"""
Résume ce document en français en 5 lignes maximum.

Document :
{text}
"""

    result = ask_model(prompt)

    return {
        "filename": file.filename,
        "summary": result
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
{text}
"""

    result = ask_model(prompt)

    return {
        "filename": file.filename,
        "pages": len(reader.pages),
        "summary": result
    }