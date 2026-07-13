import time
from fastapi import APIRouter, File, Query, UploadFile

from financial_engine.extractor import extract_financial_data
from financial_engine.pdf_reader import extract_pdf_text

router = APIRouter(tags=["Financial AI"])


@router.post("/financial-pdf-chunked")
async def financial_pdf_chunked(
    file: UploadFile = File(...),
    model: str = Query("mistral"),
):
    start = time.time()
    pdf_bytes = await file.read()
    full_text, first_pages_text, page_count = extract_pdf_text(pdf_bytes)
    final_result = extract_financial_data(full_text, first_pages_text, file.filename)

    # This v18 endpoint deliberately keeps numeric extraction deterministic.
    # LLM narrative enrichment can be reintroduced later without allowing it to overwrite numbers.
    return {
        "model": model,
        "task": "financial_pdf_chunked_extraction",
        "filename": file.filename,
        "pages": page_count,
        "total_text_length": len(full_text),
        "chunks_processed": 0,
        "invalid_chunks": 0,
        "final_result": final_result,
        "chunk_results": [],
        "execution_time_seconds": round(time.time() - start, 3),
        "extraction_strategy": "v18_modular_deterministic_financial_engine",
    }
