from __future__ import annotations
import time
from io import BytesIO
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pypdf import PdfReader
from financial_engine import FinancialExtractor
router=APIRouter(tags=["Financial analysis"])
extractor=FinancialExtractor()
@router.post("/financial-pdf-chunked")
async def financial_pdf_chunked(file:UploadFile=File(...), model:str=Query(default="mistral")):
    if not file.filename or not file.filename.lower().endswith(".pdf"): raise HTTPException(status_code=400,detail="Only PDF files are supported.")
    started=time.perf_counter(); data=await file.read()
    if not data: raise HTTPException(status_code=400,detail="The uploaded PDF is empty.")
    try: reader=PdfReader(BytesIO(data))
    except Exception as exc: raise HTTPException(status_code=400,detail=f"Unable to read PDF: {exc}") from exc
    page_texts=[page.extract_text() or "" for page in reader.pages]; full_text="\n".join(page_texts)
    if not full_text.strip(): raise HTTPException(status_code=422,detail="No extractable text found. OCR fallback is required.")
    result=extractor.extract(full_text=full_text,page_texts=page_texts,filename=file.filename)
    return {"model":model,"task":"financial_pdf_profiled_extraction","filename":file.filename,"pages":len(reader.pages),"total_text_length":len(full_text),"chunks_processed":0,"invalid_chunks":0,"final_result":result.dict(),"chunk_results":[],"execution_time_seconds":round(time.perf_counter()-started,3),"extraction_strategy":"v20_3_bank_insurance_regression_fixed"}
