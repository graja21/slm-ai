# SLM Financial Engine v19

Profile-based deterministic extraction for banks, insurers and generic companies.

Copy `financial_engine/` and `financial_router.py` into the FastAPI project root.

In the existing `main.py`:

```python
from financial_router import router as financial_router

app = FastAPI(title="SLM AI Backend")
app.include_router(financial_router)
```

Remove the old duplicate `/financial-pdf-chunked` endpoint.

Run:

```bat
cd C:\Users\Amine\Desktop\slm-stage\ai-backend
venv\Scripts\activate
python -m uvicorn main:app --reload
```

The API response must contain:

```json
"extraction_strategy": "v19_profiled_financial_engine"
```
