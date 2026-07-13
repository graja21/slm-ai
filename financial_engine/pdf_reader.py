from io import BytesIO
from pypdf import PdfReader


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, str, int]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    full_text = "\n".join(pages)
    first_pages_text = "\n".join(pages[:2])
    return full_text, first_pages_text, len(reader.pages)
