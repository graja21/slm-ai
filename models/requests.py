from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class ModelTextRequest(BaseModel):
    text: str
    model: str = "mistral"