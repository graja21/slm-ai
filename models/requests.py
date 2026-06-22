from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class ModelTextRequest(BaseModel):
    text: str
    model: str = "mistral"


class QuestionRequest(BaseModel):
    question: str
    model: str = "mistral"