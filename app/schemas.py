from pydantic import BaseModel
from typing import List, Optional


class UploadResponse(BaseModel):
    message: str
    chunks_indexed: int


class AskRequest(BaseModel):
    question: str


class SourceSnippet(BaseModel):
    page: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceSnippet]