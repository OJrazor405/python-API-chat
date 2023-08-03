from typing import List
from pydantic import BaseModel

class PDFRequest(BaseModel):
    namespace: str
    index_name: str
    url: str

class PromptRequest(BaseModel):
    namespace: str
    index_name: str
    prompt: str
