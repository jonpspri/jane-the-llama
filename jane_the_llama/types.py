from pydantic import BaseModel, Field
from typing import Optional
from uuid import uuid4

from llama_index.core.base.llms.types import ChatMessage

class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    chat_history: Optional[list[ChatMessage]] = []

