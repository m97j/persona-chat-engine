from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any, Optional

class HistoryItem(BaseModel):
    role: Literal["user", "npc"]
    text: str

class AskReq(BaseModel):
    session_id: str
    npc_id: str
    user_input: str
    persona_tag: Optional[str] = None
    short_history: List[HistoryItem] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    gen: Dict[str, Any] = Field(default_factory=dict)

class AskRes(BaseModel):
    npc_response: str
    flags: Dict[str, Any]
    valid: bool
    meta: Dict[str, Any] = Field(default_factory=dict)