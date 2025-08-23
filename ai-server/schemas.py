from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any, Optional

class NPCConfig(BaseModel):
    name: Optional[str] = None
    personality: Optional[str] = None
    backstory: Optional[str] = None
    dialogue_style: Optional[str] = None
    relationship: Optional[str] = None

class Context(BaseModel):
    player_status: Optional[Dict[str, Any]] = Field(default_factory=dict)
    game_state: Optional[Dict[str, Any]] = Field(default_factory=dict)
    npc_config: Optional[NPCConfig] = None

class AskReq(BaseModel):
    session_id: str
    npc_id: str
    user_input: str
    context: Optional[Context] = Field(default_factory=Context)

class AskRes(BaseModel):
    npc_response: str
    flags: Dict[str, Any] = Field(default_factory=dict)
    valid: bool
    meta: Dict[str, Any] = Field(default_factory=dict)