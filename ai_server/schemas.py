from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class NPCConfig(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    persona_name: Optional[str] = None
    dialogue_style: Optional[str] = None
    relationship: Optional[float] = None
    npc_mood: Optional[str] = None
    trigger_values: Optional[Dict[str, List[str]]] = None
    trigger_definitions: Optional[Dict[str, Dict[str, Any]]] = None

class DialogueTurn(BaseModel):
    player: str
    npc: str

class Context(BaseModel):
    player_status: Optional[Dict[str, Any]] = Field(default_factory=dict)
    game_state: Optional[Dict[str, Any]] = Field(default_factory=dict)
    npc_config: Optional[NPCConfig] = None
    dialogue_history: Optional[List[DialogueTurn]] = Field(default_factory=list)

class AskReq(BaseModel):
    session_id: str
    npc_id: str
    user_input: str
    context: Optional[Context] = Field(default_factory=Context)

class FlagItem(BaseModel):
    score: float
    value: Optional[str] = None

class AskRes(BaseModel):
    npc_response: str
    flags: Dict[str, FlagItem] = Field(default_factory=dict)
    valid: bool
    meta: Dict[str, Any] = Field(default_factory=dict)
