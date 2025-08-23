class ContextParser:
    def __init__(self, context: dict):
        self.player = context.get("player_status", {})
        self.game = context.get("game_state", {})
        self.npc = context.get("npc_config", {})
        self.history = context.get("dialogue_history", [])

    def get_prompt_vars(self):
        return {
            "persona": self.npc.get("persona_name", self.npc.get("name")),
            "style": self.npc.get("dialogue_style"),
            "relationship": self.npc.get("relationship"),
            "location": self.game.get("location"),
            "quest": self.game.get("current_quest"),
            "stage": self.game.get("quest_stage"),
            "player_level": self.player.get("level"),
            "player_reputation": self.player.get("reputation")
        }

    def get_rag_query(self, user_input: str) -> str:
        return f"{self.npc.get('persona_name')} {self.game.get('current_quest')} {user_input}"

    def get_filters(self) -> dict:
        return {
            "npc_id": self.npc.get("id"),
            "quest_stage": self.game.get("quest_stage")
        }

    def get_dialogue_history(self, max_turns: int = 3) -> str:
        history = self.history[-max_turns:]
        return "\n".join([f"Player: {h['player']}\nNPC: {h['npc']}" for h in history])