from typing import Dict, List
from active_inference_forager.config import MAX_CONVERSATION_HISTORY

class ConversationManager:
    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > MAX_CONVERSATION_HISTORY:
            self.conversation_history.pop(0)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history
