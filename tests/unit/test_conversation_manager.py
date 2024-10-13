import pytest
from active_inference_forager.conversation_manager import ConversationManager

@pytest.fixture
def conversation_manager():
    return ConversationManager()

class TestConversationManager:
    def test_add_message(self, conversation_manager):
        conversation_manager.add_message("user", "Hello")
        history = conversation_manager.get_conversation_history()
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
