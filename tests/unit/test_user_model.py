import pytest
from active_inference_forager.user_model import UserModel


@pytest.fixture
def user_model():
    return UserModel()


class TestUserModel:
    def test_update_preference(self, user_model):
        user_model.update_preference("test_topic", 0.7)
        assert user_model.get_preferences()["test_topic"] == 0.7

    def test_add_interaction(self, user_model):
        user_model.add_interaction("user input", "ai response", "test_topic")
        history = user_model.get_interaction_history()
        assert len(history) == 1
        assert history[0]["topic"] == "test_topic"
