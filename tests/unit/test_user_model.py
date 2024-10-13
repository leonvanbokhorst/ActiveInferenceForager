import pytest
from active_inference_forager.user_model import UserModel

@pytest.fixture
def user_model():
    return UserModel()

class TestUserModel:
    def test_update_preference(self, user_model):
        user_model.update_preference("topic1", 0.7)
        assert user_model.get_preferences()["topic1"] == 0.7
