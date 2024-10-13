import pytest
from active_inference_forager.user_model import UserModel
from active_inference_forager.fep_predictor import FEPPredictor

@pytest.fixture
def user_model():
    return UserModel()

@pytest.fixture
def fep_predictor(user_model):
    return FEPPredictor(user_model)

class TestFEPPredictor:
    def test_predict_next_topic(self, user_model, fep_predictor):
        user_model.update_preference("topic1", 0.7)
        user_model.update_preference("topic2", 0.3)
        assert fep_predictor.predict_next_topic() == "topic1"
