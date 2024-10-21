from active_inference_forager.models.generative_model import GenerativeModel

class SimpleGenerativeModel(GenerativeModel):
    def __init__(self):
        super().__init__()
        self.prior_beliefs = {"intent": None, "emotion": None}

    def predict(self, observations):
        # Return predictions based on current beliefs
        return self.posterior_beliefs

    def update_beliefs(self, observations):
        # Update beliefs based on observations
        if "intent" in observations:
            self.posterior_beliefs["intent"] = observations["intent"]
        if "emotion" in observations:
            self.posterior_beliefs["emotion"] = observations["emotion"]
