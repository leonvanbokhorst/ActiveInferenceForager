from active_inference_forager.models.inference_engine import InferenceEngine

class SimpleInferenceEngine(InferenceEngine):
    def __init__(self, generative_model):
        super().__init__(generative_model)

    def infer(self, observations):
        self.generative_model.update_beliefs(observations)
        return self.generative_model.posterior_beliefs

    def choose_action(self, beliefs):
        # Decide action based on updated beliefs
        if beliefs.get("emotion") == "frustrated":
            return "empathetic_response"
        elif beliefs.get("intent") == "reset_router":
            return "provide_router_reset_instructions"
        else:
            return "general_assistance"
