class SimpleInferenceEngine:
    def __init__(self, generative_model):
        self.generative_model = generative_model
        self.belief_state = {"reset_router": 0.5, "unknown": 0.5}
        self.memory = []  # Log of previous interactions

    def infer(self, observations):
        self.memory.append(observations)
        self.update_beliefs(observations)
        return self.belief_state

    def update_beliefs(self, observations):
        if observations.get("emotion") == "frustrated":
            self.belief_state["reset_router"] += 0.1
            self.belief_state["unknown"] = 1 - self.belief_state["reset_router"]
        else:
            self.belief_state["unknown"] += 0.1
            self.belief_state["reset_router"] = 1 - self.belief_state["unknown"]

    def choose_action(self, beliefs):
        if beliefs["reset_router"] > beliefs["unknown"]:
            return "provide_router_reset_instructions"
        else:
            return "ask_for_clarification"
