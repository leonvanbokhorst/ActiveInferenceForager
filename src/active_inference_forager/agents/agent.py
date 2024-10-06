class Agent:
    def __init__(self):
        self.belief = 0.5

    def get_belief(self):
        return self.belief

    def update_belief(self, observation):
        # Simple update for demonstration
        self.belief = (self.belief + observation["observation"]) / 2
