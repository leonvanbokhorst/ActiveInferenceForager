from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def engage(self, user_input):
        """Process user input and generate an appropriate response."""
        pass

    @abstractmethod
    def update_state(self, user_input):
        """Update the agent's internal state based on user input."""
        pass

    @abstractmethod
    def decide_action(self):
        """Decide the next action to take based on the current state."""
        pass
