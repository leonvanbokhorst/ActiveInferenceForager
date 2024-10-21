from abc import ABC, abstractmethod


class GenerativeModel(ABC):
    def __init__(self):
        self.prior_beliefs = {}
        self.posterior_beliefs = {}

    @abstractmethod
    def predict(self, observations):
        pass

    @abstractmethod
    def update_beliefs(self, observations):
        pass
