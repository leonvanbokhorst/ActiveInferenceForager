from abc import ABC, abstractmethod


class InferenceEngine(ABC):
    def __init__(self, generative_model):
        self.generative_model = generative_model

    @abstractmethod
    def infer(self, observations):
        pass

    @abstractmethod
    def choose_action(self, beliefs):
        pass
