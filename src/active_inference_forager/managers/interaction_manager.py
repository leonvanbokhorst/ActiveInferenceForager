from abc import ABC, abstractmethod


class InteractionManager(ABC):
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine

    @abstractmethod
    def process_input(self, user_input):
        pass

    @abstractmethod
    def handle_proactive_behavior(self):
        pass
