from active_inference_forager.agent import Agent
from active_inference_forager.managers.rapport_builder import RapportBuilder
from active_inference_forager.managers.goal_seeker import GoalSeeker


class LLMProactiveAgent(Agent):
    def __init__(self, rapport_builder, goal_seeker):
        super().__init__()
        self.rapport_builder = rapport_builder
        self.goal_seeker = goal_seeker
        self.conversation_history = []
        self.current_user_input = None
        self.current_beliefs = {}

    def engage(self, user_input):
        # Update the agent's state with the new user input
        self.update_state(user_input)
        # Decide the best action to take
        response = self.decide_action()
        # Update conversation history
        self.conversation_history.append({"user": user_input, "agent": response})
        return response

    def update_state(self, user_input):
        self.current_user_input = user_input
        # Optionally, you could update beliefs or other internal states here

    def decide_action(self):
        # Simple logic to choose between rapport building and goal seeking
        if "reset router" in self.current_user_input.lower():
            response = self.goal_seeker.process_input(self.current_user_input)
        else:
            response = self.rapport_builder.process_input(self.current_user_input)
        return response

    def run(self):
        print("LLM Proactive Agent is running. Type 'exit' to end the conversation.")
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                print("Agent: Goodbye!")
                break
            response = self.engage(user_input)
            print(f"Agent: {response}")
