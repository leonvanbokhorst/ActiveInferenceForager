from active_inference_forager.managers.interaction_manager import InteractionManager

class GoalSeeker(InteractionManager):
    def __init__(self, inference_engine, llm_provider):
        super().__init__(inference_engine)
        self.llm_provider = llm_provider

    def process_input(self, user_input):
        observations = self.extract_features(user_input)
        beliefs = self.inference_engine.infer(observations)
        action = self.inference_engine.choose_action(beliefs)
        response = self.generate_response(action, user_input)
        return response

    def extract_features(self, user_input):
        # Simple intent recognition
        if "reset router" in user_input.lower():
            intent = "reset_router"
        else:
            intent = "unknown"
        return {"intent": intent}

    def generate_response(self, action, user_input):
        system_prompt = """
        You are an AI assistant specialized in providing technical support and guidance. 
        Your responses should be clear, concise, and tailored to the user's specific needs.
        """

        if action == "provide_router_reset_instructions":
            user_prompt = "Provide step-by-step instructions to help the user reset their router."
        else:
            user_prompt = f"Assist the user based on their input: '{user_input}'"

        response = self.llm_provider.generate_response(user_prompt, system_prompt=system_prompt)
        return response

    def handle_proactive_behavior(self):
        system_prompt = """
        You are an AI assistant designed to engage users proactively. 
        Your suggestions should be relevant, helpful, and encourage further interaction.
        """

        user_prompt = "Suggest a proactive action or question to engage the user and further the conversation."

        proactive_response = self.llm_provider.generate_response(user_prompt, system_prompt=system_prompt)
        return proactive_response
