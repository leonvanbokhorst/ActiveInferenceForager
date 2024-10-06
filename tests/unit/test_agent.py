from ActiveInferenceForager.agents import Agent


def test_agent_initialization():
    agent = Agent()
    assert agent is not None


def test_agent_belief_update():
    agent = Agent()
    initial_belief = agent.get_belief()
    agent.update_belief({"observation": 1.0})
    updated_belief = agent.get_belief()
    assert updated_belief != initial_belief
