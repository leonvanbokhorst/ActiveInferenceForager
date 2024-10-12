# ActiveInferenceForager

[![CI](https://github.com/leonvanbokhorst/ActiveInferenceForager/actions/workflows/ci.yml/badge.svg)](https://github.com/leonvanbokhorst/ActiveInferenceForager/actions/workflows/ci.yml)

Current iteration of the Active Inference Forager project as of 2024-10-11.

This project aims to implement an intelligent chat agent using principles from Active Inference, a neuroscientific theory of brain function and decision-making. We've enhanced our existing Deep Q-Network (DQN) agent with elements of Active Inference, specifically focusing on free energy minimization.

## Decision-Making Process

We explored three main options for implementing Active Inference in our chat agent:

1. Continuous Action Space
2. Full Active Inference Implementation
3. Incorporating Free Energy into Current DQN

After careful consideration, we chose Option 3: Incorporating Free Energy into our Current DQN. Here's why:

### Reasons for Our Choice

1. **Incremental Improvement**: This option allows us to build upon our existing DQN implementation, making it easier to implement and understand the new concepts.

2. **Introduction to Active Inference**: It introduces core Active Inference principles without requiring a complete overhaul of our system.

3. **Balanced Approach**: It provides a good balance between improving the agent's capabilities and maintaining a manageable level of complexity.

4. **Learning Opportunity**: This approach serves as a stepping stone towards more advanced implementations, allowing us to gain experience with Active Inference principles gradually.

5. **Practical Implementation**: Given our current expertise level, this option is the most practical next step in enhancing our chat agent.

## Key Components of the Implementation

1. **Modified Neural Network**: We've updated our DQN to include state prediction, a key component of Active Inference.

2. **Free Energy Minimization**: Our loss function now includes both the traditional TD error and a state prediction error, aligning with the Active Inference principle of minimizing prediction errors.

3. **Flexible Architecture**: We've introduced a `state_prediction_weight` parameter, allowing us to adjust the balance between standard Q-learning and Active Inference components.

4. **Belief and Free Energy Updates**: We've incorporated methods for updating beliefs and free energy, though these are simplified versions that can be expanded upon in future iterations.

## Adapting to Chat Agent Context

To adapt this implementation for a chat agent:

1. **State Representation**: Modify the state to represent the conversation history, current query, and any relevant context.
2. **Action Space**: Define actions as different types of responses or intents the agent can choose from.
3. **Reward Function**: Design a reward function that considers user satisfaction, conversation coherence, and task completion.

## Next Steps

1. Implement the modified DQNFEPAgent class as described.
2. Adjust the main training loop to work with the new agent.
3. Develop appropriate state representations and action spaces for the chat context.
4. Create a suitable reward function for evaluating chat agent performance.
5. Test the implementation and iteratively refine as needed.

## Conclusion

This approach allows us to introduce Active Inference principles into our chat agent gradually. As we become more familiar with these concepts and their implementation, we can further refine our model and potentially move towards a more comprehensive Active Inference framework in the future.
