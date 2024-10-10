# ActiveInferenceForager

An adaptive AI agent implementing the Free Energy Principle and Active Inference in dynamic environments.

## Current state

In the ActiveInferenceForager project, we have implemented a DQNFEPAgent that combines Deep Q-Network (DQN) with the Free Energy Principle (FEP) for decision-making and learning within dynamic environments. This project aims to create an adaptive AI agent that implements the Free Energy Principle and Active Inference.

### DQNFEPAgent Implementation
The DQNFEPAgent is implemented in the `DQNFEPAgent` class found in the [advanced_fep_agent.py](src/active_inference_forager/agents/advanced_fep_agent.py) file. Key characteristics include:
- **Neural Network Architecture**: The DQN uses a multi-layer neural network to approximate the Q-values for state-action pairs.
- **Experience Replay**: The agent stores past experiences in a replay buffer to break the correlation between consecutive experiences and improve learning stability.
- **Target Network**: A separate target network is used to provide stable Q-value targets, reducing oscillations during training.
- **Epsilon-Greedy Policy**: The agent follows an epsilon-greedy policy for action selection, balancing exploration and exploitation.
- **Belief Updating**: The agent maintains and updates beliefs about the environment based on observations, implementing key concepts from the Free Energy Principle.
- **Free Energy Calculation**: The agent calculates and minimizes free energy, which drives its learning and decision-making processes.

### Learning Process
The DQNFEPAgent undergoes a learning process through interaction with the environment:
1. **Belief Update**: The agent updates its beliefs about the environment based on new observations.
2. **Free Energy Calculation**: The agent calculates the free energy based on its current beliefs and observations.
3. **Action Selection**: Based on the current state and beliefs, the agent selects an action using its Q-network and exploration strategy.
4. **Environment Interaction**: The selected action is applied to the environment, which returns the next state, reward, and a done flag.
5. **Experience Storage**: The experience (state, action, reward, next state, done) is stored in the replay buffer.
6. **Batch Learning**: A batch of experiences is sampled from the replay buffer for training the Q-network.
7. **Q-Value Update**: The Q-values are updated using the Bellman equation, with the loss computed as the difference between current Q-values and target Q-values.
8. **Exploration Rate Decay**: The exploration rate is decayed over time to reduce exploration and increase exploitation.

### Project Context
In this project, the DQNFEPAgent is designed to explore and adapt in dynamic environments, combining the strengths of DQN and FEP approaches. The application's potential includes robotic navigation, resource management, adaptive user interfaces, financial decision-making, healthcare monitoring, and game AI.

For more detailed implementation, you can review the relevant code sections:
- [DQNFEPAgent in advanced_fep_agent.py](src/active_inference_forager/agents/advanced_fep_agent.py)
- [Training the DQNFEPAgent in main.py](src/active_inference_forager/main.py)
- [Environment interaction in simple_environment.py](src/active_inference_forager/environments/simple_environment.py)
