# ActiveInferenceForager

An adaptive AI agent implementing the Free Energy Principle and Active Inference in dynamic environments.

## Current state

In the ActiveInferenceForager project, DQN (Deep Q-Network) is used as an agent for decision-making and learning within dynamic environments. This project aims to create an adaptive AI agent that implements the Free Energy Principle and Active Inference.

### DQN Implementation
The DQN is implemented within the `DQNFEPAgent` class found in the [advanced_fep_agent.py](https://github.com/leonvanbokhorst/ActiveInferenceForager/blob/2b1f7b24268a38dcebc1a5646b8cced2dad3140f/src/active_inference_forager/agents/advanced_fep_agent.py) file. Key characteristics include:
- **Neural Network Architecture**: The DQN uses a multi-layer neural network to approximate the Q-values for state-action pairs.
- **Experience Replay**: The agent stores past experiences in a replay buffer to break the correlation between consecutive experiences and improve learning stability.
- **Target Network**: A separate target network is used to provide stable Q-value targets, reducing oscillations during training.
- **Epsilon-Greedy Policy**: The agent follows an epsilon-greedy policy for action selection, balancing exploration and exploitation.

### Learning Process
The DQN agent undergoes a learning process through interaction with the environment:
1. **Action Selection**: Based on the current state, the agent selects an action using its Q-network.
2. **Environment Interaction**: The selected action is applied to the environment, which returns the next state, reward, and a done flag.
3. **Experience Storage**: The experience (state, action, reward, next state, done) is stored in the replay buffer.
4. **Batch Learning**: A batch of experiences is sampled from the replay buffer for training the Q-network.
5. **Q-Value Update**: The Q-values are updated using the Bellman equation, with the loss computed as the difference between current Q-values and target Q-values.
6. **Epsilon Decay**: The epsilon value is decayed over time to reduce exploration and increase exploitation.

### Project Context
In this project, the DQN agent is integrated into the broader framework of the ActiveInferenceForager, which is designed to explore and adapt in dynamic environments. The application's potential includes robotic navigation, resource management, adaptive user interfaces, financial decision-making, healthcare monitoring, and game AI.

For more detailed implementation, you can review the relevant code sections:
- [DQNFEPAgent in advanced_fep_agent.py](https://github.com/leonvanbokhorst/ActiveInferenceForager/blob/2b1f7b24268a38dcebc1a5646b8cced2dad3140f/src/active_inference_forager/agents/advanced_fep_agent.py#L536-L627)
- [Training the DQN agent in main.py](https://github.com/leonvanbokhorst/ActiveInferenceForager/blob/2b1f7b24268a38dcebc1a5646b8cced2dad3140f/src/active_inference_forager/main.py#L1-L102)
- [Environment interaction in environment.py](https://github.com/leonvanbokhorst/ActiveInferenceForager/blob/2b1f7b24268a38dcebc1a5646b8cced2dad3140f/src/active_inference_forager/agents/environment.py#L1-L90)
