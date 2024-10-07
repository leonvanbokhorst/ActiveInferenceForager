# ActiveInferenceForager

## Project Overview

ActiveInferenceForager is an ambitious project aimed at developing an adaptive artificial intelligence agent based on the principles of the Free Energy Principle (FEP) and Active Inference, as proposed by neuroscientist Karl Friston. This project bridges the gap between theoretical neuroscience and practical AI applications, creating an agent capable of learning, adapting, and making decisions in complex, dynamic environments.

## Current Status

As of October 2024, the project has made significant progress:

1. **Core Implementation**: 
   - Basic Active Inference algorithm implemented
   - Hierarchical belief structure in place
   - Action selection based on expected free energy minimization

2. **Environment**:
   - SimpleEnvironment class implemented with a 2D state space
   - Goal-oriented tasks with distance-based rewards

3. **Agent Development**:
   - Multiple agent versions implemented, including:
     - AdvancedFEPAgent
     - ImprovedAdvancedFEPAgent
     - FurtherImprovedAdvancedFEPAgent
     - DQNFEPAgent (combining Deep Q-Network with FEP principles)

4. **Learning and Adaptation**:
   - Belief updating mechanism in place
   - Experience replay buffer for improved learning
   - Softmax action selection with temperature adjustment

5. **Performance**:
   - The DQN-FEP agent has shown remarkable performance and quick learning
   - Rapid learning observed, reaching high average rewards within the first 20 episodes
   - Consistent high-level performance maintained after initial learning phase

6. **Visualization**:
   - Basic plotting implemented for rewards and steps per episode

## Project Structure

```
ActiveInferenceForager/
│
├── src/
│   └── active_inference_forager/
│       ├── agents/
│       │   ├── advanced_fep_agent.py
│       │   ├── belief_node.py
│       │   └── environment.py
│       └── utils/
│           └── numpy_fields.py
│
├── tests/
│
├── docs/
├── examples/
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ActiveInferenceForager.git
   cd ActiveInferenceForager
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the main training script:

```
python main.py
```

This will train the DQNFEPAgent in the SimpleEnvironment and display performance plots.

## Key Features

- Hierarchical belief structure
- Adaptive learning rate
- Softmax action selection with temperature adjustment
- Experience replay buffer
- Integration of Deep Q-Network principles with Free Energy Principle

## Future Directions

1. Implement more complex environments
2. Enhance visualization and analysis tools
3. Optimize performance for larger state spaces
4. Implement parallel processing for faster simulation
5. Develop case studies for practical applications

## Contributing

Contributions to the ActiveInferenceForager project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project is based on the theoretical work of Karl Friston and the Free Energy Principle. We thank the broader AI and neuroscience community for their ongoing research and insights in this field.

---

For more detailed information on the project's vision and task breakdown, please refer to the `PROJECT_VISION.md` and `project-task-breakdown.md` files in the repository.
