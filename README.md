# ActiveInferenceForager

[![CodeQL](https://github.com/leonvanbokhorst/ActiveInferenceForager/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/leonvanbokhorst/ActiveInferenceForager/actions/workflows/github-code-scanning/codeql) [![Run Tests](https://github.com/leonvanbokhorst/ActiveInferenceForager/actions/workflows/run-tests.yml/badge.svg)](https://github.com/leonvanbokhorst/ActiveInferenceForager/actions/workflows/run-tests.yml)

ActiveInferenceForager is an open-source action research project that implements an intelligent agent using principles from Active Inference, a neuroscientific theory of brain function and decision-making. This project aims to bridge the gap between theoretical neuroscience and practical AI applications, creating adaptive agents capable of learning and making decisions in complex, dynamic environments.

## Table of Contents

- [ActiveInferenceForager](#activeinferenceforager)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Key Components](#key-components)
  - [Multi-Agent System (MAS) Dynamics Simulation](#multi-agent-system-mas-dynamics-simulation)
  - [Proofs of Concept (POCs)](#proofs-of-concept-pocs)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Project Overview

ActiveInferenceForager implements an intelligent chat agent using principles from Active Inference, focusing on free energy minimization. The project includes:

- A core implementation of Active Inference algorithms
- A Multi-Agent System (MAS) dynamics simulation framework
- Several proofs of concept demonstrating key concepts
- A flexible and extensible architecture for further research and development

For a detailed explanation of the Free Energy Principle and Active Inference, please refer to our [Active Inference Q&A](docs/active-inference-qa.md).

## Key Components

1. LLM Proactive Agent: An advanced chatbot implementing Active Inference principles.
2. Goal Seeker: Manages the agent's goals and decision-making process.
3. Rapport Builder: Handles user interaction and relationship building.
4. OpenAI Provider: Integrates with OpenAI's language models for natural language processing.

## Multi-Agent System (MAS) Dynamics Simulation

The MAS dynamics simulation framework allows for modeling complex interactions between multiple agents in various environments. Key components include:

1. Agent: Abstract base class for implementing various types of agents.
2. Environment: Represents the world in which agents operate.
3. Personality: Implements the Big Five personality model for agents.
4. Decision Making: Handles agent decision-making processes.

## Proofs of Concept (POCs)

The project includes several POCs demonstrating key concepts:

1. Variational Free Energy Minimization
2. Prediction Error Propagation
3. VFE Minimization

These POCs provide practical implementations of core Active Inference concepts and serve as building blocks for more complex agent behaviors.

## Installation

To set up the ActiveInferenceForager project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ActiveInferenceForager.git
   cd ActiveInferenceForager
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install the project in editable mode:

   ```bash
   pip install -e .
   ```

## Usage

To use the ActiveInferenceForager project, follow these steps:

1. Ensure you have completed the installation process described above.

2. Set up your OpenAI API key:

   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

3. Run the main application:

   ```bash
   python src/main.py
   ```

4. To run specific proofs of concept:

   ```bash
   python src/poc/variational_free_energy.py
   python src/poc/prediction_error_propagation.py
   python src/poc/vfe_minimization.py
   ```

5. To run the Multi-Agent System simulation:

   ```bash
   python src/mas_dynamics_simulation/simulation.py
   ```

For more detailed usage instructions and examples, please refer to the documentation in the `docs` folder.

## Contributing

We welcome contributions to the ActiveInferenceForager project! Here's how you can contribute:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Write your code and add tests if applicable.
4. Ensure all tests pass by running:

   ```bash
   pytest
   ```

5. Submit a pull request with a clear description of your changes.

Please make sure to follow our coding standards and commit message conventions. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
