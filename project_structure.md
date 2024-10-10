# Project Structure for ActiveInferenceForager

## Root Directory
- `.gitignore`: Specifies files and directories that should be ignored by Git.
- `dqn_fep_agent.pth` and `dqn_fep_model.pth`: Likely model files for a deep Q-network (DQN) with free energy principle (FEP) agent.
- `pytest.ini`: Configuration file for pytest, a testing framework.
- `README.md`: Documentation file providing an overview of the project.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `setup_project.sh`: A shell script for setting up the project environment.
- `setup.py`: A script for installing the package, typically used with Python projects.

## config/
- `default_config.yaml`: A YAML configuration file containing default settings for the project.

## docs/
- `branch-protection-info.md`: Documentation related to branch protection.
- `project-task-breakdown.md`: A document detailing the breakdown of project tasks.
- `project-vision-markdown.md`: A document outlining the project's vision.
- `README.md`: Additional documentation for the project.

## src/
- `active_inference_forager/`: Main source directory for the project.
  - `main.py`: The main entry point for the application.
  - `agents/`: Contains agent-related code.
    - `__init__.py`: Initializes the agents module.
    - `base_agent.py`: Defines the base class for agents.
    - `belief_node.py`: Contains code related to belief nodes.
    - `dqn_fep_agent.py`: Implementation of the DQN FEP agent.
  - `environments/`: Contains environment-related code.
    - `__init__.py`: Initializes the environments module.
    - `base_environment.py`: Defines the base class for environments.
    - `simple_environment.py`: Implementation of a simple environment.
  - `utils/`: Contains utility functions and classes.
    - `__init__.py`: Initializes the utils module.
    - `numpy_fields.py`: Utility functions related to NumPy.
  - `active_inference_forager.egg-info/`: Metadata directory for the Python package.

## tests/
- `__init__.py`: Initializes the tests module.
- `integration/`: Contains integration tests.
  - `__init__.py`: Initializes the integration tests module.
- `unit/`: Contains unit tests.
  - `__init__.py`: Initializes the unit tests module.
  - `test_agent.py`: Unit tests for agent-related code.
