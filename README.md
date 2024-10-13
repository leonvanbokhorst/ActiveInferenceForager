# Active Inference Forager

Active Inference Forager is a conversational AI system based on the Free Energy Principle (FEP) and Active Inference. This project implements a basic framework for a FEP-based conversational AI as part of Stage 1 development.

## Features

- Conversation management
- User modeling
- FEP-based topic prediction
- Mock LLM integration (extensible for real LLM integration)
- Basic logging and error handling

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ActiveInferenceForager.git
   cd ActiveInferenceForager
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the conversational AI:

```
python src/active_inference_forager/main.py
```

You can specify a different LLM model (when implemented) using the `--model` argument:

```
python src/active_inference_forager/main.py --model mock
```

## Running Tests

To run the test suite:

```
python -m unittest discover tests
```

## Project Structure

- `src/active_inference_forager/`: Main source code directory
  - `main.py`: Core implementation of the conversational AI
  - `llm_interface.py`: LLM interface and implementations
  - `config.py`: Configuration settings
- `tests/`: Test suite directory
- `docs/`: Project documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
