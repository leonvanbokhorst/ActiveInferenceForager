# ActiveInferenceForager

[![CI](https://github.com/leonvanbokhorst/ActiveInferenceForager/actions/workflows/ci.yml/badge.svg)](https://github.com/leonvanbokhorst/ActiveInferenceForager/actions/workflows/ci.yml)

This project aims to implement an intelligent chat agent using principles from Active Inference, a neuroscientific theory of brain function and decision-making. We've enhanced our existing Deep Q-Network (DQN) agent with elements of Active Inference, specifically focusing on free energy minimization.

## What is the Free Energy Principle?

The Free Energy Principle is a theory that explains how living systems, like humans and animals, maintain themselves and adapt to their environment by minimizing "free energy." In simple terms, it's about how organisms, from the brain to single cells, try to stay in a predictable, stable state by reducing uncertainty or "surprise" about the world around them.

At its core, the Free Energy Principle says that all living things have a basic goal: to avoid states that could harm them or make them less likely to survive. They do this by continuously making predictions about their environment and comparing these predictions with what they actually sense (through sight, sound, touch, etc.). When there's a mismatch—when the world surprises us—our brain (or even simpler systems in the body) works to correct that, either by updating our beliefs (what we think is happening) or by acting to change the situation to match our expectations.

## What is Active Inference?

Active Inference is a way of explaining how living things, like humans and animals, interact with the world by constantly updating their understanding of it and acting to reduce uncertainty. Think of it as a process where your brain makes guesses about what's going on around you and then adjusts those guesses based on what it senses.

> Active Inference Q&A section: [docs/active-inference-qa.md](docs/active-inference-qa.md)

## Recent Updates: Improved Prompting Structure

We've recently updated our project to use a more sophisticated prompting structure for our language models. This new structure separates the system prompt from the user prompt, allowing for more flexible and context-aware responses.

### Key Changes:

1. **Separate System and User Prompts**: We now use distinct prompts for system instructions and user inputs. This allows for better control over the AI's persona and behavior.

2. **Enhanced LLM Providers**: Our LLM providers (OpenAI and HuggingFace) have been updated to support this new prompting structure, ensuring consistency across different models.

3. **Improved Response Generation**: Both the RapportBuilder and GoalSeeker classes now generate responses using this new structure, leading to more contextually appropriate and tailored responses.

These changes allow our AI agent to better understand the context of each interaction and provide more accurate and helpful responses.

For developers looking to contribute or understand the codebase, please refer to the updated classes in the `src/active_inference_forager/managers/` and `src/active_inference_forager/providers/` directories.
