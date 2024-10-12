# ActiveInferenceForager

[![CI](https://github.com/leonvanbokhorst/ActiveInferenceForager/actions/workflows/ci.yml/badge.svg)](https://github.com/leonvanbokhorst/ActiveInferenceForager/actions/workflows/ci.yml)

This project aims to implement an intelligent chat agent using principles from Active Inference, a neuroscientific theory of brain function and decision-making. We've enhanced our existing Deep Q-Network (DQN) agent with elements of Active Inference, specifically focusing on free energy minimization.

## What is the Free Energy Principle?

The Free Energy Principle is a theory that explains how living systems, like humans and animals, maintain themselves and adapt to their environment by minimizing "free energy." In simple terms, it's about how organisms, from the brain to single cells, try to stay in a predictable, stable state by reducing uncertainty or "surprise" about the world around them.

At its core, the Free Energy Principle says that all living things have a basic goal: to avoid states that could harm them or make them less likely to survive. They do this by continuously making predictions about their environment and comparing these predictions with what they actually sense (through sight, sound, touch, etc.). When there’s a mismatch—when the world surprises us—our brain (or even simpler systems in the body) works to correct that, either by updating our beliefs (what we think is happening) or by acting to change the situation to match our expectations.

## What is Active Inference?

Active Inference is a way of explaining how living things, like humans and animals, interact with the world by constantly updating their understanding of it and acting to reduce uncertainty. Think of it as a process where your brain makes guesses about what's going on around you and then adjusts those guesses based on what it senses.

Here's a simple analogy: Imagine you're reaching into your pocket to find your phone. Your brain already has an idea of what the phone should feel like (that's your "prediction"). As you feel around, you're gathering sensory information (like touch) to confirm if your guess is right. If the object you touch doesn’t match your expectation (it feels like keys instead), your brain updates its guess, and you keep searching until the sensation matches the phone.

Active Inference Q&A section: [docs/active-inference-qa.md](docs/active-inference-qa.md)
