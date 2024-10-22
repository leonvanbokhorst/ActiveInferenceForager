import pytest
import numpy as np
from mas_dynamics_simulation.personality import (
    BigFivePersonalityTrait,
    Openness,
    Conscientiousness,
    Extraversion,
    Agreeableness,
    Neuroticism,
    BigFivePersonality
)

# Tests for BigFivePersonalityTrait

def test_big_five_personality_trait_initialization():
    trait = BigFivePersonalityTrait("Test", 0.5, "Test explanation")
    assert trait.name == "Test"
    assert trait.value == 0.5
    assert trait.explanation == "Test explanation"

def test_big_five_personality_trait_value_validation():
    with pytest.raises(ValueError):
        BigFivePersonalityTrait("Test", 1.5)
    with pytest.raises(ValueError):
        BigFivePersonalityTrait("Test", -0.5)

def test_big_five_personality_trait_value_setter():
    trait = BigFivePersonalityTrait("Test", 0.5)
    trait.value = 0.7
    assert trait.value == 0.7
    with pytest.raises(ValueError):
        trait.value = 1.5

def test_big_five_personality_trait_str_representation():
    trait = BigFivePersonalityTrait("Test", 0.5, "Test explanation")
    assert str(trait) == "Test: 0.50 - Test explanation"

# Tests for specific Big Five traits

def test_openness_initialization():
    openness = Openness()
    assert openness.name == "Openness"
    assert 0 <= openness.value <= 1
    assert "creative" in openness.explanation.lower()

def test_conscientiousness_initialization():
    conscientiousness = Conscientiousness()
    assert conscientiousness.name == "Conscientiousness"
    assert 0 <= conscientiousness.value <= 1
    assert "organized" in conscientiousness.explanation.lower()

def test_extraversion_initialization():
    extraversion = Extraversion()
    assert extraversion.name == "Extraversion"
    assert 0 <= extraversion.value <= 1
    assert "outgoing" in extraversion.explanation.lower()

def test_agreeableness_initialization():
    agreeableness = Agreeableness()
    assert agreeableness.name == "Agreeableness"
    assert 0 <= agreeableness.value <= 1
    assert "compassionate" in agreeableness.explanation.lower()

def test_neuroticism_initialization():
    neuroticism = Neuroticism()
    assert neuroticism.name == "Neuroticism"
    assert 0 <= neuroticism.value <= 1
    assert "negative emotions" in neuroticism.explanation.lower()

# Tests for BigFivePersonality

def test_big_five_personality_initialization():
    personality = BigFivePersonality()
    assert len(personality.traits) == 5
    assert all(isinstance(trait, BigFivePersonalityTrait) for trait in personality.traits.values())

def test_big_five_personality_initialization_with_traits():
    traits = {
        "openness": 0.7,
        "conscientiousness": 0.6,
        "extraversion": 0.5,
        "agreeableness": 0.4,
        "neuroticism": 0.3
    }
    personality = BigFivePersonality(traits)
    assert personality.traits["openness"].value == 0.7
    assert personality.traits["conscientiousness"].value == 0.6
    assert personality.traits["extraversion"].value == 0.5
    assert personality.traits["agreeableness"].value == 0.4
    assert personality.traits["neuroticism"].value == 0.3

def test_big_five_personality_similarity():
    p1 = BigFivePersonality({"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5})
    p2 = BigFivePersonality({"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5})
    assert p1.similarity(p2) == pytest.approx(1.0)

    p3 = BigFivePersonality({"openness": 1.0, "conscientiousness": 1.0, "extraversion": 1.0, "agreeableness": 1.0, "neuroticism": 1.0})
    assert p1.similarity(p3) == pytest.approx(0.5, abs=1e-6)

def test_big_five_personality_compare():
    p1 = BigFivePersonality({"openness": 0.8, "conscientiousness": 0.6, "extraversion": 0.5, "agreeableness": 0.3, "neuroticism": 0.3})
    p2 = BigFivePersonality({"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5})
    comparison = p1.compare(p2, "Other")
    assert "more open to experience" in comparison
    assert "less agreeable" in comparison
    assert "emotionally stable" in comparison

def test_big_five_personality_str_representation():
    personality = BigFivePersonality({"openness": 0.9, "conscientiousness": 0.6, "extraversion": 0.5, "agreeableness": 0.1, "neuroticism": 0.3})
    str_rep = str(personality)
    assert "very high openness" in str_rep
    assert "high conscientiousness" in str_rep
    assert "moderate extraversion" in str_rep
    assert "very low agreeableness" in str_rep
    assert "low neuroticism" in str_rep

def test_big_five_personality_get_level_description():
    assert BigFivePersonality._get_level_description(0.1) == "Very Low"
    assert BigFivePersonality._get_level_description(0.3) == "Low"
    assert BigFivePersonality._get_level_description(0.5) == "Moderate"
    assert BigFivePersonality._get_level_description(0.7) == "High"
    assert BigFivePersonality._get_level_description(0.9) == "Very High"

def test_big_five_personality_generate_realistic_value():
    np.random.seed(42)  # Set seed for reproducibility
    value = BigFivePersonality._generate_realistic_value(0.5, 0.15)
    assert 0 <= value <= 1

def test_big_five_personality_random():
    np.random.seed(42)  # Set seed for reproducibility
    personality = BigFivePersonality.random()
    assert isinstance(personality, BigFivePersonality)
    assert all(0 <= trait.value <= 1 for trait in personality.traits.values())

def test_big_five_personality_random_variations():
    np.random.seed(42)  # Set seed for reproducibility
    variations = ["balanced", "gentle", "bold", "analytical", "random"]
    for variation in variations:
        personality = BigFivePersonality.random(variation=variation)
        assert isinstance(personality, BigFivePersonality)
        assert all(0 <= trait.value <= 1 for trait in personality.traits.values())

def test_big_five_personality_random_volatility():
    np.random.seed(42)  # Set seed for reproducibility
    low_volatility = BigFivePersonality.random(volatility=0.05)
    high_volatility = BigFivePersonality.random(volatility=0.2)
    assert isinstance(low_volatility, BigFivePersonality)
    assert isinstance(high_volatility, BigFivePersonality)

# Additional tests for edge cases and specific behaviors

def test_big_five_personality_trait_equality():
    trait1 = BigFivePersonalityTrait("Test", 0.5)
    trait2 = BigFivePersonalityTrait("Test", 0.5)
    trait3 = BigFivePersonalityTrait("Test", 0.6)
    assert trait1 == trait2
    assert trait1 != trait3

def test_big_five_personality_similarity_edge_cases():
    p1 = BigFivePersonality({"openness": 0, "conscientiousness": 0, "extraversion": 0, "agreeableness": 0, "neuroticism": 0})
    p2 = BigFivePersonality({"openness": 1, "conscientiousness": 1, "extraversion": 1, "agreeableness": 1, "neuroticism": 1})
    assert p1.similarity(p2) == 0

def test_big_five_personality_compare_identical():
    p = BigFivePersonality({"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5})
    comparison = p.compare(p, "Other")
    assert comparison == "You have a very similar personality to Other."

def test_big_five_personality_random_reproducibility():
    np.random.seed(42)
    p1 = BigFivePersonality.random()
    np.random.seed(42)
    p2 = BigFivePersonality.random()
    assert p1.similarity(p2) == 1.0

def test_big_five_personality_trait_value_extremes():
    trait_low = BigFivePersonalityTrait("Test", 0)
    trait_high = BigFivePersonalityTrait("Test", 1)
    assert trait_low.value == 0
    assert trait_high.value == 1

def test_big_five_personality_similarity_description():
    p1 = BigFivePersonality({"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5})
    p2 = BigFivePersonality({"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5})
    p3 = BigFivePersonality({"openness": 1.0, "conscientiousness": 1.0, "extraversion": 1.0, "agreeableness": 1.0, "neuroticism": 1.0})
    p4 = BigFivePersonality({"openness": 0.2, "conscientiousness": 0.2, "extraversion": 0.8, "agreeableness": 0.6, "neuroticism": 0.3})

    assert p1.similarity_description(p2, "Person B") == "You and Person B have very similar personalities."
    assert p1.similarity_description(p3, "Person C") == "You and Person C have quite different personalities."
    assert p1.similarity_description(p4, "Person D") == "You and Person D have somewhat different personalities."

def test_big_five_personality_getitem_setitem():
    personality = BigFivePersonality()
    assert isinstance(personality["openness"], BigFivePersonalityTrait)
    personality["openness"] = 0.8
    assert personality["openness"].value == 0.8
    with pytest.raises(ValueError):
        personality["invalid_trait"] = 0.5

def test_big_five_personality_similarity_invalid_type():
    p1 = BigFivePersonality()
    with pytest.raises(ValueError):
        p1.similarity("not a BigFivePersonality object")

def test_big_five_personality_random_correlated_traits():
    np.random.seed(42)
    personality = BigFivePersonality.random()
    if personality["conscientiousness"].value > 0.6:
        assert personality["neuroticism"].value < 0.6
    if personality["extraversion"].value > 0.6:
        assert personality["openness"].value > 0.4
    if personality["neuroticism"].value > 0.7:
        assert personality["extraversion"].value < 0.6

def test_big_five_personality_compare_edge_cases():
    p1 = BigFivePersonality({"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5})
    p2 = BigFivePersonality({"openness": 1.0, "conscientiousness": 1.0, "extraversion": 1.0, "agreeableness": 1.0, "neuroticism": 1.0})
    comparison = p1.compare(p2, "Other")
    assert all(trait in comparison for trait in ["open to experience", "conscientious", "extraverted", "agreeable"])
    assert "neurotic" not in comparison

def test_big_five_personality_similarity_description_all_levels():
    base = BigFivePersonality({"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5})
    
    very_similar = BigFivePersonality({"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5})
    assert "very similar" in base.similarity_description(very_similar, "Other")
    
    similar = BigFivePersonality({"openness": 0.7, "conscientiousness": 0.7, "extraversion": 0.7, "agreeableness": 0.7, "neuroticism": 0.7})
    assert "similar" in base.similarity_description(similar, "Other")
    
    somewhat_similar = BigFivePersonality({"openness": 0.8, "conscientiousness": 0.3, "extraversion": 0.6, "agreeableness": 0.4, "neuroticism": 0.5})
    assert "somewhat similar" in base.similarity_description(somewhat_similar, "Other")
    
    somewhat_different = BigFivePersonality({"openness": 0.8, "conscientiousness": 0.2, "extraversion": 0.3, "agreeableness": 0.7, "neuroticism": 0.1})
    assert "somewhat different" in base.similarity_description(somewhat_different, "Other")
    
    quite_different = BigFivePersonality({"openness": 0.9, "conscientiousness": 0.1, "extraversion": 0.1, "agreeableness": 0.9, "neuroticism": 0.1})
    assert "quite different" in base.similarity_description(quite_different, "Other")

