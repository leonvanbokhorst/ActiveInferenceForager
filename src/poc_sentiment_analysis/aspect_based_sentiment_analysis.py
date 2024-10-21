# Aspect-Based Sentiment Analysis (ABSA)
#
# ABSA breaks down sentiment analysis into specific aspects of the text. For instance,
# a restaurant review might mention both food quality and service, which have different
# sentiments. This requires training models to identify and classify sentiment at
# the aspect level.
#
# An approach with BERT might include fine-tuning on an ABSA-specific dataset,
# such as the SemEval datasets.
#
# Example of aspect-based sentiment analysis could involve using a framework
# like spacy for extracting aspects and then applying a sentiment classifier on each.abs

import spacy
from transformers import pipeline

# first, install the spacy model with: python -m spacy download en_core_web_trf
nlp = spacy.load("en_core_web_sm")


# Define a custom function to extract aspects (nouns) from text
def extract_aspects(text):
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]


# Sample review
text = "The food was great, but they had a terrible service."

# Extract aspects
aspects = extract_aspects(text)

# BERT sentiment analysis pipeline
sentiment_model = pipeline(
    "sentiment-analysis",
    device=0,
    model="distilbert-base-uncased-finetuned-sst-2-english",  # pre-trained model
)

# Apply sentiment analysis on each aspect
for aspect in aspects:
    sentiment = sentiment_model(aspect)
    print(f"Aspect: {aspect} -> Sentiment: {sentiment}")
