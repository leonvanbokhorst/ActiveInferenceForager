# Sentiment Analysis with Emotion Detection
#
# Instead of just predicting sentiment polarity, you can extend your analysis to
# detect specific emotions like joy, sadness, anger, etc. This can be particularly
# useful in more advanced user feedback analysis, chatbots, or social media monitoring
# where emotional context is critical.
#
# You can use a model like GoEmotions, which is fine-tuned to detect emotions in text

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load GoEmotions model for emotion detection
model_name = "bhadresh-savani/bert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

emotion_analyzer = pipeline(
    "text-classification", model=model, tokenizer=tokenizer, device=0
)

text = "I am so excited about this new opportunity, but I feel nervous too."
emotion_results = emotion_analyzer(text)

text = "The food was great, but they had a terrible service."
emotion_results = emotion_analyzer(text)

print(emotion_results)
