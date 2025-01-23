import dspy

from typing import Literal

lm = dspy.LM('ollama_chat/qwen2.5:0.5b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class Classify(dspy.Signature):
    """Classify sentiment of a given sentence"""
    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField()

classify = dspy.Predict(Classify)
response = classify(sentence="这门课程非常棒")

print(response)