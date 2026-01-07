from transformers import pipeline

# 情感分析（默认下载英文模型）
classifier = pipeline("sentiment-analysis")
result = classifier("I love Hugging Face!")