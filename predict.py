# predict.py
from engine import ProjectDifficultyEngine
engine = ProjectDifficultyEngine()

title = "AI based stock market forecasting using LSTM"
abstract = "This project uses news sentiment and historical prices to predict stock trends using RNNs."
# For now ask user to provide features (or auto-infer later):
feature_dict = {
    "Deep Learning": 1,
    "NLP": 1,
    "Medical/Health": 0,
    "Finance/Fraud": 1,
    "Vision": 0,
    "Real-Time": 0,
    "Unsupervised": 0,
    "Time-Series": 1,
    "Low Data": 0,
    "Noisy Data": 0
}
print(engine.predict(title, abstract, feature_dict))
