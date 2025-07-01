from core_model import NeuralChat
from data_loader import load_dialogues

model = NeuralChat()
dialogues = load_dialogues("data/dialogues.txt")  # Формат: "вопрос|ответ"

for epoch in range(100):
    for question, answer in dialogues:
        model.train(question, answer)
        print(f"Q: {question} -> A: {model.respond(question)}")
