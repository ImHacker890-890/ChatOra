import numpy as np
from core_lstm import LSTMCell
from core_attention import Attention

class NeuralChat:
    def __init__(self, vocab_size=10000, hidden_size=128):
        self.lstm = LSTMCell(vocab_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.dense = np.random.randn(hidden_size, vocab_size) * 0.01
        self.memory = []  # Хранит историю диалога

    def train(self, input_text, target_text, lr=0.01):
        # Преобразование текста в вектор (one-hot encoding)
        input_seq = self._text_to_seq(input_text)
        target_seq = self._text_to_seq(target_text)
        
        # Прямой проход
        h, c = np.zeros(self.hidden_size), np.zeros(self.hidden_size)
        outputs = []
        for word_idx in input_seq:
            x = self._onehot_encode(word_idx)
            h, c = self.lstm.forward(x, h, c)
            outputs.append(h)
        
        # Attention + предсказание
        context = self.attention.forward(np.array(outputs))
        logits = np.dot(context, self.dense)
        predicted = np.argmax(logits)
        
        # Ошибка и обновление весов
        error = self._onehot_encode(predicted) - self._onehot_encode(target_seq[-1])
        self.dense -= lr * np.outer(context, error)
        
        return self.idx_to_word.get(predicted, "<UNK>")

    def respond(self, text):
        # Генерация ответа через LSTM
        input_seq = self._text_to_seq(text)
        self.memory.extend(input_seq[-3:])  # Окно контекста
        
        h, c = np.zeros(self.hidden_size), np.zeros(self.hidden_size)
        for word_idx in self.memory:
            x = self._onehot_encode(word_idx)
            h, c = self.lstm.forward(x, h, c)
        
        # Выбираем наиболее вероятное слово
        logits = np.dot(h, self.dense)
        return self.idx_to_word.get(np.argmax(logits), "Не понимаю")
