import numpy as np
from collections import defaultdict

class DataLoader:
    def __init__(self):
        self.word_to_idx = defaultdict(lambda: len(self.word_to_idx))
        self.idx_to_word = {}
        
    def load_dialogues(self, path):
        with open(path) as f:
            pairs = [line.strip().split("|") for line in f]
        
        # Строим словарь
        for q, a in pairs:
            for word in q.split() + a.split():
                _ = self.word_to_idx[word]
        self._update_vocab()
        
        # Векторизация
        X, y = [], []
        for q, a in pairs:
            X.append([self.word_to_idx[w] for w in q.split()])
            y.append([self.word_to_idx[w] for w in a.split()])
        
        return np.array(X), np.array(y)
    
    def _update_vocab(self):
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
