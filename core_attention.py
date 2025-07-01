import numpy as np

class Attention:
    def __init__(self, hidden_size):
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01
    
    de
