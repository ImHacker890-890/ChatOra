import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # Веса для гейтов
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01

    def forward(self, x, h_prev, c_prev):
        concat = np.concatenate([h_prev, x])
        # Forget gate
        f = 1 / (1 + np.exp(-np.dot(self.Wf, concat)))
        # Input gate
        i = 1 / (1 + np.exp(-np.dot(self.Wi, concat)))
        # Output gate
        o = 1 / (1 + np.exp(-np.dot(self.Wo, concat)))
        # Cell state
        c_hat = np.tanh(np.dot(self.Wc, concat))
        
        c_new = f * c_prev + i * c_hat
        h_new = o * np.tanh(c_new)
        return h_new, c_new
