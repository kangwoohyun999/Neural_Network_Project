# common/optimizer.py
import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params_and_grads):
        for W, dW, b, db in params_and_grads:
            W -= self.lr * dW
            b -= self.lr * db

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v_W = None
        self.v_b = None

    def update(self, params_and_grads):
        if self.v_W is None:
            # init velocity arrays to zeros of same shape
            self.v_W = [np.zeros_like(W) for W, *_ in params_and_grads]
            self.v_b = [np.zeros_like(b) for *_, b, _ in [(None, None, None, None)] ]  # dummy to avoid error

        # but params_and_grads is an iterable, easier to implement per-step externally
        # To keep simple: implement per-list use in trainer

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_W = None
        self.v_W = None
        self.m_b = None
        self.v_b = None
        self.iter = 0

    def update(self, params_and_grads):
        if self.m_W is None:
            self.m_W = [np.zeros_like(W) for W, *_ in params_and_grads]
            self.v_W = [np.zeros_like(W) for W, *_ in params_and_grads]
            self.m_b = [np.zeros_like(b) for *_, b, _ in params_and_grads]
            self.v_b = [np.zeros_like(b) for *_, b, _ in params_and_grads]
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i, (W, dW, b, db) in enumerate(params_and_grads):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dW
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (dW**2)
            W -= lr_t * self.m_W[i] / (np.sqrt(self.v_W[i]) + self.eps)

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db**2)
            b -= lr_t * self.m_b[i] / (np.sqrt(self.v_b[i]) + self.eps)
