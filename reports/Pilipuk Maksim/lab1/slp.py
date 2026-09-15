import numpy as np

class SLP:
    def __init__(self, input=2, lr=0.1):
        self.lr = lr
        self.weights = np.random.uniform(-1, 1, size=(input,))
        self.bias = np.random.uniform(-1, 1)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_der(out):
        return out * (1.0 - out)

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.sigmoid(z)

    def train(self, X, y, max_epochs=10000, target_error=1e-3):
        errors = []

        for epoch in range(max_epochs):
            total_error = 0.0

            for i in range(len(X)):
                x_i = X[i]
                target = y[i]

                out = self.forward(x_i)

                error = 0.5 * ((target - out) ** 2)
                total_error += error

                delta = (out - target) * self.sigmoid_der(out)
                self.weights -= self.lr * delta * x_i
                self.bias -= self.lr * delta

            errors.append(total_error)

            if total_error <= target_error:
                return epoch + 1, total_error, errors

        return max_epochs, total_error, errors