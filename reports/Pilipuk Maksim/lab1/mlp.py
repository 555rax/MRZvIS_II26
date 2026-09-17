import numpy as np


class MLP:
    def __init__(self, input=2, hidden=2, out=1, lr=0.5):
        self.lr = lr

        self.W1 = np.random.uniform(-1, 1, size=(input, hidden))
        self.W2 = np.random.uniform(-1, 1, size=(hidden, out))

        self.b1 = np.random.uniform(-1, 1, size=(hidden,))
        self.b2 = np.random.uniform(-1, 1, size=(out,))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_der(out):
        return out * (1.0 - out)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.z1 = np.dot(x, self.W1) + self.b1
        self.y1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.y1, self.W2) + self.b2
        self.y = self.sigmoid(self.z2)

        return self.y[0]

    def train(self, X, y, max_epochs=10000, target_error=1e-3):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        errors = []

        for epoch in range(max_epochs):
            total_error = 0.0

            for i in range(len(X)):
                x_i = X[i]
                target = y[i]

                output = self.forward(x_i)

                error = 0.5 * ((target - output) ** 2)
                total_error += error

                output_error = output - target
                delta2 = output_error * self.sigmoid_der(output)

                hidden_error = self.W2.flatten() * delta2
                delta1 = hidden_error * self.sigmoid_der(self.y1)

                self.W2 -= self.lr * np.outer(self.y1, delta2)
                self.b2 -= self.lr * delta2

                self.W1 -= self.lr * np.outer(x_i, delta1)
                self.b1 -= self.lr * delta1

            errors.append(total_error)

            if total_error <= target_error:
                return epoch + 1, total_error, errors

        return max_epochs, total_error, errors