import numpy as np

class MLP:
    def __init__(self, input_dim=2, hidden_dim=2, out_dim=1, lr=0.5, seed=None, hidden_act='sigmoid'):
        if seed is not None:
            np.random.seed(seed)
        self.lr = lr
        self.hidden_act = hidden_act

        self.W1 = np.random.uniform(-0.5, 0.5, size=(input_dim, hidden_dim))
        self.W2 = np.random.uniform(-0.5, 0.5, size=(hidden_dim, out_dim))

        self.b1 = np.random.uniform(-0.5, 0.5, size=(hidden_dim,))
        self.b2 = np.random.uniform(-0.5, 0.5, size=(out_dim,))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_der(out):
        return out * (1.0 - out)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_der(x):
        return np.where(x > 0, 1.0, 0.0)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.z1 = np.dot(x, self.W1) + self.b1
        
        if self.hidden_act == 'relu':
            self.y1 = self.relu(self.z1)
        else:
            self.y1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.y1, self.W2) + self.b2
        self.y = self.sigmoid(self.z2)

        return self.y[0]

    def train(self, X, y, max_epochs=30000, target_error=1e-3):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        errors = []

        for epoch in range(max_epochs):
            total_error = 0.0

            for i in range(len(X)):
                x_i = X[i]
                target = y[i]

                output = self.forward(x_i)

                eps = 1e-15
                output_clipped = np.clip(output, eps, 1 - eps)

                error = -(target * np.log(output_clipped) + (1 - target) * np.log(1 - output_clipped))
                delta2 = output - target

                total_error += error

                hidden_error = self.W2.flatten() * delta2
                
                if self.hidden_act == 'relu':
                    delta1 = hidden_error * self.relu_der(self.z1)
                else:
                    delta1 = hidden_error * self.sigmoid_der(self.y1)

                self.W2 -= self.lr * np.outer(self.y1, delta2)
                self.b2 -= self.lr * delta2

                self.W1 -= self.lr * np.outer(x_i, delta1)
                self.b1 -= self.lr * delta1

            errors.append(total_error)

            if total_error <= target_error:
                return epoch + 1, total_error, errors

        return max_epochs, total_error, errors