import numpy as np
import matplotlib.pyplot as plt

def sigmoid(wsum):
    return 1 / (1 + np.exp(-wsum))

def der_sigmoid(y):
    return y * (1 - y)

def mse(yj, ej):
    return 0.5 * (yj - ej) ** 2

def normalize(val, min_val=-10, max_val=9):
    return (val - min_val) / (max_val - min_val)

def denormalize(val, min_val=-10, max_val=9):
    return min_val + val * (max_val - min_val)

class NN:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        self.weights = []
        self.thresholds = []

        for i in range(self.num_layers - 1):
            limit = 1 / np.sqrt(layer_sizes[i])
            w = np.random.uniform(-limit, limit,(layer_sizes[i], layer_sizes[i + 1]))
            T = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(w)
            self.thresholds.append(T)

    def forward(self, X):
        self.activations = [X]
        current_x = X   #yj

        for i in range(self.num_layers-1):
            wsum = np.dot(current_x, self.weights[i]) - self.thresholds[i]
            current_x = sigmoid(wsum)
            self.activations.append(current_x)

        return current_x

    def backward(self, ethalon, lr):
        error = (self.activations[-1] - ethalon) * der_sigmoid(self.activations[-1])
        for i in range(self.num_layers-2, -1, -1):
            inputs = self.activations[i]
            self.weights[i] -= np.dot(inputs.T, error) * lr
            self.thresholds[i]+= np.sum(error, axis=0, keepdims=True) * lr

            if i > 0:
                error = np.dot(error, self.weights[i].T) * der_sigmoid(self.activations[i])

def train_network(layer_sizes, X, y, epochs=20000, lr=0.5, target_error=0.0001):
    net = NN(layer_sizes)
    error_history = []
    
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            pred = net.forward(X[i:i+1])
            total_error += np.sum(mse(pred, y[i:i+1]))
            net.backward(y[i:i+1], lr)
            
        error_history.append(total_error)
        
        if total_error <= target_error:
            error_history.extend([total_error] * (epochs - len(error_history)))
            break
            
    return net, error_history

if __name__ == "__main__":
    X_raw = np.array([[-10, -10], [-10, 9], [9, -10], [9, 9]])
    y_raw = np.array([[-10], [9], [9], [-10]])

    X = normalize(X_raw)
    y = normalize(y_raw)

    epochs = 20000
    lr = 0.5

    print("training (2-2-1)...")
    net_mlp, history_mlp = train_network([2, 2, 1], X, y, epochs=epochs, lr=lr)

    print("training (2-1)...")
    net_perceptron, history_perceptron = train_network([2, 1], X, y, epochs=epochs, lr=lr)

    plt.figure(figsize=(10, 6))
    plt.plot(history_mlp, label='MLP)', color='blue', linewidth=2)
    plt.plot(history_perceptron, label='SLP', color='red', linestyle='--', linewidth=2)
    
    plt.title('MLP vs SLP XOR')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()