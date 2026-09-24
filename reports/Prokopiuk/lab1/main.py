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

def calculate_accuracy(net, X_norm, y_raw_true, c0=-10, c1=9):
    correct = 0
    threshold_mid = (c0 + c1) / 2
    
    for i in range(len(X_norm)):
        pred_norm = net.forward(X_norm[i:i+1])[0][0]
        pred_denorm = denormalize(pred_norm, min_val=c0, max_val=c1)
        
        assigned_class = c1 if pred_denorm >= threshold_mid else c0
        true_class = y_raw_true[i][0]
        
        if assigned_class == true_class:
            correct += 1
            
    return (correct / len(X_norm)) * 100

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

    final_epoch = epochs
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            pred = net.forward(X[i:i+1])
            total_error += np.sum(mse(pred, y[i:i+1]))
            net.backward(y[i:i+1], lr)
            
        error_history.append(total_error)

        if epoch % 2000 == 0:
            acc = calculate_accuracy(net, X, y_raw, c0, c1)
            print(f"Epoch {epoch}, Error: {total_error:.5f}, Accuracy: {acc:.1f}%")
        
        if total_error <= target_error:
            error_history.extend([total_error] * (epochs - len(error_history)))
            final_epoch = epoch + 1
            break
            
    return net, error_history, final_epoch, error_history[-1]

def predict_point(net, a, b, c0=-10, c1=9):
    raw_input = np.array([[a, b]])
    norm_input = normalize(raw_input, min_val=c0, max_val=c1)

    pred_norm = net.forward(norm_input)[0][0]
    
    pred_denorm = denormalize(pred_norm, min_val=c0, max_val=c1)
    
    threshold_mid = (c0 + c1) / 2
    assigned_class = c1 if pred_denorm >= threshold_mid else c0
    
    print(f"Input: A={a}, B={b} Output: {pred_denorm:.2f} Predicted class: {assigned_class}")

def plot_mse_history(history_mlp, history_perceptron):
    plt.figure(figsize=(10, 6))
    plt.plot(history_mlp, label='MLP', color='blue', linewidth=2)
    plt.plot(history_perceptron, label='SLP', color='red', linestyle='--', linewidth=2)
    
    plt.title('SLP vs cooler SLP(MLP)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

def plot_decision_boundary(net, c0=-10, c1=9):
    padding = 1.0
    x_min, x_max = c0 - padding, c1 + padding
    y_min, y_max = c0 - padding, c1 + padding

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_coords = np.c_[xx.ravel(), yy.ravel()]
    
    grid_norm = normalize(grid_coords, min_val=c0, max_val=c1)
    preds = np.array([net.forward(pt.reshape(1, -1))[0][0] for pt in grid_norm])
    preds_denorm = denormalize(preds, min_val=c0, max_val=c1)
    ZZ = preds_denorm.reshape(xx.shape)

    threshold_mid = (c0 + c1) / 2

    plt.figure(figsize=(8, 6))
    
    plt.contourf(xx, yy, ZZ, levels=[c0, threshold_mid, c1], colors=['lightblue', 'lightpink'], alpha=0.6)
    plt.contour(xx, yy, ZZ, levels=[threshold_mid], colors='red', linewidths=2)

    X_orig = np.array([[c0, c0], [c0, c1], [c1, c0], [c1, c1]])
    y_orig = np.array([c0, c1, c1, c0])
    
    for i, pt in enumerate(X_orig):
        dot_color = 'darkblue' if y_orig[i] == c0 else 'darkred'
        plt.scatter(pt[0], pt[1], color=dot_color, edgecolors='black', s=180, linewidth=1.5, zorder=5)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.title('Perceptron Decision Boundary')
    plt.xlabel('A')
    plt.ylabel('B')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(np.arange(-11, 11, 1))
    plt.yticks(np.arange(-11, 11, 1))
    plt.show()

if __name__ == "__main__":
    c0, c1 = -10, 9
    X_raw = np.array([[c0, c0], [c0, c1], [c1, c0], [c1, c1]])
    y_raw = np.array([[c0], [c1], [c1], [c0]])

    X = normalize(X_raw)
    y = normalize(y_raw)

    epochs = 200000
    lr = 1e-1

    print("training (2-2-1)...")
    net_mlp, history_mlp, final_epoch_mlp, final_error_mlp = train_network([2, 2, 1], X, y, epochs=epochs, lr=lr)
    final_acc_mlp = calculate_accuracy(net_mlp, X, y_raw, c0, c1)

    print("training (2-1)...")
    net_perceptron, history_perceptron, final_epoch_slp, final_error_slp = train_network([2, 1], X, y, epochs=epochs, lr=lr)
    final_acc_slp = calculate_accuracy(net_perceptron, X, y_raw, c0, c1)
    
    print(f"MLP Epochs: {final_epoch_mlp}   Error: {final_error_mlp}   Accuracy: {final_acc_mlp}")
    print(f"SLP Epochs: {final_epoch_slp}   Error: {final_error_slp}   Accuracy: {final_acc_slp}")

    print("\n--- test case:")
    test_cases = [(-10, -10), (-10, 9), (9, -10), (9, 9), (0, 0), (5, -5)]
    for a, b in test_cases:
        predict_point(net_mlp, a, b, c0, c1)

    plot_mse_history(history_mlp, history_perceptron)
    plot_decision_boundary(net_mlp)

    while True:
        user_input = input("\nInput nubers A and B through the space(or 'q' for exit): ").strip()
        if user_input.lower() in ['exit', 'выход', 'q']:
            print("Exit...")
            break
        
        try:
            parts = user_input.split()
            if len(parts) != 2:
                print("Error: you must input two numbers!")
                continue
            
            a = float(parts[0])
            b = float(parts[1])
            
            print(f"--- Point A={a}, B={b} ---")
            print("MLP (2-2-1):")
            predict_point(net_mlp, a, b, c0, c1)
            
        except ValueError:
            print("Error: invalid input!")