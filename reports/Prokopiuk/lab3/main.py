import numpy as np
import matplotlib.pyplot as plt

def sigmoid(wsum):
    wsum = np.clip(wsum, -500, 500)
    return 1 / (1 + np.exp(-wsum))

def der_sigmoid(wsum):
    y = sigmoid(wsum)
    return y * (1 - y)

def relu(wsum):
    return np.maximum(0, wsum)

def der_relu(wsum):
    return (wsum > 0).astype(float)

def mse(yj, ej):
    return 0.5 * (yj - ej) ** 2

def bce(yj, ej):
    eps = 1e-12
    yj = np.clip(yj, eps, 1 - eps)
    return -(ej * np.log(yj) + (1 - ej) * np.log(1 - yj))

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

def calculate_mae(net, X_norm, y_raw, c0=-10, c1=9):
    total_error = 0

    for i in range(len(X_norm)):
        pred_norm = net.forward(X_norm[i:i+1])[0][0]
        pred_raw = denormalize(pred_norm, min_val=c0, max_val=c1)

        total_error += abs(pred_raw - y_raw[i][0])

    return total_error / len(X_norm)

class NN:
    def __init__(self, layer_sizes, act_funcs, der_funcs):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.act_funcs = act_funcs
        self.der_funcs = der_funcs

        self.weights = []
        self.thresholds = []

        for i in range(self.num_layers - 1):
            if self.act_funcs[i] == relu:
                limit = np.sqrt(2 / layer_sizes[i])
            else:
                limit = 1 / np.sqrt(layer_sizes[i])
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            T = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(w)
            self.thresholds.append(T)

    def forward(self, X):
        self.wsums = []
        self.activations = [X]
        current_x = X   #yj

        for i in range(self.num_layers-1):
            wsum = np.dot(current_x, self.weights[i]) - self.thresholds[i]
            self.wsums.append(wsum)
            current_x = self.act_funcs[i](wsum)
            self.activations.append(current_x)

        return current_x

    def backward(self, ethalon, lr):
        error = (self.activations[-1] - ethalon) * der_sigmoid(self.activations[-1])
        for i in range(self.num_layers-2, -1, -1):
            inputs = self.activations[i]

            if i > 0:
                prev_error = (np.dot(error, self.weights[i].T) * der_sigmoid(self.activations[i]))

            self.weights[i] -= np.dot(inputs.T, error) * lr
            self.thresholds[i]+= np.sum(error, axis=0, keepdims=True) * lr

            if i > 0:
                error = prev_error

    def bce_backward(self, ethalon, lr):
        error = self.activations[-1] - ethalon
        for i in range(self.num_layers - 2, -1, -1):
            inputs = self.activations[i]

            w_current = self.weights[i].copy()
            
            self.weights[i] -= np.dot(inputs.T, error) * lr
            self.thresholds[i] += np.sum(error, axis=0, keepdims=True) * lr

            if i > 0:
                der = self.der_funcs[i-1](self.wsums[i-1])
                error = np.dot(error, w_current.T) * der


def train_network(layer_sizes, X, y, act_funcs, der_funcs, epochs=20000, lr=0.5, target_error=0.0001):
    net = NN(layer_sizes, act_funcs, der_funcs)
    error_history = []

    final_epoch = epochs
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            pred = net.forward(X[i:i+1])
            total_error += np.sum(mse(pred, y[i:i+1]))
            net.backward(y[i:i+1], lr)
            
        error_history.append(total_error)

        if epoch % 10000 == 0:
            acc = calculate_accuracy(net, X, y_raw, c0, c1)
            mae = calculate_mae(net, X, y_raw, c0, c1)
            print(f"Epoch {epoch}, Error: {total_error:.5f}, Accuracy: {acc:.1f}%, MAE: {mae}")
        
        if total_error <= target_error:
            final_epoch = epoch + 1
            converged = True
            break

    mae = calculate_mae(net, X, y_raw, c0, c1)
            
    return net, error_history, final_epoch, error_history[-1], mae, converged

def bce_train_network(layer_sizes, X, y, act_funcs, der_funcs, epochs=20000, lr=0.05, target_error=0.0001):
    net = NN(layer_sizes, act_funcs, der_funcs)
    error_history = []
    converged = False

    final_epoch = epochs
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            pred = net.forward(X[i:i+1])
            total_error += np.sum(bce(pred, y[i:i+1]))
            net.bce_backward(y[i:i+1], lr)
            
        error_history.append(total_error)

        if epoch % 10000 == 0:
            acc = calculate_accuracy(net, X, y_raw, c0, c1)
            mae = calculate_mae(net, X, y_raw, c0, c1)
            print(f"Epoch {epoch}, Error: {total_error:.5f}, Accuracy: {acc:.1f}%, MAE: {mae}")
                    
        if total_error <= target_error:
            final_epoch = epoch + 1
            converged = True
            break

    mae = calculate_mae(net, X, y_raw, c0, c1)
            
    return net, error_history, final_epoch, error_history[-1], mae, converged

def predict_point(net, a, b, c0=-10, c1=9):
    raw_input = np.array([[a, b]])
    norm_input = normalize(raw_input, min_val=c0, max_val=c1)

    pred_norm = net.forward(norm_input)[0][0]
    pred_denorm = denormalize(pred_norm, min_val=c0, max_val=c1)
    
    threshold_mid = (c0 + c1) / 2
    assigned_class = c1 if pred_denorm >= threshold_mid else c0
    
    print(
        f"Input: A={a}, B={b}\n"
        f"Normalized output: {pred_norm:.4f}\n"
        f"Original scale output: {pred_denorm:.4f}\n"
        f"Predicted class: {assigned_class}"
    )

def plot_convergence(hist_a, hist_b, target_error=0.01, label_a="Config A", label_b="Config B"):
    plt.figure(figsize=(10, 6))
    plt.plot(hist_a, label=label_a)
    plt.plot(hist_b, label=label_b)
    plt.axhline(y=target_error, linestyle="--", color='red', label=f"Target Ee = {target_error}")
    plt.xlabel("Epochs")
    plt.ylabel("Total Error (BCE)")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_epochs_bar(epochs_a, epochs_b):
    x = np.arange(5)
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, epochs_a, width, label="Config A (Sigmoid)")
    plt.bar(x + width/2, epochs_b, width, label="Config B (ReLU)")
    plt.xlabel("Run Number")
    plt.ylabel("Epochs to convergence")
    plt.title("Stability: Epochs across 5 seeds")
    plt.xticks(x, [f"Seed {i+1}" for i in range(5)])
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(net, title, X_raw, y_raw, c0=-10, c1=9):
    A = np.linspace(-10, 10, 200)
    B = np.linspace(-10, 10, 200)
    AA, BB = np.meshgrid(A, B)
    points = np.column_stack((AA.ravel(), BB.ravel()))
    
    points_norm = normalize(points, min_val=c0, max_val=c1)
    predictions_norm = net.forward(points_norm).reshape(AA.shape)
    predictions = denormalize(predictions_norm, min_val=c0, max_val=c1)

    plt.figure(figsize=(8, 7))
    contour = plt.contourf(AA, BB, predictions, levels=50, cmap="viridis")
    plt.colorbar(contour, label="Network output")
    
    plt.scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw[:, 0], cmap="coolwarm", 
                edgecolors="black", s=100, zorder=5)
    
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("A")
    plt.ylabel("B")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

def run_experiments(act_funcs, der_funcs, config_name, X, y, y_raw, seeds, epochs, lr, target_error, c0, c1):
    results = {
        "epochs": [], "errors": [], "accuracies": [],
        "maes": [], "converged": [], "nets": [], "histories": []
    }
    
    print(f"\n--- Запуск экспериментов: {config_name} ---")
    
    for seed in seeds:
        np.random.seed(seed)
        net, history, epoch, error, mae, converged = bce_train_network(
            [2, 2, 1], X, y, act_funcs, der_funcs,
            epochs=epochs, lr=lr, target_error=target_error
        )
        
        accuracy = calculate_accuracy(net, X, y_raw, c0, c1)
        
        results["epochs"].append(epoch)
        results["errors"].append(error)
        results["accuracies"].append(accuracy)
        results["maes"].append(mae)
        results["converged"].append(converged)
        results["nets"].append(net)
        results["histories"].append(history)
        
        print(f"Seed {seed}: epochs={epoch}, error={error:.6f}, acc={accuracy:.1f}%, MAE={mae:.4f}")
        
    return results



if __name__ == "__main__":
    c0, c1 = -10, 9
    
    X_raw = np.array([[c0, c0], [c0, c1], [c1, c0], [c1, c1]])
    y_raw = np.array([[c0], [c1], [c1], [c0]])
    
    X = normalize(X_raw, c0, c1)
    y_target = np.array([[0], [1], [1], [0]])

    # Настройки
    epochs = 50000
    lr = 0.01
    target_error = 0.01
    seeds = [3, 6, 7, 8, 9]

    act_A = [sigmoid, sigmoid]
    der_A = [der_sigmoid, der_sigmoid]
    
    act_B = [relu, sigmoid]
    der_B = [der_relu, der_sigmoid]

    # Запуск экспериментов
    res_A = run_experiments(act_A, der_A, "Config A (Sigmoid)", X, y_target, y_raw, 
                            seeds, epochs, lr, target_error, c0, c1)
    res_B = run_experiments(act_B, der_B, "Config B (ReLU)", X, y_target, y_raw, 
                            seeds, epochs, lr, target_error, c0, c1)

    # === ТАБЛИЦА СРАВНЕНИЯ (Пункт 10) ===
    print("\n" + "="*70)
    print("СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ (Средние значения за 5 запусков)")
    print("="*70)
    print(f"{'Метрика':<30} | {'Config A (Sigmoid)':<20} | {'Config B (ReLU)':<20}")
    print("-" * 70)
    print(f"{'Среднее кол-во эпох':<30} | {np.mean(res_A['epochs']):<20.1f} | {np.mean(res_B['epochs']):<20.1f}")
    print(f"{'Средняя итоговая ошибка':<30} | {np.mean(res_A['errors']):<20.5f} | {np.mean(res_B['errors']):<20.5f}")
    print(f"{'Средний Accuracy (%)':<30} | {np.mean(res_A['accuracies']):<20.1f} | {np.mean(res_B['accuracies']):<20.1f}")
    print(f"{'Средний MAE (исх. шкала)':<30} | {np.mean(res_A['maes']):<20.3f} | {np.mean(res_B['maes']):<20.3f}")
    print(f"{'Успешных запусков':<30} | {sum(res_A['converged'])}/5{'':<15} | {sum(res_B['converged'])}/5")
    print("="*70)

    # === ГРАФИКИ ===
    plot_convergence(res_A["histories"][0], res_B["histories"][0], target_error)
    
    plot_epochs_bar(res_A["epochs"], res_B["epochs"])
    
    plot_decision_boundary(res_A["nets"][0], "Boundary: Config A (Sigmoid)", X_raw, y_raw, c0, c1)
    plot_decision_boundary(res_B["nets"][0], "Boundary: Config B (ReLU)", X_raw, y_raw, c0, c1)

    print("\n--- Тестовые точки (4 обучающих + 3 дополнительных) ---")
    test_cases = [(-10, -10), (-10, 9), (9, -10), (9, 9), (0, 0), (5, -5), (-5, 5)]
    for a, b in test_cases:
        print(f"\nPoint ({a}, {b}):")
        print("[Sigmoid]")
        predict_point(res_A["nets"][0], a, b, c0, c1)
        print("[ReLU]")
        predict_point(res_B["nets"][0], a, b, c0, c1)

    
    
    print("\n" + "="*50)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ (Введите 'q' для выхода)")
    print("="*50)
    while True:
        user_input = input("\nВведите A и B через пробел: ").strip()
        if user_input.lower() in ["q", "exit", "выход"]:
            break
        try:
            a, b = map(float, user_input.split())
            if not (-10 <= a <= 10 and -10 <= b <= 10):
                print("Ошибка: A и B должны быть в диапазоне [-10; 10]!")
                continue
            
            print("\n[Config A - Sigmoid]")
            predict_point(res_A["nets"][0], a, b, c0, c1)
            print("\n[Config B - ReLU]")
            predict_point(res_B["nets"][0], a, b, c0, c1)
        except ValueError:
            print("Ошибка ввода! Введите два числа.")