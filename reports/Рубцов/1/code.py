import numpy as np
import matplotlib.pyplot as plt

def sigmoid(wsum):
    return 1 / (1 + np.exp(-np.clip(wsum, -500, 500)))

def der_sigmoid(y):
    return y * (1 - y)

def mse(yj, ej):
    return 0.5 * (yj - ej) ** 2

def normalize(val, min_val=-1, max_val=4):
    return (val - min_val) / (max_val - min_val)

def denormalize(val, min_val=-1, max_val=4):
    return min_val + val * (max_val - min_val)

def calculate_accuracy(net, X_norm, y_raw_true, c0=4, c1=-1):
    correct = 0
    for i in range(len(X_norm)):
        pred_norm = net.forward(X_norm[i:i+1])[0][0]
        pred_denorm = denormalize(pred_norm, min_val=c1, max_val=c0)
        
        assigned_class = c1 if abs(pred_denorm - c1) < abs(pred_denorm - c0) else c0
        true_class = y_raw_true[i][0]
        
        if assigned_class == true_class:
            correct += 1
    return (correct / len(X_norm)) * 100

class NN:
    def __init__(self, layer_sizes):
        np.random.seed(1234)  
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        self.weights = []
        self.thresholds = []

        for i in range(self.num_layers - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            T = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(w)
            self.thresholds.append(T)

    def forward(self, X):
        self.activations = [X]
        current_x = X  

        for i in range(self.num_layers - 1):
            wsum = np.dot(current_x, self.weights[i]) - self.thresholds[i]
            current_x = sigmoid(wsum)
            self.activations.append(current_x)

        return current_x

    def backward(self, ethalon, lr):
        error = (self.activations[-1] - ethalon) * der_sigmoid(self.activations[-1])
        for i in range(self.num_layers - 2, -1, -1):
            inputs = self.activations[i]
            self.weights[i] -= np.dot(inputs.T, error) * lr
            self.thresholds[i] += np.sum(error, axis=0, keepdims=True) * lr

            if i > 0:
                error = np.dot(error, self.weights[i].T) * der_sigmoid(self.activations[i])

def train_network(layer_sizes, X, y, y_raw, c0, c1, epochs=10000, lr=0.7, target_error=0.001):
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
        
        if total_error <= target_error:
            final_epoch = epoch + 1
            break
            
    return net, error_history, final_epoch, error_history[-1]

def predict_point(net, a, b, c0=4, c1=-1):
    raw_input = np.array([[a, b]])
    norm_input = normalize(raw_input, min_val=c1, max_val=c0)

    pred_norm = net.forward(norm_input)[0][0]
    pred_denorm = denormalize(pred_norm, min_val=c1, max_val=c0)
    
    assigned_class = c1 if abs(pred_denorm - c1) < abs(pred_denorm - c0) else c0
    print(f"Вход: A={a:5.1f}, B={b:5.1f} | Выход сети: {pred_denorm:6.2f} | Ближайший класс: {assigned_class}")

if __name__ == "__main__":
    c0, c1 = 4.0, -1.0
    X_raw = np.array([[c0, c0], [c0, c1], [c1, c0], [c1, c1]])
    y_raw = np.array([[c0], [c1], [c1], [c0]])

    X = normalize(X_raw, min_val=c1, max_val=c0)
    y = normalize(y_raw, min_val=c1, max_val=c0)

    epochs = 10000
    lr = 0.7

    print("Обучение MLP (2-2-1)...")
    net_mlp, history_mlp, final_epoch_mlp, final_error_mlp = train_network([2, 2, 1], X, y, y_raw, c0, c1, epochs=epochs, lr=lr)
    final_acc_mlp = calculate_accuracy(net_mlp, X, y_raw, c0, c1)

    print("Обучение Однослойного персептрона (2-1)...")
    net_perceptron, history_perceptron, final_epoch_slp, final_error_slp = train_network([2, 1], X, y, y_raw, c0, c1, epochs=epochs, lr=lr)
    final_acc_slp = calculate_accuracy(net_perceptron, X, y_raw, c0, c1)
    
    print(f"\n--- ИТОГОВЫЕ РЕЗУЛЬТАТЫ ---")
    print(f"MLP (2-2-1) Эпох: {final_epoch_mlp} | Ошибка: {final_error_mlp:.5f} | Accuracy: {final_acc_mlp:.1f}%")
    print(f"SLP (2-1)   Эпох: {final_epoch_slp} | Ошибка: {final_error_slp:.5f} | Accuracy: {final_acc_slp:.1f}%")

    while True:
        print("\n=== РЕЖИМ ФУНКЦИОНИРОВАНИЯ ===")
        print("1. Автоматический тест (предопределенные примеры)")
        print("2. Ввод чисел с клавиатуры")
        print("3. Выход (и показать графики)")
        
        choice = input("Выберите режим (1, 2 или 3): ").strip()
        
        if choice == '1':
            print("\n--- Автоматический тест (MLP) ---")
            test_cases = [(4.0, 4.0), (4.0, -1.0), (-1.0, 4.0), (-1.0, -1.0), (2.0, 2.0), (0.0, 0.0)]
            for a, b in test_cases:
                predict_point(net_mlp, a, b, c0, c1)
                
        elif choice == '2':
            print("\nВведите два числа A и B через пробел (или 'q' для возврата в меню):")
            while True:
                user_input = input("-> ").strip()
                if user_input.lower() in ['q', 'exit', 'выход']:
                    break
                try:
                    parts = user_input.split()
                    if len(parts) != 2:
                        print("Ошибка: нужно ввести ровно два числа через пробел!")
                        continue
                    a = float(parts[0])
                    b = float(parts[1])
                    predict_point(net_mlp, a, b, c0, c1)
                except ValueError:
                    print("Ошибка: неверный формат чисел!")
                    
        elif choice == '3':
            print("Завершение работы и построение графика...")
            break
        else:
            print("Неверный выбор, попробуйте снова.")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history_mlp) + 1), history_mlp, label='MLP (2-2-1)', color='blue', linewidth=2)
    plt.plot(range(1, len(history_perceptron) + 1), history_perceptron, label='SLP (2-1)', color='red', linestyle='--', linewidth=2)
    
    plt.yscale('log')
    plt.title('График сходимости ошибок')
    plt.xlabel('Эпохи')
    plt.ylabel('Суммарная ошибка MSE')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()