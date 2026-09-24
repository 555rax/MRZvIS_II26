import numpy as np
import matplotlib.pyplot as plt

def sigmoid(wsum):
    return 1 / (1 + np.exp(-wsum))

def der_sigmoid(y):
    return y * (1 - y)

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
            if i > 0:
                prev_error = (np.dot(error, self.weights[i].T) * der_sigmoid(self.activations[i]))
            
            self.weights[i] -= np.dot(inputs.T, error) * lr
            self.thresholds[i] += np.sum(error, axis=0, keepdims=True) * lr

            if i > 0:
                error = prev_error


def train_network(layer_sizes, X, y, epochs=20000, lr=0.5, target_error=0.0001):
    net = NN(layer_sizes)
    error_history = []
    converged = False

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

def bce_train_network(layer_sizes, X, y, epochs=20000, lr=0.5, target_error=0.0001):
    net = NN(layer_sizes)
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

def plot_all_histories(history_mlp_mse, history_mlp_bce,
                       history_slp_mse, history_slp_bce):

    histories = [
        (history_mlp_mse, "MLP 2-2-1 + MSE"),
        (history_mlp_bce, "MLP 2-2-1 + BCE"),
        (history_slp_mse, "SLP 2-1 + MSE"),
        (history_slp_bce, "SLP 2-1 + BCE")
    ]

    for history, title in histories:
        plt.figure(figsize=(10, 6))

        plt.plot(
            range(1, len(history) + 1),
            history,
            label="Sum error"
        )

        plt.xlabel("Epoches")
        plt.ylabel("Sum error")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_convergence(history_mse, history_bce, target_error=0.0001):
    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(history_mse) + 1),
        history_mse,
        label="MSE"
    )

    plt.plot(
        range(1, len(history_bce) + 1),
        history_bce,
        label="BCE"
    )

    plt.axhline(
        y=target_error,
        linestyle="--",
        label=f"Target Ee = {target_error}"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Es")
    plt.title("Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_epoch_scatter(mse_epochs, mse_converged,
                       bce_epochs, bce_converged):

    x = np.arange(5)
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        mse_epochs,
        width,
        label="MSE"
    )

    plt.bar(
        x + width / 2,
        bce_epochs,
        width,
        label="BCE"
    )

    plt.xlabel("№ launch")
    plt.ylabel("Epochs")
    plt.title("Scatter")
    plt.xticks(x, [f"Seed {seed}" for seed in [1, 2, 3, 4, 5]])
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_decision_boundary(net, title, c0=-10, c1=9):
    A = np.linspace(-10, 10, 200)
    B = np.linspace(-10, 10, 200)

    AA, BB = np.meshgrid(A, B)

    points = np.column_stack((AA.ravel(), BB.ravel()))

    points_norm = normalize(points, min_val=c0, max_val=c1)

    predictions_norm = net.forward(points_norm).reshape(AA.shape)

    predictions = denormalize(
        predictions_norm,
        min_val=c0,
        max_val=c1
    )

    plt.figure(figsize=(8, 7))

    contour = plt.contourf(
        AA,
        BB,
        predictions,
        levels=50,
        cmap="viridis"
    )

    plt.colorbar(contour, label="Network output")

    plt.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        c=y_raw[:, 0],
        cmap="coolwarm",
        edgecolors="black",
        s=100
    )

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.xlabel("A")
    plt.ylabel("B")
    plt.title(title)

    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_mae_comparison(net_mlp_mse, net_mlp_bce,
                        net_slp_mse, net_slp_bce,
                        c0=-10, c1=9):

    names = [
        "MLP + MSE",
        "MLP + BCE",
        "SLP + MSE",
        "SLP + BCE"
    ]

    values = [
        calculate_mae(net_mlp_mse, X, y_raw, c0, c1),
        calculate_mae(net_mlp_bce, X, y_raw, c0, c1),
        calculate_mae(net_slp_mse, X, y_raw, c0, c1),
        calculate_mae(net_slp_bce, X, y_raw, c0, c1)
    ]

    plt.figure(figsize=(9, 6))

    plt.bar(names, values)

    plt.xlabel("Config")
    plt.ylabel("MAE")
    plt.title("Scale Recovery Accuracy [-10, 9]")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

def run_experiments(train_func, X, y, y_raw, seeds, epochs, lr, target_error, c0, c1):
    results = {
        "epochs": [], "errors": [], "accuracies": [],
        "maes": [], "converged": [], "nets": [], "histories": []
    }

    for seed in seeds:
        np.random.seed(seed)

        net, history, epoch, error, mae, converged = train_func(
            [2, 2, 1], X, y,
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

        print(
            f"Seed {seed}: epochs={epoch}, error={error:.6f}, "
            f"accuracy={accuracy:.1f}%, MAE={mae:.6f}, converged={converged}"
        )

    return results



if __name__ == "__main__":

    c0, c1 = -10, 9

    X_raw = np.array([
        [c0, c0],
        [c0, c1],
        [c1, c0],
        [c1, c1]
    ])

    y_raw = np.array([
        [c0],
        [c1],
        [c1],
        [c0]
    ])

    X = normalize(X_raw, c0, c1)

    y_mse = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    y_bce = y_mse.copy()

    epochs = 200000
    lr = 0.01
    target_error = 0.01
    seeds = [1, 2, 3, 4, 5]

    print("\n========================================")
    print("MLP 2-2-1 + MSE")
    print("========================================")

    mse = run_experiments(
        train_network,
        X, y_mse, y_raw,
        seeds, epochs, lr, target_error, c0, c1
    )

    print("\n========================================")
    print("MLP 2-2-1 + BCE")
    print("========================================")

    bce = run_experiments(
        bce_train_network,
        X, y_bce, y_raw,
        seeds, epochs, lr, target_error, c0, c1
    )

    net_mlp_mse = mse["nets"][0]
    history_mlp_mse = mse["histories"][0]

    net_mlp_bce = bce["nets"][0]
    history_mlp_bce = bce["histories"][0]

    print("\n========================================")
    print("FINAL RESULTS (Seed 1)")
    print("========================================")

    print(
        f"MSE: epochs={mse['epochs'][0]}, "
        f"error={mse['errors'][0]:.6f}, "
        f"accuracy={mse['accuracies'][0]:.1f}%, "
        f"MAE={mse['maes'][0]:.6f}, "
        f"converged={mse['converged'][0]}"
    )

    print(
        f"BCE: epochs={bce['epochs'][0]}, "
        f"error={bce['errors'][0]:.6f}, "
        f"accuracy={bce['accuracies'][0]:.1f}%, "
        f"MAE={bce['maes'][0]:.6f}, "
        f"converged={bce['converged'][0]}"
    )


    print("\n========================================")
    print("5-RUN STATISTICS")
    print("========================================")

    for name, result in [("MSE", mse), ("BCE", bce)]:
        print(f"\n{name}:")
        print(f"Epochs: {result['epochs']}")
        print(f"Errors: {result['errors']}")
        print(f"Converged: {result['converged']}")
        print(f"Successful runs: {sum(result['converged'])}/5")

    plot_convergence(
        history_mlp_mse,
        history_mlp_bce,
        target_error
    )

    plot_epoch_scatter(
        mse["epochs"],
        mse["converged"],
        bce["epochs"],
        bce["converged"]
    )

    plot_decision_boundary(
        net_mlp_mse,
        "Decision surface: MLP 2-2-1 + MSE",
        c0, c1
    )

    plot_decision_boundary(
        net_mlp_bce,
        "Decision surface: MLP 2-2-1 + BCE",
        c0, c1
    )

    plt.figure(figsize=(8, 6))

    plt.bar(
        ["MLP + MSE", "MLP + BCE"],
        [mse["maes"][0], bce["maes"][0]]
    )

    plt.xlabel("Configuration")
    plt.ylabel("MAE")
    plt.title("Scale recovery accuracy [-10, 9]")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n========================================")
    print("TEST CASES")
    print("========================================")

    test_cases = [
        (-10, -10),
        (-10, 9),
        (9, -10),
        (9, 9),
        (0, 0),
        (5, -5),
        (-5, 5)
    ]

    for a, b in test_cases:

        print(f"\n--- Point A={a}, B={b} ---")

        print("\nMSE:")
        predict_point(net_mlp_mse, a, b, c0, c1)

        print("\nBCE:")
        predict_point(net_mlp_bce, a, b, c0, c1)

    print("\n========================================")
    print("INTERACTIVE MODE")
    print("========================================")
    print("Enter A and B from [-10; 10].")
    print("Enter 'q' to exit.")

    while True:

        user_input = input("\nInput A and B: ").strip()

        if user_input.lower() in ["q", "exit", "выход"]:
            print("Exit...")
            break

        try:
            parts = user_input.split()

            if len(parts) != 2:
                print("Error: enter exactly two numbers!")
                continue

            a, b = map(float, parts)

            if not (-10 <= a <= 10 and -10 <= b <= 10):
                print("Error: A and B must be in [-10; 10]!")
                continue

            print(f"\n--- Point A={a}, B={b} ---")

            print("\nMLP 2-2-1 + MSE:")
            predict_point(net_mlp_mse, a, b, c0, c1)

            print("\nMLP 2-2-1 + BCE:")
            predict_point(net_mlp_bce, a, b, c0, c1)

        except ValueError:
            print("Error: invalid input!")