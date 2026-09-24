import matplotlib.pyplot as plt
import numpy as np
from mlp import MLP
from slp import SLP

C0 = 5.0
C1 = -2.0

X_raw = np.array([[5, 5], [5, -2], [-2, 5], [-2, -2]], dtype=np.float64)
y_raw = np.array([5, -2, -2, 5], dtype=np.float64)

def normalize_x(X):
    return (X - C1) / (C0 - C1)

y_norm = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
X_norm = normalize_x(X_raw)

def evaluate(model, X_norm_data, y_raw_data):
    correct = 0
    preds = []
    for i in range(len(X_norm_data)):
        out = model.forward(X_norm_data[i])
        pred_class = C0 if out < 0.5 else C1
        preds.append(pred_class)
        if pred_class == y_raw_data[i]:
            correct += 1
    acc = (correct / len(y_raw_data)) * 100.0
    return acc, preds

def plot_results(slp_errors, mlp_errors, mlp_model):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(slp_errors, label="SLP", color="red", linestyle="--")
    plt.plot(mlp_errors, label="MLP", color="blue")
    plt.axhline(
        y=1e-3, color="green", linestyle=":", label="Stop.Crit. (1e-3)"
    )
    plt.yscale("log")
    plt.title("Es - Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("(MSE, log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    x_range = np.linspace(-10, 10, 200)
    y_range = np.linspace(-10, 10, 200)
    xx, yy = np.meshgrid(x_range, y_range)

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_norm = normalize_x(grid_points)

    preds_grid = np.array([mlp_model.forward(p) for p in grid_norm])
    preds_grid = preds_grid.reshape(xx.shape)

    contour = plt.contourf(
        xx, yy, preds_grid, levels=50, cmap="coolwarm", alpha=0.8
    )
    plt.colorbar(contour, label="Sigmoid output")

    for i in range(len(X_raw)):
        color = "cyan" if y_raw[i] == C0 else "yellow"
        plt.scatter(
            X_raw[i, 0],
            X_raw[i, 1],
            color=color,
            edgecolors="black",
            s=120,
            zorder=5,
            label=f"Dot ({X_raw[i,0]}, {X_raw[i,1]}) -> {y_raw[i]}",
        )

    plt.title("Sep.Surface [-10, 10]")
    plt.xlabel("Class A")
    plt.ylabel("Class B")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    print("=" * 60)
    print("1. SLP training...")
    slp = SLP()
    slp_epochs, slp_err, slp_errors = slp.train(
        X_norm, y_norm, max_epochs=20000, target_error=1e-3
    )
    slp_acc, slp_preds = evaluate(slp, X_norm, y_raw)

    print("\n2. MLP training...")
    max_attempts = 10
    mlp = None
    mlp_epochs, mlp_err, mlp_errors = 0, 0.0, []

    for attempt in range(max_attempts):
        mlp_candidate = MLP()
        epochs, err, errors = mlp_candidate.train(
            X_norm, y_norm, max_epochs=30000, target_error=1e-3
        )
        if err <= 1e-3:
            mlp = mlp_candidate
            mlp_epochs, mlp_err, mlp_errors = epochs, err, errors
            break

    if mlp is None:
        mlp = mlp_candidate
        mlp_epochs, mlp_err, mlp_errors = epochs, err, errors

    mlp_acc, mlp_preds = evaluate(mlp, X_norm, y_raw)

    print("\n" + "=" * 60)
    print(f"{'Criterion':<25} | {'SLP':<20} | {'MLP':<20}")
    print("-" * 70)
    print(f"{'Epochs':<25} | {slp_epochs:<20} | {mlp_epochs:<20}")
    print(f"{'Final MSE':<25} | {slp_err:<20.6f} | {mlp_err:<20.6f}")
    print(f"{'Accuracy (%)':<25} | {slp_acc:<20.1f} | {mlp_acc:<20.1f}")
    print(f"{'Predictions':<25} | {str(slp_preds):<20} | {str(mlp_preds):<20}")

    plot_results(slp_errors, mlp_errors, mlp)

    print("\n" + "=" * 60)

    while True:
        user_input = input("\nEnter A and B in [-10, 10]: ").strip()
        if user_input.lower() == "q":
            break

        parts = user_input.split()

        a, b = float(parts[0]), float(parts[1])
        x_user = np.array([a, b], dtype=np.float64)
        x_user_norm = normalize_x(x_user)

        prob = mlp.forward(x_user_norm)
        pred_class = C0 if prob < 0.5 else C1

        print(f"Sigmoid: {prob:.4f}")
        print(f"Prediction: {pred_class}")

if __name__ == "__main__":
    main()