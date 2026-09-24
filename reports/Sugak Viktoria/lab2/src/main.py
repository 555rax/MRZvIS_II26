import numpy as np
import matplotlib.pyplot as plt

c0 = 7
c1 = -7
Ee = 0.02
MAX_EPOCHS = 10000
SEEDS = [42, 43, 44, 45, 46]

X = np.array([
    [7, 7],
    [7, -7],
    [-7, 7],
    [-7, -7]
], dtype=float)

y = np.array([7, -7, -7, 7], dtype=float)
t = np.array([0, 1, 1, 0], dtype=float)

def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def restore_scale(o):
    return c0 + (c1 - c0) * o

def classify(o):
    return (o >= 0.5).astype(int)

def forward(x, weights):
    W1, b1, W2, b2 = weights
    h = sigmoid(x @ W1 + b1)
    o = float(sigmoid(h @ W2 + b2))
    return h, o

def total_error(o, loss):
    if loss == "BCE":
        eps = 1e-12
        return float(
            -np.sum(
                t * np.log(np.clip(o, eps, 1.0)) +
                (1 - t) * np.log(np.clip(1 - o, eps, 1.0))
            )
        )

    y_hat = restore_scale(o)
    return float(0.5 * np.sum((y_hat - y) ** 2))

def train(seed, loss, learning_rate):
    rng = np.random.default_rng(seed)

    W1 = rng.normal(0, 0.5, (2, 2))
    b1 = rng.normal(0, 0.5, 2)
    W2 = rng.normal(0, 0.5, 2)
    b2 = float(rng.normal(0, 0.5))

    history = []
    X_train = X / 10.0

    for epoch in range(1, MAX_EPOCHS + 1):
        for i in rng.permutation(4):
            h = sigmoid(X_train[i] @ W1 + b1)
            o = float(sigmoid(h @ W2 + b2))

            if loss == "BCE":
                delta2 = o - t[i]
            else:
                y_hat = restore_scale(o)
                delta2 = (y_hat - y[i]) * (c1 - c0) * o * (1 - o)

            delta1 = (delta2 * W2) * h * (1 - h)

            W2 -= learning_rate * delta2 * h
            b2 -= learning_rate * delta2
            W1 -= learning_rate * np.outer(X_train[i], delta1)
            b1 -= learning_rate * delta1

        weights = (W1, b1, W2, b2)
        outputs = np.array([
            forward(X_train[i], weights)[1]
            for i in range(4)
        ])

        error = total_error(outputs, loss)
        history.append(error)

        if error <= Ee:
            break

    outputs = np.array([
        forward(X_train[i], weights)[1]
        for i in range(4)
    ])

    y_hat = restore_scale(outputs)
    predicted = classify(outputs)
    accuracy = float(np.mean(predicted == t))
    mae = float(np.mean(np.abs(y_hat - y)))

    return {
        "seed": seed,
        "loss": loss,
        "epochs": epoch,
        "error": float(history[-1]),
        "accuracy": accuracy,
        "mae": mae,
        "history": np.array(history),
        "weights": weights,
        "outputs": outputs,
        "y_hat": y_hat,
        "converged": history[-1] <= Ee
    }

mse_results = [
    train(seed, "MSE", 0.1)
    for seed in SEEDS
]

bce_results = [
    train(seed, "BCE", 1.0)
    for seed in SEEDS
]

mse_rep = train(43, "MSE", 0.1)
bce_rep = train(43, "BCE", 1.0)

print("ВАРИАНТ 4: c0 = 7, c1 = -7")
print("Ee =", Ee)
print()

print("РЕПРЕЗЕНТАТИВНЫЕ ЗАПУСКИ, seed = 43")
print(
    "MSE:",
    mse_rep["epochs"],
    mse_rep["error"],
    mse_rep["accuracy"],
    mse_rep["mae"]
)
print(
    "BCE:",
    bce_rep["epochs"],
    bce_rep["error"],
    bce_rep["accuracy"],
    bce_rep["mae"]
)
print()

print("СЕРИЯ ИЗ 5 ЗАПУСКОВ")

print("MSE")
for r in mse_results:
    print(
        r["seed"],
        r["epochs"],
        r["error"],
        r["accuracy"],
        r["mae"],
        r["converged"]
    )

print("BCE")
for r in bce_results:
    print(
        r["seed"],
        r["epochs"],
        r["error"],
        r["accuracy"],
        r["mae"],
        r["converged"]
    )

print()
print("ИТОГОВЫЕ ПОКАЗАТЕЛИ")

for name, results in [
    ("MSE", mse_results),
    ("BCE", bce_results)
]:
    converged = [
        r["epochs"]
        for r in results
        if r["converged"]
    ]

    print(
        name,
        "сошлись:",
        len(converged),
        "из",
        len(results)
    )

    print(
        name,
        "среднее число эпох:",
        np.mean(converged) if converged else np.nan
    )

    print(
        name,
        "разброс эпох:",
        (min(converged), max(converged))
        if converged else "нет"
    )

print()
print("ПРОВЕРКА РЕЖИМА ФУНКЦИОНИРОВАНИЯ")

test_points = [
    (7, 7),
    (7, -7),
    (-7, 7),
    (-7, -7),
    (0, 0),
    (5, -3),
    (-2, 4)
]

for a, b in test_points:
    h, o = forward(
        np.array([a, b], dtype=float) / 10.0,
        bce_rep["weights"]
    )

    real = restore_scale(o)

    cls = (
        c1
        if abs(real - c1) < abs(real - c0)
        else c0
    )

    print(
        (a, b),
        round(o, 4),
        round(real, 4),
        cls
    )

fig1 = plt.figure(figsize=(9, 6))

plt.plot(
    np.arange(1, len(mse_rep["history"]) + 1),
    mse_rep["history"],
    label="MSE"
)

plt.plot(
    np.arange(1, len(bce_rep["history"]) + 1),
    bce_rep["history"],
    label="BCE"
)

plt.axhline(
    Ee,
    linestyle="--",
    label="Порог Ee"
)

plt.xlabel("Номер эпохи")
plt.ylabel("Суммарная ошибка Es")
plt.title("Сходимость MSE и BCE")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig("01_convergence.png", dpi=200)

fig2 = plt.figure(figsize=(9, 6))

x = np.arange(len(SEEDS))
width = 0.36

mse_epochs = [
    r["epochs"]
    if r["converged"]
    else MAX_EPOCHS
    for r in mse_results
]

bce_epochs = [
    r["epochs"]
    if r["converged"]
    else MAX_EPOCHS
    for r in bce_results
]

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

plt.xticks(
    x,
    [str(s) for s in SEEDS]
)

plt.xlabel("Seed")
plt.ylabel("Число эпох")
plt.title("Разброс числа эпох по пяти запускам")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig2.savefig("02_epochs.png", dpi=200)

def plot_surface(result, title, filename):
    grid = np.linspace(-10, 10, 180)
    A, B = np.meshgrid(grid, grid)

    points = np.column_stack([
        A.ravel(),
        B.ravel()
    ]) / 10.0

    values = np.array([
        forward(p, result["weights"])[1]
        for p in points
    ])

    Z = restore_scale(values).reshape(A.shape)

    fig = plt.figure(figsize=(8, 6))

    contour = plt.contourf(
        A,
        B,
        Z,
        levels=30
    )

    plt.contour(
        A,
        B,
        Z,
        levels=[0],
        linewidths=2
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=t,
        edgecolors="black",
        s=90,
        label="Обучающие примеры"
    )

    plt.colorbar(
        contour,
        label="Выход сети в шкале [c0; c1]"
    )

    plt.xlabel("A")
    plt.ylabel("B")
    plt.title(title)
    plt.legend()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.tight_layout()
    fig.savefig(filename, dpi=200)

    return fig

fig3 = plot_surface(
    mse_rep,
    "Разделяющая поверхность MSE",
    "03_surface_mse.png"
)

fig4 = plot_surface(
    bce_rep,
    "Разделяющая поверхность BCE",
    "04_surface_bce.png"
)

fig5 = plt.figure(figsize=(8, 6))

plt.bar(
    ["MSE", "BCE"],
    [
        mse_rep["mae"],
        bce_rep["mae"]
    ]
)

plt.xlabel("Конфигурация")
plt.ylabel("Средняя абсолютная ошибка")
plt.title("Сравнение точности восстановления шкалы")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig5.savefig("05_mae.png", dpi=200)

plt.show()