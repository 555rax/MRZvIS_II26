import numpy as np
import matplotlib.pyplot as plt

C0 = 2.0
C1 = 10.0
RANGE_MIN = -10.0
RANGE_MAX = 10.0

# Таблица истинности для XOR (Вариант 2)
X_raw = np.array([
    [C0, C0],  # 2, 2
    [C0, C1],  # 2, 10
    [C1, C0],  # 10, 2
    [C1, C1]  # 10, 10
], dtype=float)

# Целевые значения (A XOR B в исходной шкале)
Y_raw = np.array([
    [C0],  # 2 XOR 2 = 2
    [C1],  # 2 XOR 10 = 10
    [C1],  # 10 XOR 2 = 10
    [C0]  # 10 XOR 10 = 2
], dtype=float)


# Нормализация входов в диапазон [0, 1]
def normalize(v):
    return (v - RANGE_MIN) / (RANGE_MAX - RANGE_MIN)


# ИСПРАВЛЕНО: Правильное обратное преобразование из [0, 1] в [C0, C1]
def to_original(a):
    return C0 + a * (C1 - C0)


X = normalize(X_raw)
# Нормализация целей: C0 -> 1.0, C1 -> 0.0
Y_bin = np.array([[1.0] if y[0] == C0 else [0.0] for y in Y_raw])


# ==========================================
# 2. ФУНКЦИИ АКТИВАЦИИ И ПОТЕРЬ
# ==========================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(a):
    return a * (1.0 - a)


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def bce_loss(a, y):
    eps = 1e-9
    a = np.clip(a, eps, 1.0 - eps)
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))


# ==========================================
# 3. КЛАСС НЕЙРОСЕТИ (MLP 2-2-1)
# ==========================================
class MLP:
    def __init__(self, hidden_activation="sigmoid", lr=0.5, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden_activation = hidden_activation
        self.lr = lr

        # Инициализация весов (2-2-1)
        self.W1 = rng.normal(0, 0.5, (2, 2))
        self.b1 = np.zeros((1, 2))
        self.W2 = rng.normal(0, 0.5, (2, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, x):
        x = x.reshape(1, -1)
        self.x = x
        self.z1 = x @ self.W1 + self.b1

        if self.hidden_activation == "relu":
            self.a1 = relu(self.z1)
        else:
            self.a1 = sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, target):
        target = target.reshape(1, -1)
        delta_out = self.a2 - target

        if self.hidden_activation == "relu":
            delta_hid = (delta_out @ self.W2.T) * relu_deriv(self.z1)
        else:
            delta_hid = (delta_out @ self.W2.T) * sigmoid_deriv(self.a1)

        self.W2 -= self.lr * self.a1.T @ delta_out
        self.b2 -= self.lr * delta_out
        self.W1 -= self.lr * self.x.T @ delta_hid
        self.b1 -= self.lr * delta_hid

    def train(self, X, Y, Ee, max_epochs, record_history=False):
        history = [] if record_history else None
        for epoch in range(1, max_epochs + 1):
            Es = 0.0
            order = np.random.permutation(len(X))
            for i in order:
                out = self.forward(X[i])
                Es += bce_loss(out, Y[i]).item()
                self.backward(Y[i])

            if record_history:
                history.append(Es)
            if Es <= Ee:
                return epoch, Es, history
        return max_epochs, Es, history

    def predict(self, x):
        return self.forward(np.array(x)).flatten()[0]


# ==========================================
# 4. ОЦЕНКА И ОБУЧЕНИЕ
# ==========================================
def evaluate(model):
    correct = 0
    abs_errors = []
    for i in range(len(X)):
        a = model.predict(X[i])
        y_orig_pred = to_original(a)

        # Если выход >= 0.5, то это класс C0, иначе C1
        pred_class = C0 if a >= 0.5 else C1
        target_class = Y_raw[i][0]

        correct += (pred_class == target_class)
        abs_errors.append(abs(y_orig_pred - target_class))

    accuracy = correct / len(X)
    mae = np.mean(abs_errors)
    return accuracy, mae


Ee = 0.01
MAX_EPOCHS = 15000  # Для Варианта 2 достаточно 15к
LR = 0.5
SEEDS = [0, 1, 2, 3, 4]

results_A = []
results_B = []
history_A0 = None
history_B0 = None
model_A0 = None
model_B0 = None

print("Идет обучение, пожалуйста подождите...")

for seed in SEEDS:
    # --- Конфигурация А (Sigmoid) ---
    model_A = MLP(hidden_activation="sigmoid", lr=LR, seed=seed)
    epochs_A, Es_A, hist_A = model_A.train(X, Y_bin, Ee, MAX_EPOCHS, record_history=(seed == 0))
    acc_A, mae_A = evaluate(model_A)
    results_A.append((epochs_A, Es_A, acc_A, mae_A))

    if seed == 0:
        history_A0 = hist_A
        model_A0 = model_A

    # --- Конфигурация В (ReLU) ---
    model_B = MLP(hidden_activation="relu", lr=LR, seed=seed)
    epochs_B, Es_B, hist_B = model_B.train(X, Y_bin, Ee, MAX_EPOCHS, record_history=(seed == 0))
    acc_B, mae_B = evaluate(model_B)
    results_B.append((epochs_B, Es_B, acc_B, mae_B))

    if seed == 0:
        history_B0 = hist_B
        model_B0 = model_B


def print_results(name, results):
    print(f"\n{name}")
    for i, (ep, es, acc, mae) in enumerate(results):
        print(f"seed={i} | эпох={ep:>5} | Es={es:.5f} | acc={acc * 100:>3.0f}% | MAE={mae:.3f}")
    epochs_arr = np.array([r[0] for r in results])
    acc_arr = np.array([r[2] for r in results])
    mae_arr = np.array([r[3] for r in results])
    print(
        f"Среднее: эпох={epochs_arr.mean():.0f} (std={epochs_arr.std():.0f}) | acc={acc_arr.mean() * 100:.0f}% | MAE={mae_arr.mean():.3f}")


print_results("Конфигурация А (Sigmoid)", results_A)
print_results("Конфигурация В (ReLU)", results_B)

# ==========================================
# 5. ПОСТРОЕНИЕ ГРАФИКОВ
# ==========================================
# График сходимости
plt.figure(figsize=(8, 5))
plt.plot(history_A0, label="Sigmoid (Конфигурация А)", color='tab:blue')
plt.plot(history_B0, label="ReLU (Конфигурация В)", color='tab:orange')
plt.xlabel("Эпоха")
plt.ylabel("Суммарная ошибка Es (BCE)")
plt.title("Сходимость обучения (Вариант 2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Диаграмма разброса эпох
epochs_A_list = [r[0] for r in results_A]
epochs_B_list = [r[0] for r in results_B]
x_idx = np.arange(len(SEEDS))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x_idx - width / 2, epochs_A_list, width, label="Sigmoid (А)", color='tab:blue')
plt.bar(x_idx + width / 2, epochs_B_list, width, label="ReLU (В)", color='tab:orange')
plt.xticks(x_idx, [f"seed {s}" for s in SEEDS])
plt.ylabel("Эпох до сходимости")
plt.title("Число эпох по 5 запускам")
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================
# 6. ВИЗУАЛИЗАЦИЯ РАЗДЕЛЯЮЩЕЙ ПОВЕРХНОСТИ
# ==========================================
def plot_heatmap(model, title):
    grid = np.linspace(RANGE_MIN, RANGE_MAX, 100)
    AA, BB = np.meshgrid(grid, grid)
    Z = np.zeros_like(AA)

    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
            x_norm = normalize(np.array([AA[i, j], BB[i, j]]))
            Z[i, j] = model.predict(x_norm)

    plt.figure(figsize=(6, 5))
    plt.contourf(AA, BB, Z, levels=50, cmap="coolwarm")
    plt.colorbar(label="Выход сети (норм.)")
    plt.scatter(X_raw[:, 0], X_raw[:, 1], c="black", marker="x", s=100, label="Обучающие точки")
    plt.xlabel("A")
    plt.ylabel("B")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_heatmap(model_A0, "Разделяющая поверхность (Sigmoid, А)")
plot_heatmap(model_B0, "Разделяющая поверхность (ReLU, В)")


# ==========================================
# 7. РЕЖИМ ФУНКЦИОНИРОВАНИЯ
# ==========================================
def run_network(model, A, B):
    x_norm = normalize(np.array([A, B]))
    a_norm = model.predict(x_norm)
    y_orig = to_original(a_norm)
    cls = C0 if a_norm >= 0.5 else C1
    return a_norm, y_orig, cls


print("\n" + "=" * 50)
print("РЕЖИМ ФУНКЦИОНИРОВАНИЯ (Демонстрация на ReLU модели)")
print("=" * 50)
test_pairs = [(2, 2), (2, 10), (10, 2), (10, 10), (5, 5), (6, 10), (-5, 5)]
for A, B in test_pairs:
    a_norm, y_orig, cls = run_network(model_B0, A, B)
    print(f"A={A:>4} B={B:>4} | выход norm={a_norm:.3f} | выход [{C0};{C1}]={y_orig:.3f} | класс = {cls}")

print("\n" + "=" * 50)
print("ИНТЕРАКТИВНЫЙ РЕЖИМ (Введите 'q' для выхода)")
print("=" * 50)
while True:
    a_in = input("Введите A: ").strip()
    if a_in.lower() == 'q': break
    b_in = input("Введите B: ").strip()
    if b_in.lower() == 'q': break
    try:
        A = float(a_in)
        B = float(b_in)
        if not (RANGE_MIN <= A <= RANGE_MAX) or not (RANGE_MIN <= B <= RANGE_MAX):
            print(f"Внимание: значения должны быть в диапазоне [{RANGE_MIN}; {RANGE_MAX}]")
        a_norm, y_orig, cls = run_network(model_B0, A, B)
        print(f"Выход norm={a_norm:.3f} | выход [{C0};{C1}]={y_orig:.3f} | класс = {cls}\n")
    except ValueError:
        print("Некорректный ввод. Введите число.\n")