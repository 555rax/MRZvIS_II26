import numpy as np
import matplotlib.pyplot as plt



c0 = 6.0
c1 = -4.0

# Обучающая выборка для XOR в исходной шкале
X = np.array([[ c0,  c0],
              [ c0,  c1],
              [ c1,  c0],
              [ c1,  c1]], dtype=float)

y = np.array([[c0],
              [c1],
              [c1],
              [c0]], dtype=float)

# Нормализация целевых значений в диапазон [0; 1]
min_y = min(c0, c1)
max_y = max(c0, c1)
range_y = max_y - min_y

def to_norm(y_orig):
    return (y_orig - min_y) / range_y

def from_norm(y_norm):
    return y_norm * range_y + min_y

y_norm = to_norm(y)



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv_from_output(sig_x):
    return sig_x * (1.0 - sig_x)

def relu(x):
    return np.maximum(0.0, x)

def relu_deriv(x):
    return (x > 0).astype(float)



class MLP221:
    def __init__(self, hidden_activation='sigmoid', lr=0.1, seed=None):
        self.hidden_activation = hidden_activation
        self.lr = lr

        if seed is not None:
            np.random.seed(seed)

        # Малая случайная инициализация весов
        self.W1 = np.random.uniform(-0.5, 0.5, (2, 2))
        self.b1 = np.zeros((1, 2))
        self.W2 = np.random.uniform(-0.5, 0.5, (2, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, x):
        h_in = x @ self.W1 + self.b1

        if self.hidden_activation == 'sigmoid':
            h_out = sigmoid(h_in)
        else:
            h_out = relu(h_in)

        o_in = h_out @ self.W2 + self.b2
        o_out = sigmoid(o_in)

        return h_in, h_out, o_in, o_out

    def predict_norm(self, x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return self.forward(x)[3].item()

    def predict_rescaled(self, x):
        return from_norm(self.predict_norm(x))

    def predict_norm_batch(self, X_batch):
        h_in = X_batch @ self.W1 + self.b1

        if self.hidden_activation == 'sigmoid':
            h_out = sigmoid(h_in)
        else:
            h_out = relu(h_in)

        o_in = h_out @ self.W2 + self.b2
        o_out = sigmoid(o_in)

        return o_out.ravel()

    def train_epoch_online(self, X_train, y_train_norm):
        total_loss = 0.0

        for xi, yi in zip(X_train, y_train_norm):
            xi = xi.reshape(1, -1)
            yi = yi.reshape(1, -1)

            h_in, h_out, o_in, o_out = self.forward(xi)

            eps = 1e-12
            o_clip = np.clip(o_out, eps, 1.0 - eps)

            # BCE для одного выхода
            sample_loss = -(yi * np.log(o_clip) + (1.0 - yi) * np.log(1.0 - o_clip))
            total_loss += sample_loss.item()

            # Для BCE + Sigmoid на выходе: delta_out = o_out - yi
            delta_out = o_out - yi

            grad_W2 = h_out.T @ delta_out
            grad_b2 = delta_out

            if self.hidden_activation == 'sigmoid':
                delta_hidden = (delta_out @ self.W2.T) * sigmoid_deriv_from_output(h_out)
            else:
                delta_hidden = (delta_out @ self.W2.T) * relu_deriv(h_in)

            grad_W1 = xi.T @ delta_hidden
            grad_b1 = delta_hidden

            # Онлайн-обновление
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1

        return total_loss

    def count_dead_neurons(self, X_data):
        """Считает нейроны ReLU, которые выдают 0 на всех обучающих примерах."""
        if self.hidden_activation != 'relu':
            return 0

        h_in = X_data @ self.W1 + self.b1
        h_out = relu(h_in)
        dead = np.all(h_out == 0.0, axis=0)
        return int(np.sum(dead))



def run_experiments(seeds, config, lr=0.1, max_epochs=20000, Ee=0.01):
    results = []

    for seed in seeds:
        hidden = 'sigmoid' if config == 'A' else 'relu'
        model = MLP221(hidden_activation=hidden, lr=lr, seed=seed)

        history = []
        reached = False
        epochs_done = 0

        for epoch in range(max_epochs):
            Es = model.train_epoch_online(X, y_norm)
            history.append(Es)
            epochs_done = epoch + 1

            if Es <= Ee:
                reached = True
                break

        outputs_norm = np.array([model.predict_norm(xi) for xi in X]).reshape(-1, 1)
        outputs_rescaled = from_norm(outputs_norm)

        preds_class = np.where(outputs_norm >= 0.5, c0, c1)
        accuracy = np.mean(preds_class == y)
        mae = np.mean(np.abs(outputs_rescaled - y))
        dead = model.count_dead_neurons(X)

        results.append({
            'seed': seed,
            'config': config,
            'reached': reached,
            'epochs': epochs_done if reached else max_epochs,
            'Es_final': Es,
            'accuracy': float(accuracy),
            'mae': float(mae),
            'dead_neurons': dead,
            'history': history,
            'model': model,
            'outputs_norm': outputs_norm,
            'outputs_rescaled': outputs_rescaled
        })

    return results

def pick_representative(results):
    for r in results:
        if r['reached'] and r['accuracy'] == 1.0:
            return r
    for r in results:
        if r['reached']:
            return r
    return results[0]

def print_table(res_A, res_B):
    print("\n" + "=" * 110)
    print("Таблица сравнения (5 запусков)")
    print("=" * 110)
    print(f"{'Config':<7} {'Seed':<5} {'Reached':<8} {'Epochs':<8} "
          f"{'Es_final':<12} {'Accuracy':<9} {'MAE':<10} {'Dead':<5}")
    print("-" * 110)

    for res in (res_A, res_B):
        for r in res:
            print(f"{r['config']:<7} {r['seed']:<5} {str(r['reached']):<8} "
                  f"{r['epochs']:<8} {r['Es_final']:<12.6f} "
                  f"{r['accuracy']:<9.3f} {r['mae']:<10.4f} "
                  f"{r['dead_neurons']:<5}")

    print("-" * 110)
    for name, res in [('A (Sigmoid)', res_A), ('B (ReLU)', res_B)]:
        epochs = [r['epochs'] for r in res]
        mae = [r['mae'] for r in res]
        acc = [r['accuracy'] for r in res]
        dead_total = sum(r['dead_neurons'] for r in res)

        print(f"{name}: "
              f"mean epochs={np.mean(epochs):.1f}±{np.std(epochs):.1f}, "
              f"mean MAE={np.mean(mae):.4f}, "
              f"mean accuracy={np.mean(acc):.3f}, "
              f"dead neurons total={dead_total}")



seeds = [1, 2, 3, 4, 5]
lr = 0.1
max_epochs = 20000
Ee = 0.01

res_A = run_experiments(seeds, config='A', lr=lr, max_epochs=max_epochs, Ee=Ee)
res_B = run_experiments(seeds, config='B', lr=lr, max_epochs=max_epochs, Ee=Ee)

print_table(res_A, res_B)

rep_A = pick_representative(res_A)
rep_B = pick_representative(res_B)

print(f"\nПредставители: A seed={rep_A['seed']}, B seed={rep_B['seed']}")



plt.figure(figsize=(10, 5))
plt.plot(rep_A['history'], label=f"A (Sigmoid), seed={rep_A['seed']}")
plt.plot(rep_B['history'], label=f"B (ReLU), seed={rep_B['seed']}")
plt.axhline(Ee, color='red', linestyle='--', label=f'Ee = {Ee}')
plt.xlabel("Эпоха")
plt.ylabel("Суммарная ошибка Es (BCE)")
plt.title("График сходимости: Sigmoid vs ReLU")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(8, 4))
indices = np.arange(len(seeds))
width = 0.35

epochs_A = [r['epochs'] for r in res_A]
epochs_B = [r['epochs'] for r in res_B]

bars1 = ax.bar(indices - width / 2, epochs_A, width, label='A (Sigmoid)')
bars2 = ax.bar(indices + width / 2, epochs_B, width, label='B (ReLU)')

for i, r in enumerate(res_A):
    if not r['reached']:
        bars1[i].set_hatch('//')

for i, r in enumerate(res_B):
    if not r['reached']:
        bars2[i].set_hatch('\\\\')

ax.set_xticks(indices)
ax.set_xticklabels([str(s) for s in seeds])
ax.set_xlabel("seed")
ax.set_ylabel("Эпохи")
ax.set_title("Разброс числа эпох по 5 запускам")
ax.legend()
plt.tight_layout()
plt.show()



def plot_decision_surface(model, title):
    grid_n = 200
    xs = np.linspace(-10, 10, grid_n)
    ys = np.linspace(-10, 10, grid_n)
    XX, YY = np.meshgrid(xs, ys)

    pts = np.c_[XX.ravel(), YY.ravel()]
    outs_norm = model.predict_norm_batch(pts).reshape(grid_n, grid_n)
    outs_rescaled = from_norm(outs_norm)

    plt.figure(figsize=(6, 5))
    plt.contourf(XX, YY, outs_rescaled, levels=50, cmap='RdYlBu', alpha=0.9)

    for xi, yi in zip(X, y):
        if yi[0] == c0:
            plt.scatter(xi[0], xi[1], c='black', marker='o',
                        edgecolors='white', s=80, label='c0' if xi[0] == c0 and xi[1] == c0 else "")
        else:
            plt.scatter(xi[0], xi[1], c='white', marker='s',
                        edgecolors='black', s=80, label='c1' if xi[0] == c0 and xi[1] == c1 else "")

    plt.xlabel("A")
    plt.ylabel("B")
    plt.title(title)
    plt.colorbar(label="Выход в исходной шкале")
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.tight_layout()
    plt.show()

plot_decision_surface(rep_A['model'], f"Разделяющая поверхность — A (Sigmoid), seed={rep_A['seed']}")
plot_decision_surface(rep_B['model'], f"Разделяющая поверхность — B (ReLU), seed={rep_B['seed']}")



mae_A = np.mean([r['mae'] for r in res_A])
mae_B = np.mean([r['mae'] for r in res_B])

plt.figure(figsize=(5, 4))
plt.bar(['A (Sigmoid)', 'B (ReLU)'], [mae_A, mae_B], color=['blue', 'orange'])
plt.ylabel("MAE в исходной шкале")
plt.title("Сравнение средней абсолютной ошибки")
plt.tight_layout()
plt.show()



def print_prediction(model, a, b, config_name):
    x = np.array([[a, b]], dtype=float)
    out_norm = model.predict_norm(x)
    out_res = from_norm(out_norm)
    cls = c0 if out_norm >= 0.5 else c1
    print(f"{config_name}: output_norm={out_norm:.6f}, "
          f"output_res={out_res:.4f}, class={cls}")

def demo_mode(model_A, model_B):
    print("\nДемонстрация на обучающих примерах и дополнительных парах:")
    test_inputs = [
        [c0, c0], [c0, c1], [c1, c0], [c1, c1],
        [0, 0], [3, -7], [5, 2]
    ]

    for a, b in test_inputs:
        print(f"\nВход (A, B) = ({a}, {b})")
        print_prediction(model_A, a, b, "A (Sigmoid)")
        print_prediction(model_B, a, b, "B (ReLU)")

def interactive_mode(model_A, model_B):
    print("\nРежим функционирования.")
    print("Введите A и B из [-10; 10] через пробел (или 'q' для выхода):")

    while True:
        try:
            line = input("A B > ").strip()
        except EOFError:
            break

        if line.lower() == 'q':
            break

        parts = line.split()
        if len(parts) != 2:
            print("Введите два числа.")
            continue

        try:
            a, b = map(float, parts)
        except ValueError:
            print("Ошибка: введите числа.")
            continue

        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("Значения должны быть в диапазоне [-10; 10].")
            continue

        print_prediction(model_A, a, b, "A (Sigmoid)")
        print_prediction(model_B, a, b, "B (ReLU)")


demo_mode(rep_A['model'], rep_B['model'])

