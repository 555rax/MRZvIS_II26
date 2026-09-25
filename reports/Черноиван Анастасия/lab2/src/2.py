import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

C0, C1 = -8.0, 3.0

X_raw = np.array([
    [-8.0, -8.0],
    [-8.0,  3.0],
    [ 3.0, -8.0],
    [ 3.0,  3.0],
], dtype=float)

Y_raw = np.array([-8.0, 3.0, 3.0, -8.0], dtype=float)

def normalize(t):
    return (t - C0) / (C1 - C0)


def denormalize(o):
    return C0 + (C1 - C0) * o


X = normalize(X_raw)
Y = normalize(Y_raw)

LR = 5.0
MAX_EPOCHS = 5000

EE_MSE = 0.01
EE_BCE = 0.05

SEEDS = [0, 1, 2, 3, 4]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(o):
    return o * (1.0 - o)

class MLP:
    def __init__(self, loss="mse", lr=LR, seed=None):
        rng = np.random.RandomState(seed)
        self.W1 = rng.uniform(-0.5, 0.5, (2, 2))
        self.b1 = rng.uniform(-0.5, 0.5, 2)
        self.W2 = rng.uniform(-0.5, 0.5, 2)
        self.b2 = rng.uniform(-0.5, 0.5)
        self.loss = loss
        self.lr = lr

    def forward(self, x):
        z1 = self.W1 @ x + self.b1
        h = sigmoid(z1)
        z2 = self.W2 @ h + self.b2
        o = sigmoid(z2)
        return h, o

    def _loss_and_delta(self, o, t):
        if self.loss == "mse":
            err = t - o
            sample_loss = 0.5 * err ** 2
            delta_out = err * sigmoid_deriv(o)
        else:
            eps = 1e-9
            o_c = np.clip(o, eps, 1 - eps)
            sample_loss = -(t * np.log(o_c) + (1 - t) * np.log(1 - o_c))
            delta_out = t - o
        return sample_loss, delta_out

    def train_epoch(self, X, Y):
        total = 0.0
        for i in range(len(X)):
            x, t = X[i], Y[i]
            h, o = self.forward(x)
            sample_loss, delta_out = self._loss_and_delta(o, t)
            total += sample_loss

            grad_W2 = delta_out * h
            grad_b2 = delta_out
            delta_hidden = delta_out * self.W2 * sigmoid_deriv(h)
            grad_W1 = np.outer(delta_hidden, x)
            grad_b1 = delta_hidden

            self.W2 += self.lr * grad_W2
            self.b2 += self.lr * grad_b2
            self.W1 += self.lr * grad_W1
            self.b1 += self.lr * grad_b1
        return total

    def predict(self, x):
        _, o = self.forward(x)
        return o


def train_network(loss, seed, ee, max_epochs=MAX_EPOCHS):
    net = MLP(loss=loss, seed=seed)
    history = []
    converged_epoch = None
    for epoch in range(1, max_epochs + 1):
        Es = net.train_epoch(X, Y)
        history.append(Es)
        if Es <= ee and converged_epoch is None:
            converged_epoch = epoch
            break
    return net, converged_epoch, history


def evaluate(net):
    correct = 0
    abs_errors = []
    for i in range(len(X)):
        o = net.predict(X[i])
        y_pred_real = denormalize(o)
        y_true_real = Y_raw[i]
        pred_class = C0 if abs(y_pred_real - C0) < abs(y_pred_real - C1) else C1
        if pred_class == y_true_real:
            correct += 1
        abs_errors.append(abs(y_pred_real - y_true_real))
    accuracy = correct / len(X)
    mae = float(np.mean(abs_errors))
    return accuracy, mae

def run_series(loss, ee):
    results = []
    for seed in SEEDS:
        net, conv_epoch, history = train_network(loss, seed, ee)
        Es_final = history[-1]
        accuracy, mae = evaluate(net)
        results.append({
            "seed": seed,
            "converged": conv_epoch is not None,
            "epochs": conv_epoch if conv_epoch is not None else len(history),
            "Es_final": Es_final,
            "accuracy": accuracy,
            "mae": mae,
            "net": net,
            "history": history,
        })
    return results


print("Обучение конфигурации А (MSE)...")
results_mse = run_series("mse", EE_MSE)
print("Обучение конфигурации Б (BCE)...")
results_bce = run_series("bce", EE_BCE)


def print_table(name, results, ee):
    print(f"\n=== {name} ===")
    n_converged = sum(r["converged"] for r in results)
    for r in results:
        status = f"{r['epochs']} эпох" if r["converged"] else f"НЕ сошёлся за {r['epochs']}"
        print(f"seed={r['seed']}: {status}, Es={r['Es_final']:.6f}, "
              f"accuracy={r['accuracy']*100:.1f}%, MAE={r['mae']:.3f}")
    epochs_conv = [r["epochs"] for r in results if r["converged"]]
    print(f"Сошлось: {n_converged}/{len(results)} запусков "
          f"(Ee={ee}); "
          f"разброс эпох: {epochs_conv if epochs_conv else '—'}")


print_table("Конфигурация А (MSE)", results_mse, EE_MSE)
print_table("Конфигурация Б (BCE)", results_bce, EE_BCE)

def summarize(results):
    conv = [r for r in results if r["converged"]]
    base = conv if conv else results
    return {
        "epochs_mean": np.mean([r["epochs"] for r in base]),
        "Es_mean": np.mean([r["Es_final"] for r in base]),
        "accuracy_mean": np.mean([r["accuracy"] for r in base]) * 100,
        "mae_mean": np.mean([r["mae"] for r in base]),
        "n_converged": sum(r["converged"] for r in results),
    }

sum_mse = summarize(results_mse)
sum_bce = summarize(results_bce)
print("\n=== Сводная таблица (средние по сошедшимся запускам) ===")
print(f"{'Метрика':<20}{'MSE (A)':<15}{'BCE (Б)':<15}")
print(f"{'Эпох':<20}{sum_mse['epochs_mean']:<15.1f}{sum_bce['epochs_mean']:<15.1f}")
print(f"{'Es итог':<20}{sum_mse['Es_mean']:<15.5f}{sum_bce['Es_mean']:<15.5f}")
print(f"{'Accuracy, %':<20}{sum_mse['accuracy_mean']:<15.1f}{sum_bce['accuracy_mean']:<15.1f}")
print(f"{'MAE (в [c0;c1])':<20}{sum_mse['mae_mean']:<15.3f}{sum_bce['mae_mean']:<15.3f}")
print(f"{'Сошлось из 5':<20}{sum_mse['n_converged']:<15}{sum_bce['n_converged']:<15}")

def pick_representative(results):
    for r in results:
        if r["converged"]:
            return r
    return results[0]

rep_mse = pick_representative(results_mse)
rep_bce = pick_representative(results_bce)


plt.figure(figsize=(8, 5))
plt.plot(rep_mse["history"], label="MSE (конфигурация А)", color="tab:blue")
plt.plot(rep_bce["history"], label="BCE (конфигурация Б)", color="tab:orange")
plt.axhline(EE_MSE, color="tab:blue", linestyle="--", linewidth=1,
            label=f"Ee (MSE) = {EE_MSE}")
plt.axhline(EE_BCE, color="tab:orange", linestyle="--", linewidth=1,
            label=f"Ee (BCE) = {EE_BCE}")
plt.yscale("log")
plt.xlabel("Эпоха")
plt.ylabel("Суммарная ошибка Es (лог. шкала)")
plt.title("Сходимость обучения: MSE vs BCE (вариант 8)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_convergence.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
width = 0.35
x_idx = np.arange(len(SEEDS))

epochs_mse = [r["epochs"] for r in results_mse]
epochs_bce = [r["epochs"] for r in results_bce]
conv_mse = [r["converged"] for r in results_mse]
conv_bce = [r["converged"] for r in results_bce]

bars1 = ax.bar(x_idx - width / 2, epochs_mse, width, label="MSE (А)", color="tab:blue")
bars2 = ax.bar(x_idx + width / 2, epochs_bce, width, label="BCE (Б)", color="tab:orange")

for bar, ok in zip(bars1, conv_mse):
    if not ok:
        bar.set_hatch("//")
        bar.set_edgecolor("black")
for bar, ok in zip(bars2, conv_bce):
    if not ok:
        bar.set_hatch("//")
        bar.set_edgecolor("black")

ax.set_xticks(x_idx)
ax.set_xticklabels([f"seed={s}" for s in SEEDS])
ax.set_ylabel("Эпох до сходимости (или MAX_EPOCHS, если не сошёлся)")
ax.set_title("Устойчивость сходимости по 5 запускам (штриховка = не сошёлся)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_epochs_spread.png"), dpi=150)
plt.close()

def decision_surface(net, use_real_scale):
    a_vals = np.linspace(-10, 10, 200)
    b_vals = np.linspace(-10, 10, 200)
    AA, BB = np.meshgrid(a_vals, b_vals)
    Z = np.zeros_like(AA)
    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
            x = normalize(np.array([AA[i, j], BB[i, j]]))
            o = net.predict(x)
            Z[i, j] = denormalize(o) if use_real_scale else o
    return AA, BB, Z

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

AA, BB, Z_mse = decision_surface(rep_mse["net"], use_real_scale=True)
cs0 = axes[0].contourf(AA, BB, Z_mse, levels=20, cmap="coolwarm")
axes[0].scatter(X_raw[:, 0], X_raw[:, 1], c=Y_raw, cmap="coolwarm",
                edgecolors="black", s=120, linewidths=1.5, zorder=5)
axes[0].set_title("Конфигурация А (MSE): ŷ в шкале [c0; c1]")
axes[0].set_xlabel("A")
axes[0].set_ylabel("B")
fig.colorbar(cs0, ax=axes[0], label="ŷ (реальная шкала)")

AA, BB, Z_bce = decision_surface(rep_bce["net"], use_real_scale=False)
cs1 = axes[1].contourf(AA, BB, Z_bce, levels=20, cmap="coolwarm", vmin=0, vmax=1)
Y_binary = normalize(Y_raw)  # 0/1 метки для отображения точек
axes[1].scatter(X_raw[:, 0], X_raw[:, 1], c=Y_binary, cmap="coolwarm",
                edgecolors="black", s=120, linewidths=1.5, zorder=5,
                vmin=0, vmax=1)
axes[1].set_title("Конфигурация Б (BCE): ŷ ∈ (0;1) (нормализованная шкала)")
axes[1].set_xlabel("A")
axes[1].set_ylabel("B")
fig.colorbar(cs1, ax=axes[1], label="ŷ (норм.)")

plt.suptitle("Разделяющая поверхность скрытого слоя, вариант 8")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_decision_surface.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
mae_values = [sum_mse["mae_mean"], sum_bce["mae_mean"]]
bars = ax.bar(["MSE (А)", "BCE (Б)"], mae_values, color=["tab:blue", "tab:orange"])
for bar, val in zip(bars, mae_values):
    ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}",
            ha="center", va="bottom")
ax.set_ylabel("Средняя абсолютная ошибка (шкала [c0; c1])")
ax.set_title("Точность восстановления исходной шкалы: MSE vs BCE")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_mae_comparison.png"), dpi=150)
plt.close()

print(f"\nГрафики сохранены в папке: {OUTPUT_DIR}")
print("(fig1_convergence.png, fig2_epochs_spread.png, "
      "fig3_decision_surface.png, fig4_mae_comparison.png)")

def run_inference(net, A, B):
    x = normalize(np.array([A, B], dtype=float))
    o = net.predict(x)
    y_real = denormalize(o)
    nearest_class = C0 if abs(y_real - C0) < abs(y_real - C1) else C1
    return o, y_real, nearest_class


def demo_inference(net, title):
    print(f"\n=== Режим функционирования: {title} ===")
    print(f"{'A':>6}{'B':>6}   {'ŷ_норм':>8}   {'ŷ_реал':>8}   класс")
    demo_pairs = list(map(tuple, X_raw)) + [(-5, 0), (1, -2), (10, -10)]
    for A, B in demo_pairs:
        o, y_real, cls = run_inference(net, A, B)
        print(f"{A:>6}{B:>6}   {o:>8.4f}   {y_real:>8.3f}   {cls}")


demo_inference(rep_mse["net"], "конфигурация А (MSE)")
demo_inference(rep_bce["net"], "конфигурация Б (BCE)")

if __name__ == "__main__":
    print("\n=== Интерактивный режим (модель BCE) ===")
    print("Введите A и B из диапазона [-10;10] ('q' для выхода).")
    while True:
        raw = input("A = ")
        if raw.strip().lower() == "q":
            break
        try:
            A = float(raw)
            B = float(input("B = "))
        except ValueError:
            print("Ошибка: введите числа.")
            continue
        o, y_real, cls = run_inference(rep_bce["net"], A, B)
        print(f"ŷ_норм = {o:.4f}   ŷ_реал = {y_real:.3f}   "
              f"ближе к классу {cls} ({'c0' if cls == C0 else 'c1'})")