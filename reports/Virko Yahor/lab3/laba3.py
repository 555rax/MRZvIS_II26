import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.edgecolor': '#444444',
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.color': '#dddddd',
    'grid.linestyle': '--',
    'grid.linewidth': 0.7,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#888888',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})

VARIANT = 1
C0, C1 = 0.0, -6.0

MAX_EPOCHS = 15000
SEEDS = [0, 1, 2, 3, 4]
LR = 0.5
EE = 0.05

X_raw = np.array([[C0, C0], [C0, C1], [C1, C0], [C1, C1]], dtype=float)
Y_REAL = np.array([C0, C1, C1, C0], dtype=float)
Y_NORM = (Y_REAL - C0) / (C1 - C0)
X = (X_raw - C0) / (C1 - C0)


def denorm(y):
    return C0 + y * (C1 - C0)


class MLP:
    def __init__(self, n_in=2, n_hidden=2, n_out=1, lr=0.5, seed=0, hidden_activation='sigmoid'):
        self.rng = np.random.default_rng(seed)
        self.W1 = self.rng.normal(0, 0.5, size=(n_in, n_hidden))
        self.b1 = self.rng.normal(0, 0.5, size=(n_hidden,))
        self.W2 = self.rng.normal(0, 0.5, size=(n_hidden, n_out))
        self.b2 = self.rng.normal(0, 0.5, size=(n_out,))
        self.lr = lr
        self.hidden_activation = hidden_activation

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_deriv(self, z):
        s = self.sigmoid(z)
        return s * (1.0 - s)

    def relu(self, z):
        return np.maximum(0.0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def activate_hidden(self, z):
        if self.hidden_activation == 'relu':
            return self.relu(z)
        return self.sigmoid(z)

    def activate_hidden_deriv(self, z):
        if self.hidden_activation == 'relu':
            return self.relu_deriv(z)
        return self.sigmoid_deriv(z)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = self.activate_hidden(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)
        return a1, a2, z1, z2

    def forward_batch(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.activate_hidden(z1)
        z2 = a1 @ self.W2 + self.b2
        return self.sigmoid(z2)

    def backward(self, x, y):
        a1, a2, z1, z2 = self.forward(x)
        a2 = float(a2.item())
        dz2 = a2 - y
        dW2 = a1.reshape(-1, 1) * dz2
        db2 = np.array([dz2])
        da1 = self.W2.flatten() * dz2
        dz1 = da1 * self.activate_hidden_deriv(z1)
        dW1 = np.outer(x, dz1)
        db1 = dz1

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def loss(self, y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    def total_error(self, X, y):
        preds = self.forward_batch(X).ravel()
        return float(np.sum([self.loss(y[i], preds[i]) for i in range(len(X))]))

    def train(self, X, y, Ee=0.05, max_epochs=15000):
        history = []
        n = len(X)
        for epoch in range(max_epochs):
            idx = self.rng.permutation(n)
            for i in idx:
                self.backward(X[i], y[i])
            Es = self.total_error(X, y)
            if not np.isfinite(Es):
                return epoch + 1, float('inf'), False, history
            history.append(Es)
            if Es <= Ee:
                return epoch + 1, Es, True, history
        return max_epochs, history[-1] if history else float('inf'), False, history

    def predict(self, x):
        return float(self.forward_batch(np.array([x], dtype=float)).ravel()[0])

    def count_dead(self, X):
        a1 = self.activate_hidden(X @ self.W1 + self.b1)
        return int(np.sum(np.all(a1 <= 1e-12, axis=0)))


print(f"Вариант {VARIANT}:  c0 = {C0},  c1 = {C1}")

configs = [
    ('A (Sigmoid)', 'sigmoid', '#1f77b4', '#aec7e8'),
    ('B (ReLU)', 'relu', '#ff7f0e', '#ffbb78'),
]

results = {}
models = {}

for cfg_name, hid_act, color, fill in configs:
    epochs_list, conv_list, Es_list, acc_list, mae_list, hists, dead_list = [], [], [], [], [], [], []
    best_Es, best_model, best_hist = np.inf, None, None
    best_seed, best_ep = None, None

    print(f"\nОбучение {cfg_name} ...")
    for seed in SEEDS:
        model = MLP(lr=LR, seed=seed, hidden_activation=hid_act)
        ep, Es, conv, hist = model.train(X, Y_NORM, EE, MAX_EPOCHS)

        preds = np.array([model.predict(x) for x in X])
        preds_real = denorm(preds)
        pred_class = (preds > 0.5).astype(int)
        true_class = Y_NORM.astype(int)
        acc = float(np.mean(pred_class == true_class))
        mae = float(np.mean(np.abs(preds_real - Y_REAL)))
        dead = model.count_dead(X)

        stopped_ep = ep if conv else MAX_EPOCHS

        epochs_list.append(stopped_ep)
        conv_list.append(conv)
        Es_list.append(Es)
        acc_list.append(acc)
        mae_list.append(mae)
        hists.append(hist)
        dead_list.append(dead)

        if np.isfinite(Es) and Es < best_Es:
            best_Es, best_model, best_hist = Es, model, hist
            best_seed, best_ep = seed, stopped_ep

        print(f"  seed={seed}: epochs={stopped_ep:>6}, converged={conv}, "
              f"Es={Es:.6f}, acc={acc:.2f}, MAE={mae:.4f}, dead={dead}")

    results[cfg_name] = dict(
        epochs_list=epochs_list, conv_list=conv_list, Es_list=Es_list,
        acc_list=acc_list, mae_list=mae_list, hists=hists, dead_list=dead_list,
        best_hist=best_hist, best_seed=best_seed, best_ep=best_ep,
        color=color, fill=fill,
    )
    models[cfg_name] = best_model

COLOR_A = results['A (Sigmoid)']['color']
COLOR_B = results['B (ReLU)']['color']
FILL_A = results['A (Sigmoid)']['fill']
FILL_B = results['B (ReLU)']['fill']

CLS_C0 = '#e63946'
CLS_C1 = '#2a9d8f'
CMAP = 'RdYlBu_r'


fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.15], hspace=0.45, wspace=0.30)

ax_conv = fig.add_subplot(gs[0, :])
for cfg_name in ['A (Sigmoid)', 'B (ReLU)']:
    r = results[cfg_name]
    if r['best_hist'] is None:
        continue
    h = np.clip(np.array(r['best_hist'], dtype=float), 1e-12, None)
    ep = np.arange(1, len(h) + 1)
    ax_conv.plot(ep, h, color=r['color'], linewidth=2.5,
                 label=(f"{cfg_name}: seed={r['best_seed']}, "
                        f"{r['best_ep']} эпох, $E_s$={r['best_hist'][-1]:.2e}"))
    ax_conv.scatter([r['best_ep']], [h[-1]], color=r['color'],
                    s=110, edgecolors='white', linewidths=2.2, zorder=6)

ax_conv.axhline(EE, color='red', linestyle=':', linewidth=2, alpha=0.85,
                label=f'Порог $E_e$ = {EE}')
ax_conv.set_xlabel('Номер эпохи')
ax_conv.set_ylabel('Суммарная ошибка  $E_s$')
ax_conv.set_title('Сходимость на задаче XOR: Sigmoid vs ReLU')
ax_conv.set_yscale('log')
ax_conv.set_xlim(1, MAX_EPOCHS)
ax_conv.legend(loc='upper right', fontsize=10, ncol=2, columnspacing=1.2)
ax_conv.grid(True, which='both', linestyle='--', alpha=0.4)

ax_ep = fig.add_subplot(gs[1, 0])
x = np.arange(len(SEEDS))
w = 0.38
vals_a = np.maximum(np.array(results['A (Sigmoid)']['epochs_list'], dtype=float), 1.0)
vals_b = np.maximum(np.array(results['B (ReLU)']['epochs_list'], dtype=float), 1.0)

bars_a = ax_ep.bar(x - w / 2, vals_a, w, color=FILL_A, edgecolor=COLOR_A,
                   linewidth=1.5, label='A (Sigmoid)')
bars_b = ax_ep.bar(x + w / 2, vals_b, w, color=FILL_B, edgecolor=COLOR_B,
                   linewidth=1.5, label='B (ReLU)')

for bars, key in [(bars_a, 'A (Sigmoid)'), (bars_b, 'B (ReLU)')]:
    for bar, conv in zip(bars, results[key]['conv_list']):
        if not conv:
            bar.set_hatch('////')
            bar.set_edgecolor('#333333')
            bar.set_linewidth(1.8)
        h = bar.get_height()
        ax_ep.text(bar.get_x() + bar.get_width() / 2, h * 1.12,
                   f'{int(h)}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

ax_ep.set_xlabel('Seed')
ax_ep.set_ylabel('Эпох до остановки (лог. шкала)')
ax_ep.set_title('Разброс числа эпох по 5 запускам\n(штриховка — не сошлась)')
ax_ep.set_xticks(x)
ax_ep.set_xticklabels([f'seed={s}' for s in SEEDS])
ax_ep.set_yscale('log')
ax_ep.set_ylim(1, MAX_EPOCHS * 4)
ax_ep.legend(loc='upper left', fontsize=10)

ax_mae = fig.add_subplot(gs[1, 1])
mae_a = results['A (Sigmoid)']['mae_list']
mae_b = results['B (ReLU)']['mae_list']
ax_mae.bar(x - w / 2, mae_a, w, color=FILL_A, edgecolor=COLOR_A,
           linewidth=1.5, label='A (Sigmoid)')
ax_mae.bar(x + w / 2, mae_b, w, color=FILL_B, edgecolor=COLOR_B,
           linewidth=1.5, label='B (ReLU)')
mA, mB = float(np.mean(mae_a)), float(np.mean(mae_b))
ax_mae.axhline(mA, color=COLOR_A, linestyle='--', linewidth=1.8, alpha=0.9,
               label=f'Среднее A = {mA:.3f}')
ax_mae.axhline(mB, color=COLOR_B, linestyle='--', linewidth=1.8, alpha=0.9,
               label=f'Среднее B = {mB:.3f}')
ax_mae.set_xlabel('Seed')
ax_mae.set_ylabel('MAE в шкале $[c_0;\\, c_1]$')
ax_mae.set_title('Точность восстановления исходной шкалы')
ax_mae.set_xticks(x)
ax_mae.set_xticklabels([f'seed={s}' for s in SEEDS])
ax_mae.legend(loc='upper right', fontsize=9)


def surface_grid(model, n=200):
    g = np.linspace(-10, 10, n)
    xx, yy = np.meshgrid(g, g)
    inputs_raw = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inputs = (inputs_raw - C0) / (C1 - C0)
    out = model.forward_batch(inputs).ravel()
    out = denorm(out)
    return xx, yy, out.reshape(n, n)


def draw_surface(ax, model, cfg_name):
    if model is None:
        ax.text(0.5, 0.5, 'модель недоступна', ha='center', va='center',
                transform=ax.transAxes)
        return
    xx, yy, Z = surface_grid(model)
    cf = ax.contourf(xx, yy, Z, levels=25, cmap=CMAP, alpha=0.95)
    ax.contour(xx, yy, Z, levels=15, colors='white', linewidths=0.5, alpha=0.5)

    for (a, b), y in zip(X_raw, Y_REAL):
        col = CLS_C0 if y == C0 else CLS_C1
        ax.scatter(a, b, c=col, s=180, edgecolors='white',
                   linewidths=2.3, zorder=5)

    ax.set_xlabel('Вход A')
    ax.set_ylabel('Вход B')
    ax.set_title(f'Разделяющая поверхность — {cfg_name}')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')

    cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Выход сети, шкала $[c_0;\\, c_1]$', fontsize=10)

    leg = [
        Line2D([0], [0], marker='o', color='w', label=f'$c_0$ = {C0}',
               markerfacecolor=CLS_C0, markersize=11,
               markeredgecolor='white', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', label=f'$c_1$ = {C1}',
               markerfacecolor=CLS_C1, markersize=11,
               markeredgecolor='white', markeredgewidth=1.5),
    ]
    ax.legend(handles=leg, loc='lower right', fontsize=10)


ax_sa = fig.add_subplot(gs[2, 0])
draw_surface(ax_sa, models['A (Sigmoid)'], 'A (Sigmoid)')

ax_sb = fig.add_subplot(gs[2, 1])
draw_surface(ax_sb, models['B (ReLU)'], 'B (ReLU)')

fig.suptitle(f'ЛР №3 — MLP 2-2-1 для XOR: Sigmoid vs ReLU  |  '
             f'$c_0$ = {C0},  $c_1$ = {C1}',
             fontsize=15, fontweight='bold', y=0.995)

plt.savefig('lab3_main.png', dpi=150, bbox_inches='tight')
plt.show()


fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Детальные графики', fontsize=14, fontweight='bold')

for ax, cfg_name in zip([axes2[0, 0], axes2[0, 1]], ['A (Sigmoid)', 'B (ReLU)']):
    r = results[cfg_name]
    for seed, hist, conv in zip(SEEDS, r['hists'], r['conv_list']):
        if not hist:
            continue
        h = np.clip(np.array(hist, dtype=float), 1e-12, None)
        ep = np.arange(1, len(h) + 1)
        style = '-' if conv else '--'
        lbl = (f'seed={seed}: {len(hist)} эп.' if conv
               else f'seed={seed}: не сошлась ({len(hist)} эп.)')
        ax.plot(ep, h, style, linewidth=1.6, alpha=0.85, label=lbl)
    ax.axhline(EE, color='red', linestyle=':', linewidth=1.8, label=f"$E_e$ = {EE}")
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('$E_s$')
    ax.set_yscale('log')
    ax.set_title(f'{cfg_name} — кривые сходимости всех запусков')
    ax.legend(fontsize=9, loc='upper right', ncol=2, columnspacing=1.0)
    ax.grid(True, which='both', alpha=0.35)

ax_acc = axes2[1, 0]
ax_acc.plot(SEEDS, results['A (Sigmoid)']['acc_list'], 'o-',
            color=COLOR_A, linewidth=2, markersize=11,
            markeredgecolor='white', markeredgewidth=1.6, label='A (Sigmoid)')
ax_acc.plot(SEEDS, results['B (ReLU)']['acc_list'], 's-',
            color=COLOR_B, linewidth=2, markersize=11,
            markeredgecolor='white', markeredgewidth=1.6, label='B (ReLU)')
ax_acc.set_xlabel('Seed')
ax_acc.set_ylabel('Accuracy')
ax_acc.set_title('Точность классификации по запускам')
ax_acc.set_ylim(-0.1, 1.15)
ax_acc.set_xticks(SEEDS)
ax_acc.legend(fontsize=10, loc='lower right')
ax_acc.grid(True, alpha=0.4)

ax_tab = axes2[1, 1]
ax_tab.axis('off')
rows = []
for cfg_name in ['A (Sigmoid)', 'B (ReLU)']:
    r = results[cfg_name]
    n_conv = sum(r['conv_list'])
    conv_eps = [e for e, c in zip(r['epochs_list'], r['conv_list']) if c]
    mean_ep = float(np.mean(conv_eps)) if conv_eps else float('nan')
    rows.append([
        cfg_name,
        f"{n_conv} / 5",
        f"{mean_ep:.0f}" if np.isfinite(mean_ep) else '—',
        f"{np.mean(r['Es_list']):.5f}",
        f"{np.mean(r['acc_list']):.2f}",
        f"{np.mean(r['mae_list']):.3f}",
        f"{np.mean(r['dead_list']):.1f}",
    ])

cols = ['Конфигурация', 'Сошлось', 'Сред. эпох', 'Сред. $E_s$',
        'Accuracy', 'MAE', 'Мёртвых']
tbl = ax_tab.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center',
                   colWidths=[0.20, 0.12, 0.14, 0.16, 0.14, 0.12, 0.12])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 2.4)
for j in range(len(cols)):
    tbl[0, j].set_facecolor('#34495e')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows) + 1):
    for j in range(len(cols)):
        tbl[i, j].set_facecolor('#f4f6f8' if i % 2 else '#e8ecf1')
        tbl[i, j].set_edgecolor('#aab4bf')
ax_tab.set_title('Сводная таблица результатов',
                 fontsize=13, fontweight='bold', pad=22)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('lab3_details.png', dpi=150, bbox_inches='tight')
plt.show()


def run_inference(model, a, b):
    xr = np.array([a, b], dtype=float)
    xn = (xr - C0) / (C1 - C0)
    yn = model.predict(xn)
    yr = denorm(yn)
    cls = C0 if abs(yr - C0) < abs(yr - C1) else C1
    return yn, yr, cls


def demo_mode(model):
    print("РЕЖИМ ФУНКЦИОНИРОВАНИЯ (конфигурация B, ReLU)")
    print(f"{'A':>7} | {'B':>7} | {'y_norm':>9} | {'y_real':>11} | {'класс':>8}")
    print("-" * 55)
    pairs = [(C0, C0), (C0, C1), (C1, C0), (C1, C1), (-5, 3), (7, -2), (0, 5)]
    for a, b in pairs:
        yn, yr, cls = run_inference(model, a, b)
        print(f"{a:>7.2f} | {b:>7.2f} | {yn:>9.4f} | {yr:>11.4f} | {cls:>8}")


def interactive_mode(model):
    print("\nИнтерактивный режим, введите 'q' для выхода")
    while True:
        s = input("Введите A B через пробел: ").strip()
        if s.lower() in ('q', 'quit', 'exit'):
            break
        parts = s.split()
        if len(parts) != 2:
            print("Нужно два числа")
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            print("Некорректный ввод")
            continue
        yn, yr, cls = run_inference(model, a, b)
        print(f"y_norm = {yn:.4f},  y_real = {yr:.4f},  класс = {cls}")


demo_mode(models['B (ReLU)'])
interactive_mode(models['B (ReLU)'])