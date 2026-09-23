import numpy as np
import matplotlib.pyplot as plt

c0 = 6.0
c1 = -4.0

X = np.array([[ 6.,  6.],
              [ 6., -4.],
              [-4.,  6.],
              [-4., -4.]])
y = np.array([[ c0],
              [ c1],
              [ c1],
              [ c0]])

def to_norm(y_orig, c0, c1):
    return (y_orig - min(c0, c1)) / (abs(c1 - c0))

def from_norm(y_norm, c0, c1):
    return y_norm * (c1 - c0) + c0

y_norm = to_norm(y, c0, c1)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative_from_output(sig_x):
    return sig_x * (1.0 - sig_x)

class MLP221:
    def __init__(self, lr=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.uniform(-0.5, 0.5, (2, 2))
        self.b1 = np.zeros((1, 2))
        self.W2 = np.random.uniform(-0.5, 0.5, (2, 1))
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    def forward(self, x):
        h_in = np.dot(x, self.W1) + self.b1
        h_out = sigmoid(h_in)
        o_in = np.dot(h_out, self.W2) + self.b2
        o_out = sigmoid(o_in)
        return h_in, h_out, o_in, o_out

    def predict_norm(self, x):
        return self.forward(x)[3].item()

    def predict_rescaled(self, x, c0, c1):
        return from_norm(self.predict_norm(x), c0, c1)

    def train_epoch_online(self, X_train, y_train_norm, loss='mse'):
        losses = []
        for xi, yi in zip(X_train, y_train_norm):
            xi = xi.reshape(1, -1)
            yi = yi.reshape(1, -1)
            h_in = np.dot(xi, self.W1) + self.b1
            h_out = sigmoid(h_in)
            o_in = np.dot(h_out, self.W2) + self.b2
            o_out = sigmoid(o_in)
            if loss == 'mse':
                delta_out = (o_out - yi) * sigmoid_derivative_from_output(o_out)
            elif loss == 'bce':
                delta_out = (o_out - yi)
            grad_W2 = np.dot(h_out.T, delta_out)
            grad_b2 = np.sum(delta_out, axis=0, keepdims=True)
            delta_hidden = np.dot(delta_out, self.W2.T) * sigmoid_derivative_from_output(h_out)
            grad_W1 = np.dot(xi.T, delta_hidden)
            grad_b1 = np.sum(delta_hidden, axis=0, keepdims=True)
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1
            if loss == 'mse':
                sample_loss = np.mean((yi - o_out) ** 2)
            else:
                eps = 1e-12
                o = np.clip(o_out, eps, 1 - eps)
                sample_loss = - (yi * np.log(o) + (1 - yi) * np.log(1 - o))
                sample_loss = sample_loss.mean()
            losses.append(sample_loss)
        return np.mean(losses)

def run_experiments(seeds, config='A', lr=0.1, max_epochs=5000, Ee=None):
    results = []
    all_histories = []
    for seed in seeds:
        mlp = MLP221(lr=lr, seed=seed)
        history = []
        reached = False
        for epoch in range(max_epochs):
            if config == 'A':
                loss_val = mlp.train_epoch_online(X, y_norm, loss='mse')
                Es = loss_val
            else:
                loss_val = mlp.train_epoch_online(X, y_norm, loss='bce')
                Es = loss_val
            history.append(Es)
            if Ee is not None and Es <= Ee:
                reached = True
                break
        final_outputs_norm = np.array([mlp.predict_norm(xi) for xi in X]).reshape(-1,1)
        final_outputs_rescaled = from_norm(final_outputs_norm, c0, c1)
        if config == 'A':
            Es_final = np.mean((y_norm - final_outputs_norm)**2)
        else:
            eps = 1e-12
            o = np.clip(final_outputs_norm, eps, 1-eps)
            Es_final = - np.mean(y_norm * np.log(o) + (1-y_norm) * np.log(1-o))
        midpoint = (c0 + c1) / 2.0
        preds_class = np.where(final_outputs_rescaled >= midpoint, c1, c0).reshape(-1,1)
        accuracy = np.mean(preds_class == y)
        mae = np.mean(np.abs(final_outputs_rescaled - y))
        results.append({
            'seed': seed,
            'reached': reached,
            'epochs': epoch if reached else max_epochs,
            'Es_final': float(Es_final),
            'accuracy': float(accuracy),
            'mae': float(mae),
            'final_outputs_norm': final_outputs_norm,
            'final_outputs_rescaled': final_outputs_rescaled,
            'history': history,
            'model': mlp
        })
        all_histories.append(history)
    return results, all_histories

seeds = [1, 2, 3, 4, 5]
lr = 0.1
max_epochs = 5000
Ee_MSE = 0.01
Ee_BCE = 1e-3

res_A, hist_A = run_experiments(seeds, config='A', lr=lr, max_epochs=max_epochs, Ee=Ee_MSE)
res_B, hist_B = run_experiments(seeds, config='B', lr=lr, max_epochs=max_epochs, Ee=Ee_BCE)

def summarize(results, name):
    print(f"\n{name}")
    for r in results:
        print(f"{r['seed']} {r['reached']} {r['epochs']} {r['Es_final']:.6f} {r['accuracy']:.3f} {r['mae']:.4f}")

summarize(res_A, 'A')
summarize(res_B, 'B')

def pick_representative(results):
    for r in results:
        if r['reached']:
            return r
    return results[0]

rep_A = pick_representative(res_A)
rep_B = pick_representative(res_B)

plt.figure(figsize=(10,5))
plt.suptitle("График сходимости: суммарная ошибка Es от номера эпохи", fontsize=14, y=0.98)
plt.plot(rep_A['history'], label=f"A seed={rep_A['seed']}")
plt.plot(rep_B['history'], label=f"B seed={rep_B['seed']}")
plt.axhline(Ee_MSE, color='blue', linestyle='--')
plt.axhline(Ee_BCE, color='orange', linestyle=':')
plt.xlabel("epoch")
plt.ylabel("Es")
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

fig, ax = plt.subplots(figsize=(8,4))
fig.suptitle("Разброс числа эпох по 5 запускам для конфигураций A и B", fontsize=14, y=0.98)
indices = np.arange(len(seeds))
width = 0.35
epochs_A = [r['epochs'] for r in res_A]
epochs_B = [r['epochs'] for r in res_B]
bars1 = ax.bar(indices - width/2, epochs_A, width, label='A')
bars2 = ax.bar(indices + width/2, epochs_B, width, label='B')
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
ax.legend()
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

def plot_decision_surface(model, title, c0, c1, plot_rescaled=True):
    grid_n = 200
    xs = np.linspace(-10, 10, grid_n)
    ys = np.linspace(-10, 10, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel()]
    outs = np.array([model.predict_norm(p) for p in pts]).reshape(grid_n, grid_n)
    if plot_rescaled:
        outs_plot = from_norm(outs, c0, c1)
    else:
        outs_plot = outs
    plt.figure(figsize=(6,5))
    plt.suptitle(title, fontsize=12, y=0.98)
    plt.contourf(XX, YY, outs_plot, levels=50, cmap='RdYlBu', alpha=0.9)
    for xi, yi in zip(X, y):
        if yi == c1:
            plt.scatter(xi[0], xi[1], c='black', marker='o', edgecolors='white', s=80)
        else:
            plt.scatter(xi[0], xi[1], c='white', marker='s', edgecolors='black', s=80)
    plt.xlabel("A")
    plt.ylabel("B")
    plt.colorbar()
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

plot_decision_surface(rep_A['model'], f"Разделяющая поверхность — A (MSE) seed={rep_A['seed']}", c0, c1, plot_rescaled=True)
plot_decision_surface(rep_B['model'], f"Разделяющая поверхность — B (BCE) seed={rep_B['seed']}", c0, c1, plot_rescaled=True)

mae_A = np.mean([r['mae'] for r in res_A])
mae_B = np.mean([r['mae'] for r in res_B])
plt.figure(figsize=(5,4))
plt.suptitle("Сравнение MAE в исходной шкале для конфигураций A и B", fontsize=14, y=0.98)
plt.bar(['A','B'], [mae_A, mae_B], color=['blue','orange'])
plt.ylabel("MAE")
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

print("\nDemo:")
test_inputs = [[7,7],[7,-7],[-7,7],[-7,-7],[0,0],[3,-7],[5,2]]
for inp in test_inputs:
    a,b = inp
    yA_norm = rep_A['model'].predict_norm(np.array([a,b]))
    yA_res = from_norm(yA_norm, c0, c1)
    yB_norm = rep_B['model'].predict_norm(np.array([a,b]))
    yB_res = from_norm(yB_norm, c0, c1)
    print(inp, yA_norm, yA_res, yB_norm, yB_res)