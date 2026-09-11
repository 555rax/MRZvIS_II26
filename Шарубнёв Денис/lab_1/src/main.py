import numpy as np
from itertools import product
import time
import json


c0 = 3.0
c1 = -8.0
X = np.array([[3.0, 3.0],
              [3.0, -8.0],
              [-8.0, 3.0],
              [-8.0, -8.0]])
y = np.array([c0, c1, c1, c0]).reshape(-1,1)

X_min, X_max = -8.0, 3.0
X_norm = (X - X_min) / (X_max - X_min) * 2 - 1  # в [-1,1]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(s):
    return s * (1 - s)

y_min, y_max = min(c0, c1), max(c0, c1)

def scale_y(y_raw):
    return (y_raw - y_min) / (y_max - y_min)

def descale_y(y_scaled):
    return y_scaled * (y_max - y_min) + y_min

y_scaled = scale_y(y)

def accuracy_from_preds(preds, targets):
    preds = preds.ravel()
    targets = targets.ravel()
    pick = np.where(np.abs(preds - c0) <= np.abs(preds - c1), c0, c1)
    return np.mean(pick == targets)


class SingleLayerPerceptron:
    def __init__(self, n_in, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W = np.random.uniform(-0.5,0.5,(n_in+1,1))

    def forward(self, x, linear_output=False):
        x_b = np.hstack([x, np.ones((x.shape[0],1))])
        z = x_b.dot(self.W)
        out = z if linear_output else sigmoid(z)
        return out, x_b

    def train(self, X, y_scaled, lr, max_epochs, Ee, linear_output=False, shuffle=True, momentum=0.0):
        n = X.shape[0]
        vW = np.zeros_like(self.W)
        for epoch in range(1, max_epochs+1):
            Es = 0.0
            idx = np.random.permutation(n) if shuffle else np.arange(n)
            for i in idx:
                xi = X[i:i+1]
                yi = y_scaled[i:i+1]
                out, x_b = self.forward(xi, linear_output)
                err = yi - out
                Es += 0.5 * np.sum(err**2)
                delta = err if linear_output else err * dsigmoid(out)
                grad = x_b.T.dot(delta)
                vW = momentum * vW + lr * grad
                self.W += vW
            if Es <= Ee:
                return epoch, Es
        return None, Es


class MLP_2_2_1:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.uniform(-0.5,0.5,(3,2))
        self.W2 = np.random.uniform(-0.5,0.5,(3,1))

    def forward(self, x, linear_output=False):
        x_b = np.hstack([x, np.ones((x.shape[0],1))])
        z1 = x_b.dot(self.W1)
        a1 = sigmoid(z1)
        a1_b = np.hstack([a1, np.ones((a1.shape[0],1))])
        z2 = a1_b.dot(self.W2)
        a2 = z2 if linear_output else sigmoid(z2)
        return a2, a1, x_b, a1_b

    def train(self, X, y_scaled, lr, max_epochs, Ee, linear_output=False, shuffle=True, momentum=0.0):
        n = X.shape[0]
        vW1 = np.zeros_like(self.W1)
        vW2 = np.zeros_like(self.W2)
        for epoch in range(1, max_epochs+1):
            Es = 0.0
            idx = np.random.permutation(n) if shuffle else np.arange(n)
            for i in idx:
                xi = X[i:i+1]
                yi = y_scaled[i:i+1]
                out, a1, x_b, a1_b = self.forward(xi, linear_output)
                err = yi - out
                Es += 0.5 * np.sum(err**2)
                delta2 = err if linear_output else err * dsigmoid(out)
                gradW2 = a1_b.T.dot(delta2)
                W2_no_bias = self.W2[:-1,:]
                delta1 = (delta2.dot(W2_no_bias.T)) * dsigmoid(a1)
                gradW1 = x_b.T.dot(delta1)
                vW2 = momentum * vW2 + lr * gradW2
                vW1 = momentum * vW1 + lr * gradW1
                self.W2 += vW2
                self.W1 += vW1
            if Es <= Ee:
                return epoch, Es
        return None, Es



def main():
    lr_list = [0.05, 0.1, 0.2, 0.5]
    momentum_list = [0.0, 0.9]
    linear_output_list = [False, True]
    seeds = [1, 7, 42]
    max_epochs = 20000
    Ee = 1e-3

    results = []
    start_all = time.time()
    total_runs = len(lr_list)*len(momentum_list)*len(linear_output_list)*len(seeds)
    run_counter = 0

    for lr, momentum, linear_output, seed in product(lr_list, momentum_list, linear_output_list, seeds):
        run_counter += 1
        slp = SingleLayerPerceptron(n_in=2, seed=seed)
        mlp = MLP_2_2_1(seed=seed+100)

        if linear_output:
            y_train_scaled = y.copy()
            Ee_run = 1e-2
        else:
            y_train_scaled = y_scaled
            Ee_run = Ee

        epoch_slp, Es_slp = slp.train(X_norm, y_train_scaled, lr, max_epochs, Ee_run,
                                      linear_output=linear_output, shuffle=True, momentum=momentum)
        epoch_mlp, Es_mlp = mlp.train(X_norm, y_train_scaled, lr, max_epochs, Ee_run,
                                      linear_output=linear_output, shuffle=True, momentum=momentum)

        def pred_slp_vals(model):
            out, _ = model.forward(X_norm, linear_output=linear_output)
            return out if linear_output else descale_y(out)

        def pred_mlp_vals(model):
            out, _, _, _ = model.forward(X_norm, linear_output=linear_output)
            return out if linear_output else descale_y(out)

        preds_slp = pred_slp_vals(slp)
        preds_mlp = pred_mlp_vals(mlp)

        acc_slp = accuracy_from_preds(preds_slp, y)
        acc_mlp = accuracy_from_preds(preds_mlp, y)

        results.append({
            'lr': lr,
            'momentum': momentum,
            'linear_output': linear_output,
            'seed': seed,
            'epoch_slp': epoch_slp,
            'Es_slp': float(Es_slp),
            'acc_slp': acc_slp,
            'epoch_mlp': epoch_mlp,
            'Es_mlp': float(Es_mlp),
            'acc_mlp': acc_mlp
        })

        print(f"[{run_counter}/{total_runs}] lr={lr} mom={momentum} lin={linear_output} seed={seed} | "
              f"SLP: ep={epoch_slp} Es={Es_slp:.6f} acc={acc_slp:.2f} | "
              f"MLP: ep={epoch_mlp} Es={Es_mlp:.6f} acc={acc_mlp:.2f}")

    elapsed = time.time() - start_all
    print("\nОбучение завершено. Время:", elapsed)


    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Результаты сохранены в results.json")



if __name__ == "__main__":
    main()
