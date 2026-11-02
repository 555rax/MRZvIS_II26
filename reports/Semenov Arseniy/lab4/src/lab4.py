
import argparse
import csv
import json
from pathlib import Path

import numpy as np

SPLIT, LENGTH = 140, 200
P_BASE, N_BASE = 6, 4
SEEDS = [42, 43, 44, 45, 46]


def generate_series():
    k = np.arange(LENGTH, dtype=float)
    return np.cos(.30 * k) + .5 * np.sin(.70 * k)


def fit_scaler(train):
    low, high = float(np.min(train)), float(np.max(train))
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError('Для нормализации нужен конечный непостоянный ряд.')
    return low, high


def normalize(values, scaler):
    low, high = scaler
    return (np.asarray(values, dtype=float) - low) / (high - low)


def denormalize(values, scaler):
    low, high = scaler
    return np.asarray(values) * (high - low) + low


def make_windows(series, p, start, stop):
    """Цели k=start..stop-1; окно в хронологическом порядке k-p..k-1."""
    if not (1 <= p <= start < stop <= len(series)):
        raise ValueError('Некорректные границы окон.')
    x = np.empty((stop-start, p))
    y = np.empty(stop-start)
    for row, k in enumerate(range(start, stop)):
        x[row] = series[k-p:k]
        y[row] = series[k]
    return x, y


def sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0., 1. / (1. + e), e / (1. + e))


class MLP:
    def __init__(self, p, n, seed):
        rng = np.random.default_rng(seed)
        self.w1 = rng.uniform(-.3, .3, (p, n))
        self.b1 = rng.uniform(-.3, .3, n)
        self.w2 = rng.uniform(-.3, .3, n)
        self.b2 = np.array(rng.uniform(-.3, .3))

    def parameters(self):
        return self.w1, self.b1, self.w2, self.b2

    def forward(self, x):
        h = sigmoid(np.asarray(x) @ self.w1 + self.b1)
        return h, h @ self.w2 + self.b2  # Линейный выход.

    def gradients(self, x, target):
        h, prediction = self.forward(x)
        delta = 2. * (prediction - target)  # L=(prediction-target)^2
        hidden_delta = delta * self.w2 * h * (1. - h)
        return (x[:, None] * hidden_delta, hidden_delta,
                delta * h, np.asarray(delta))

    def step(self, x, target, lr):
        # Скрытая дельта вычислена до обновления выходных весов.
        gradients = self.gradients(x, target)
        for param, grad in zip(self.parameters(), gradients):
            param -= lr * grad

    def to_list(self):
        return [param.tolist() for param in self.parameters()]


def mse(model, x, y):
    return float(np.mean((model.forward(x)[1] - y) ** 2))


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    error = np.abs(prediction-truth)
    denom = np.abs(truth) + np.abs(prediction)
    ratios = np.divide(2.*error, denom, out=np.zeros_like(error), where=denom > 0)
    return {'MAE': float(np.mean(error)),
            'RMSE': float(np.sqrt(np.mean(error**2))),
            'sMAPE': float(100.*np.mean(ratios))}


def train(x, y, n, seed, lr, ee, max_epochs):
    model = MLP(x.shape[1], n, seed)
    # Отдельный генератор: порядок эпох одинаков для одинаковых p и seed.
    order_rng = np.random.default_rng(seed + 100000)
    history = [mse(model, x, y)]
    while history[-1] > ee and len(history)-1 < max_epochs:
        for i in order_rng.permutation(len(y)):
            model.step(x[i], y[i], lr)
        error = mse(model, x, y)
        if not np.isfinite(error):
            raise FloatingPointError('Обучение разошлось; уменьшите --lr.')
        history.append(error)
    return model, history


def run_experiment(series, scaler, p, n, seed, args, noise=0.):
    z = normalize(series, scaler)
    x_train, y_train = make_windows(z, p, p, SPLIT)
    x_test, y_test = make_windows(z, p, SPLIT, LENGTH)
    fitting_x = x_train.copy()
    if noise:
        # Только обучающие входы. Цели и тестовые данные не изменяются.
        rng = np.random.default_rng(seed + 200000)
        fitting_x += rng.normal(0., noise, fitting_x.shape)
    model, history = train(fitting_x, y_train, n, seed,
                           args.lr, args.ee, args.max_epochs)
    train_pred = denormalize(model.forward(x_train)[1], scaler)
    test_pred = denormalize(model.forward(x_test)[1], scaler)
    return {'p': p, 'n': n, 'seed': seed, 'noise': noise,
            'epochs': len(history)-1, 'Es': history[-1],
            'converged': history[-1] <= args.ee, 'history': history,
            'train_count': len(y_train), 'test_count': len(y_test),
            'train': metrics(series[p:SPLIT], train_pred),
            'test': metrics(series[SPLIT:], test_pred),
            'train_prediction': train_pred.tolist(),
            'test_prediction': test_pred.tolist(), 'weights': model.to_list()}


def restore(result):
    model = MLP(result['p'], result['n'], result['seed'])
    for param, value in zip(model.parameters(), result['weights']):
        param[...] = value
    return model


def recursive_forecast(model, initial_window, scaler, steps=30):
    window = normalize(initial_window, scaler).copy()
    result = []
    for _ in range(steps):
        prediction = float(model.forward(window)[1])
        result.append(float(denormalize(prediction, scaler)))
        window[:-1] = window[1:]
        window[-1] = prediction  # Только прогноз, без подстановки эталона.
    return np.array(result)


def predict_window(model, values, scaler):
    values = np.asarray(values, dtype=float)
    if values.shape != (model.w1.shape[0],) or not np.isfinite(values).all():
        raise ValueError(f'Нужно {model.w1.shape[0]} конечных чисел.')
    normalized = normalize(values, scaler)
    with np.errstate(over='raise', invalid='raise'):
        prediction = float(model.forward(normalized)[1])
    return {'window': values.tolist(), 'normalized': normalized.tolist(),
            'prediction_normalized': prediction,
            'prediction': float(denormalize(prediction, scaler))}


def demonstrations(model, series, scaler):
    rows = []
    for k in [20, 70, 120, 140, 165, 199]:
        row = predict_window(model, series[k-P_BASE:k], scaler)
        row.update(label=f'k={k}', k=k, truth=float(series[k]),
                   absolute_error=abs(row['prediction']-float(series[k])))
        rows.append(row)
    for label, window in [('Пользовательское', [.1, .2, .4, .3, .1, -.1]),
                          ('С выбросом', [.1, .2, .4, 4., .1, -.1])]:
        row = predict_window(model, window, scaler)
        row.update(label=label, k=None, truth=None, absolute_error=None)
        rows.append(row)
    return rows


def manual_step(lr=.05):
    """Параметры из задания для V=1; тот же градиент MSE, что в обучении."""
    model = MLP(3, 2, 0)
    model.w1[:] = [[.05, -.2], [.3, .05], [-.1, .4]]
    model.b1[:] = [.1, -.1]
    model.w2[:] = [.5, -.2]; model.b2[...] = .2
    x, target = np.array([.2, .5, .8]), .6
    a = x @ model.w1 + model.b1
    h, pred = model.forward(x)
    delta = 2.*(pred-target)
    hidden_delta = delta*model.w2*h*(1-h)
    gradients = model.gradients(x, target)
    before = model.to_list()
    model.step(x, target, lr)
    return {'lr': lr, 'a': a.tolist(), 'h': h.tolist(), 'prediction': float(pred),
            'loss': float((pred-target)**2), 'delta': float(delta),
            'hidden_delta': hidden_delta.tolist(), 'before': before,
            'gradients': [v.tolist() for v in gradients], 'after': model.to_list(),
            'prediction_after': float(model.forward(x)[1])}


def summary(rows):
    epochs = np.array([r['epochs'] for r in rows])
    maes = np.array([r['test']['MAE'] for r in rows])
    return {'p': rows[0]['p'], 'n': rows[0]['n'], 'runs': len(rows),
            'epochs_mean': float(epochs.mean()), 'epochs_std': float(epochs.std()),
            'epochs_min': int(epochs.min()), 'epochs_max': int(epochs.max()),
            'test_mae_mean': float(maes.mean()), 'test_mae_std': float(maes.std()),
            'test_mae_min': float(maes.min()), 'test_mae_max': float(maes.max()),
            'train_mae_mean': float(np.mean([r['train']['MAE'] for r in rows])),
            'failures': sum(not r['converged'] for r in rows)}


def visualize(data, out, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    series = np.array(data['series']); base = data['base']
    def save(fig, filename):
        fig.savefig(out/filename, dpi=180, bbox_inches='tight')
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    ax.plot(series, color='black', lw=1.2, label='Эталон')
    ax.plot(range(P_BASE, SPLIT), base['train_prediction'], '--', label='Прогноз train')
    ax.plot(range(SPLIT, LENGTH), base['test_prediction'], '--', label='Прогноз test')
    ax.axvline(139.5, color='gray', ls=':', label='Граница train/test')
    ax.set(xlabel='k', ylabel='x(k)', title='Базовая сеть 6–4–1: одношаговый прогноз')
    ax.legend(ncol=2); ax.grid(alpha=.25); save(fig, 'series.png')
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for p in [2, P_BASE, 2*P_BASE]:
        row = next(r for r in data['runs'] if r['p']==p and r['n']==N_BASE
                   and r['seed']==data['parameters']['seeds'][0])
        ax.semilogy(row['history'], label=f'p={p}, N=4')
    ax.axhline(data['parameters']['ee'], color='black', ls='--', label='Ee')
    ax.set(xlabel='Эпоха', ylabel='Es = MSE (нормированная шкала)')
    ax.legend(); ax.grid(alpha=.25); save(fig, 'convergence.png')
    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    ax.scatter(series[SPLIT:], base['test_prediction'], s=23)
    low = min(series[SPLIT:].min(), min(base['test_prediction']))
    high = max(series[SPLIT:].max(), max(base['test_prediction']))
    ax.plot([low, high], [low, high], 'k--', label='y = x')
    ax.set(xlabel='Эталон', ylabel='Прогноз', title='Тест: базовая сеть', aspect='equal')
    ax.legend(); ax.grid(alpha=.25); save(fig, 'scatter.png')
    rec = data['recursive']; steps = np.arange(1, 31)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), layout='constrained')
    axes[0].plot(range(140, 170), series[140:170], 'k-o', ms=3, label='Эталон')
    axes[0].plot(range(140, 170), rec['prediction'], '--o', ms=3, label='Рекурсивный')
    axes[0].set(xlabel='k', ylabel='x(k)',
                title=f'Сеть {rec["p"]}–{rec["n"]}–1, seed={rec["seed"]}')
    axes[1].plot(steps, rec['absolute_error'], '-o', ms=3, label='Рекурсивный')
    axes[1].plot(steps, rec['one_step_error'], '--', label='Одношаговый')
    axes[1].axhline(rec['threshold'], color='red', ls=':', label='5% размаха ряда')
    axes[1].set(xlabel='Шаг (1 соответствует k=140)', ylabel='Абсолютная ошибка')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.25)
    save(fig, 'recursive.png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, field, key, xlabel in [(axes[0], 'window_summary', 'p', 'Окно p'),
                                  (axes[1], 'hidden_summary', 'n', 'Нейроны N')]:
        rows = data[field]
        ax.errorbar([r[key] for r in rows], [r['test_mae_mean'] for r in rows],
                    yerr=[r['test_mae_std'] for r in rows], fmt='o-', capsize=5)
        ax.set(xlabel=xlabel, ylabel='Тестовая MAE, среднее ± σ')
        ax.grid(alpha=.25)
    save(fig, 'studies.png')
    calculation_sheets(data, out)
    if show:
        plt.show()
    plt.close('all')


def calculation_sheets(data, out):
    """Электронные расчётные листы; не имитация фотографий рукописи."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    settings = data.get('parameters', {'ee': .0001, 'max_epochs': 20000})
    header = 'Семёнов А. В.  |  ИИ-26(2)  |  Вариант 1  |  26.10.2026'
    fig, ax = plt.subplots(figsize=(8.2, 10.4))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    ax.text(.5, .98, header, ha='center', va='top', fontsize=11)
    ax.text(.5, .935, 'Блок-схема обработки временного ряда', ha='center', fontsize=14)
    labels = ['Генерация ряда: k = 0…199', 'Train/Test split: 0…139 / 140…199',
              'Min-Max train → нормализация обеих частей',
              'Sliding window: вход k−p…k−1, цель k',
              'MLP p–N–1: скрытая сигмоида, линейный выход',
              'Перемешать train → Online backprop по всем примерам',
              'Es = MSE на train при весах конца эпохи',
              f'Stop: Es ≤ {settings["ee"]:g} или {settings["max_epochs"]} эпох?',
              'Одношаговый прогноз / Recursive forecast',
              'Denormalization → MAE, RMSE, sMAPE']
    ys = np.linspace(.86, .16, len(labels))
    for i,(label,y) in enumerate(zip(labels,ys)):
        ax.add_patch(FancyBboxPatch((.08,y-.024), .84,.048,
                     boxstyle='round,pad=0.008',facecolor='white',edgecolor='#26364a'))
        ax.text(.5,y,label,ha='center',va='center',fontsize=10)
        if i:
            ax.annotate('',xy=(.5,y+.032),xytext=(.5,ys[i-1]-.032),
                        arrowprops={'arrowstyle':'->','color':'#26364a'})
    ax.plot([.93,.98,.98,.93],[ys[7],ys[7],ys[5],ys[5]],color='#26364a')
    ax.annotate('',xy=(.92,ys[5]),xytext=(.98,ys[5]),arrowprops={'arrowstyle':'->'})
    ax.text(.98,(ys[5]+ys[7])/2,'нет',rotation=90,ha='right',va='center',fontsize=9)
    ax.text(.53,(ys[7]+ys[8])/2,'да',fontsize=9)
    ax.text(.5,.065,'Тест не участвует в градиентах и критерии остановки.\n'
            'В рекурсии окно пополняется прогнозом, а не эталоном.',ha='center',fontsize=10)
    ax.text(.5,.015,'Электронный расчётный лист',ha='center',fontsize=9,color='#555555')
    fig.savefig(out/'pipeline.png',dpi=180,bbox_inches='tight')
    m=data['manual']; w,b,v,bo=m['after']; gw,gb,gv,gbo=m['gradients']
    lines = [
        'X = [0.2; 0.5; 0.8],  d = 0.6,  V = 1',
        f'η = {m["lr"]:g};  L = (ŷ − d)²;  θ_new = θ − η · ∂L/∂θ',
        'W = [[0.05, −0.2], [0.3, 0.05], [−0.1, 0.4]]',
        'b = [0.1, −0.1];  v = [0.5, −0.2];  b_out = 0.2',
        '', '1. Прямой проход',
        'a₁ = 0.2·0.05 + 0.5·0.3 + 0.8·(−0.1) + 0.1 = 0.18',
        'a₂ = 0.2·(−0.2) + 0.5·0.05 + 0.8·0.4 − 0.1 = 0.205',
        f'h₁ = σ(a₁) = {m["h"][0]:.9f};  h₂ = σ(a₂) = {m["h"][1]:.9f}',
        f'ŷ = 0.5h₁ − 0.2h₂ + 0.2 = {m["prediction"]:.9f}',
        f'L = (ŷ − 0.6)² = {m["loss"]:.9f}',
        '', '2. Обратный проход со старыми выходными весами',
        f'δ_out = 2(ŷ − d) = {m["delta"]:.9f}',
        f'δ₁ = δ_out · 0.5 · h₁(1−h₁) = {m["hidden_delta"][0]:.9f}',
        f'δ₂ = δ_out · (−0.2) · h₂(1−h₂) = {m["hidden_delta"][1]:.9f}',
        f'∂L/∂v = [{gv[0]:.9f}; {gv[1]:.9f}],  ∂L/∂b_out = δ_out',
        '∂L/∂Wᵢⱼ = Xᵢδⱼ;  ∂L/∂bⱼ = δⱼ',
        *[f'∂L/∂W, строка {i+1}: [{row[0]:.9f}; {row[1]:.9f}]'
          for i,row in enumerate(gw)],
        '', '3. Обновлённые параметры',
        *[f'W_new, строка {i+1}: [{row[0]:.9f}; {row[1]:.9f}]'
          for i,row in enumerate(w)],
        f'b_new = [{b[0]:.9f}; {b[1]:.9f}]',
        f'v_new = [{v[0]:.9f}; {v[1]:.9f}],  b_out_new = {bo:.9f}',
        f'Проверка: ŷ_new = {m["prediction_after"]:.9f}',
        f'L_new = {(m["prediction_after"]-.6)**2:.9f} < {m["loss"]:.9f}']
    fig, ax=plt.subplots(figsize=(8.2,10.4)); ax.axis('off')
    ax.text(.5,1.,header,ha='center',va='top',fontsize=11)
    ax.text(.5,.96,'Один шаг обучения сети 3–2–1',ha='center',fontsize=14)
    for y,line in zip(np.linspace(.915,.06,len(lines)),lines):
        ax.text(.015,y,line,fontsize=10,va='center',fontfamily='DejaVu Sans')
    ax.text(.5,.01,'Электронный расчётный лист',ha='center',fontsize=9,color='#555555')
    fig.savefig(out/'manual_step.png',dpi=180,bbox_inches='tight')


def interactive(model, series, scaler):
    p = model.w1.shape[0]
    print(f'\nБазовая сеть: введите {p} чисел от старого к новому.')
    print('Или k 140: прогноз x(140) по значениям x(134)..x(139). q — выход.')
    while True:
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if line.lower() in {'q', 'exit', 'выход'}:
            break
        try:
            parts = line.replace(',', '.').split()
            k = None
            if parts and parts[0].lower() == 'k':
                if len(parts) != 2:
                    raise ValueError('Формат: k 140')
                k = int(parts[1])
                if not p <= k < LENGTH:
                    raise ValueError(f'Индекс цели k должен быть от {p} до 199.')
                values = series[k-p:k]
            else:
                values = list(map(float, parts))
            row = predict_window(model, values, scaler)
            print('Нормированные входы:', np.round(row['normalized'], 6))
            print(f'Прогноз норм.: {row["prediction_normalized"]:.4f}; '
                  f'исходный: {row["prediction"]:.8f}')
            if k is not None:
                print(f'Эталон x({k})={series[k]:.8f}; '
                      f'ошибка={abs(row["prediction"]-series[k]):.8f}')
        except (ValueError, FloatingPointError, OverflowError) as exc:
            print(f'Ошибка ввода или вычисления: {exc}')


def self_test():
    series = generate_series(); scaler = fit_scaler(series[:SPLIT])
    z = normalize(series, scaler)
    assert np.allclose(denormalize(z, scaler), series)
    for p in [2, 4, 6, 12]:
        x, y = make_windows(z, p, p, SPLIT)
        tx, ty = make_windows(z, p, SPLIT, LENGTH)
        assert len(y)==140-p and len(ty)==60
        assert np.array_equal(tx[0], z[140-p:140]) and ty[0]==z[140]
    model = MLP(3, 2, 42); x = np.array([.2, .5, .8]); t = .6
    grads = model.gradients(x, t); max_error = 0.
    for param, grad in zip(model.parameters(), grads):
        for idx in np.ndindex(param.shape):
            old = float(param[idx]); eps = 1e-6
            param[idx] = old+eps; plus = float((model.forward(x)[1]-t)**2)
            param[idx] = old-eps; minus = float((model.forward(x)[1]-t)**2)
            param[idx] = old
            max_error = max(max_error, abs((plus-minus)/(2*eps)-grad[idx]))
    assert max_error < 1e-8
    assert metrics(np.array([0., 1.]), np.array([0., 1.])) == dict(MAE=0., RMSE=0., sMAPE=0.)
    assert metrics(np.array([0.]), np.array([1.]))['sMAPE']==200.
    x, y = make_windows(z, 6, 6, 140)
    a, ha = train(x, y, 4, 42, .05, 1e-8, 3)
    b, hb = train(x, y, 4, 42, .05, 1e-8, 3)
    assert ha==hb and a.to_list()==b.to_list() and len(ha)==4
    assert ha[-1]==mse(a,x,y)
    _, early = train(x, y, 4, 42, .05, 100., 3)
    assert len(early)==1
    first = recursive_forecast(a, series[134:140], scaler, 1)[0]
    assert first==predict_window(a, series[134:140], scaler)['prediction']
    for bad in [[1.], [float('nan')]*6, [float('inf')]*6]:
        try:
            predict_window(a, bad, scaler)
        except ValueError:
            pass
        else:
            raise AssertionError('Недопустимый ввод принят')
    step = manual_step()
    assert np.allclose(step['a'], [.18, .205])
    assert (step['prediction_after']-.6)**2 < step['loss']
    print(f'Проверки пройдены; ошибка градиента {max_error:.3e}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr', type=float, default=.05)
    parser.add_argument('--ee', type=float, default=.0001)
    parser.add_argument('--max-epochs', type=int, default=20000)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'lab4_results')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--no-input', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if (not np.isfinite([args.lr, args.ee]).all() or args.lr<=0 or args.ee<=0
            or args.max_epochs<1 or len(args.seeds)<5 or min(args.seeds)<0
            or len(set(args.seeds))!=len(args.seeds)):
        parser.error('lr и ee > 0; max-epochs >= 1; >=5 различных seed >=0.')
    args.out.mkdir(parents=True, exist_ok=True)
    series = generate_series(); scaler = fit_scaler(series[:SPLIT])
    configs = [(p,N_BASE) for p in [2,4,6,12]]
    configs += [(P_BASE,n) for n in [1,2,8]]  # 6–4–1 уже включена.
    runs = []
    for p,n in configs:
        for seed in args.seeds:
            row = run_experiment(series, scaler, p, n, seed, args)
            runs.append(row)
            print(f'p={p:2} N={n} seed={seed}: эпох={row["epochs"]:5}, '
                  f'Es={row["Es"]:.7f}, test MAE={row["test"]["MAE"]:.6f}, '
                  f'порог={row["converged"]}', flush=True)
    groups = {(p,n): [r for r in runs if r['p']==p and r['n']==n] for p,n in configs}
    summaries = {key: summary(rows) for key,rows in groups.items()}
    best_key = min(configs, key=lambda key: summaries[key]['test_mae_mean'])
    # Конфигурация выбирается по среднему пяти запусков, seed — заранее первый.
    best = groups[best_key][0]; base = groups[(P_BASE,N_BASE)][0]
    prediction = recursive_forecast(restore(best), series[140-best['p']:140], scaler)
    error = np.abs(prediction-series[140:170])
    threshold = .05*float(np.ptp(series))
    exceed = np.flatnonzero(error>threshold)
    noise_runs = []
    for seed in args.seeds:
        row = run_experiment(series, scaler, 6, 4, seed, args, noise=.01)
        noise_runs.append(row)
        print(f'Шум 0.01, seed={seed}: эпох={row["epochs"]}, '
              f'test MAE={row["test"]["MAE"]:.6f}', flush=True)
    data = {'parameters': {'lr': args.lr, 'ee': args.ee, 'max_epochs': args.max_epochs,
                           'seeds': args.seeds}, 'numpy_version': np.__version__,
            'series': series.tolist(), 'scaler': scaler, 'runs': runs, 'base': base,
            'window_summary': [summaries[(p,4)] for p in [2,4,6,12]],
            'hidden_summary': [summaries[(6,n)] for n in [1,2,4,8]],
            'noise_runs': noise_runs, 'noise_summary': summary(noise_runs),
            'recursive': {'p': best['p'], 'n': best['n'], 'seed': best['seed'],
                          'prediction': prediction.tolist(), 'absolute_error': error.tolist(),
                          'one_step_error': np.abs(np.array(best['test_prediction'][:30])
                                                  -series[140:170]).tolist(),
                          'threshold': threshold,
                          'first_exceed_step': int(exceed[0]+1) if len(exceed) else None,
                          'metrics': metrics(series[140:170], prediction),
                          'one_step_metrics': metrics(series[140:170], best['test_prediction'][:30])},
            'demonstrations': demonstrations(restore(base),series,scaler),
            'manual': manual_step(args.lr)}
    (args.out/'results.json').write_text(json.dumps(data, ensure_ascii=False,
                              indent=2, allow_nan=False), encoding='utf-8')
    for name in ['window_summary','hidden_summary']:
        with (args.out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[name][0]))
            writer.writeheader(); writer.writerows(data[name])
    print('\nБазовый эксперимент:', {k:base[k] for k in ['epochs','Es','train','test']})
    print('Лучшая конфигурация по среднему test MAE:',best_key)
    print('Первое превышение 5% размаха, шаг:',data['recursive']['first_exceed_step'])
    for row in data['demonstrations']:
        print('\n',row['label'],'входы:',np.round(row['window'],6))
        print('Нормированные:',np.round(row['normalized'],6))
        print(f'Прогноз норм.={row["prediction_normalized"]:.4f}; '
              f'исходный={row["prediction"]:.8f}; эталон={row["truth"]}; '
              f'ошибка={row["absolute_error"]}')
    visualize(data,args.out,not args.no_show)
    print('Результаты сохранены:',args.out.resolve())
    if not args.no_input:
        interactive(restore(base),series,scaler)


if __name__=='__main__':
    main()
