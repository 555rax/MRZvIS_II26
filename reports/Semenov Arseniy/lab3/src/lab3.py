
import argparse
import csv
import json
from pathlib import Path

import numpy as np

C0, C1 = 0., -6.
RAW_X = np.array([[0., 0.], [0., -6.], [-6., 0.], [-6., -6.]])
RAW_Y = np.array([0., -6., -6., 0.])
CHECK_X = np.vstack((RAW_X, [[-1., -5.], [-1., -1.], [-5., -5.]]))
NAMES = {'sigmoid': 'А: Сигмоида', 'relu': 'В: ReLU'}


def normalize(v):
    return (np.asarray(v, dtype=float) - C0) / (C1 - C0)


def decode(v):
    return C0 + (C1 - C0) * np.asarray(v)


def sigmoid(v):
    v = np.asarray(v, dtype=float)
    e = np.exp(-np.abs(v))
    return np.where(v >= 0, 1. / (1. + e), e / (1. + e))


def nearest(v):
    v = np.asarray(v)
    return np.where(np.abs(v - C0) <= np.abs(v - C1), C0, C1)


class MLP:
    def __init__(self, activation, seed):
        if activation not in NAMES:
            raise ValueError('Неизвестная активация')
        self.activation = activation
        rng = np.random.default_rng(seed)
        self.w1 = rng.uniform(-.5, .5, (2, 2))
        self.b1 = rng.uniform(-.5, .5, 2)
        self.w2 = rng.uniform(-.5, .5, 2)
        self.b2 = np.array(rng.uniform(-.5, .5))

    def forward(self, x):
        a = np.asarray(x) @ self.w1 + self.b1
        h = sigmoid(a) if self.activation == 'sigmoid' else np.maximum(0., a)
        z = h @ self.w2 + self.b2
        return a, h, z, sigmoid(z)

    def parameters(self):
        return self.w1, self.b1, self.w2, self.b2

    def gradients(self, x, target):
        a, h, _, p = self.forward(x)
        delta = p - target  # BCE + сигмоида на выходе
        derivative = h * (1. - h) if self.activation == 'sigmoid' else (a > 0.)
        # В нуле выбираем производную ReLU, равную 0.
        hidden_delta = delta * self.w2 * derivative
        return np.outer(x, hidden_delta), hidden_delta, delta * h, np.asarray(delta)

    def step(self, x, target, lr):
        # Все градиенты вычислены до изменения любого параметра.
        grads = self.gradients(x, target)
        for param, grad in zip(self.parameters(), grads):
            param -= lr * grad

    def dead_units(self):
        # Номера нейронов, не активных ни на одной обучающей точке.
        if self.activation != 'relu':
            return []
        a = self.forward(normalize(RAW_X))[0]
        return (np.flatnonzero(np.all(a <= 0., axis=0)) + 1).tolist()


def loss_sum(model, x, targets):
    z = model.forward(x)[2]
    return float(np.sum(np.maximum(z, 0.) - targets * z
                        + np.log1p(np.exp(-np.abs(z)))))


def metrics(model):
    p = model.forward(normalize(RAW_X))[3]
    real = decode(p)
    return {'Es': loss_sum(model, normalize(RAW_X), normalize(RAW_Y)),
            'accuracy': float(np.mean(nearest(real) == RAW_Y)),
            'mae': float(np.mean(np.abs(real - RAW_Y))),
            'p': p.tolist(), 'real': real.tolist()}


def train(activation, seed, lr, ee, max_epochs):
    model = MLP(activation, seed)
    x, targets = normalize(RAW_X), normalize(RAW_Y)
    initial_dead = model.dead_units()
    first_dead = 0 if initial_dead else None
    history = [loss_sum(model, x, targets)]
    while history[-1] > ee and len(history) - 1 < max_epochs:
        for row, target in zip(x, targets):
            model.step(row, target, lr)
        error = loss_sum(model, x, targets)
        if not np.isfinite(error):
            raise FloatingPointError('Обучение разошлось; уменьшите --lr.')
        history.append(error)
        if activation == 'relu' and first_dead is None and model.dead_units():
            first_dead = len(history) - 1
    result = metrics(model)
    result.update(seed=seed, epochs=len(history)-1, converged=result['Es'] <= ee,
                  history=history, initial_dead=initial_dead,
                  final_dead=model.dead_units(), first_dead_epoch=first_dead,
                  weights=[a.tolist() for a in model.parameters()])
    return model, result


def predict(model, a, b):
    pair = np.asarray([a, b], dtype=float)
    if not np.all(np.isfinite(pair)) or np.any(np.abs(pair) > 10):
        raise ValueError('A и B должны быть конечными числами из [-10; 10].')
    p = float(model.forward(normalize(pair))[3])
    real = float(decode(p))
    return {'A': float(a), 'B': float(b), 'p': p,
            'real': real, 'class': float(nearest(real))}


def summarize(rows):
    successful = [r['epochs'] for r in rows if r['converged']]
    return {'successes': len(successful), 'runs': len(rows),
            'epochs_all_mean': float(np.mean([r['epochs'] for r in rows])),
            'epochs_success_mean': float(np.mean(successful)) if successful else None,
            'epochs_success_std': float(np.std(successful)) if successful else None,
            'epochs_success_min': min(successful) if successful else None,
            'epochs_success_max': max(successful) if successful else None,
            'accuracy_mean': float(np.mean([r['accuracy'] for r in rows])),
            'Es_mean': float(np.mean([r['Es'] for r in rows])),
            'mae_mean': float(np.mean([r['mae'] for r in rows])),
            'mae_std': float(np.std([r['mae'] for r in rows]))}


def visualize(results, models, ee, out, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    colors = ['#2466a4', '#bf4c24']
    def save(fig, name):
        fig.savefig(out / name, dpi=180, bbox_inches='tight')
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for (act, rows), color in zip(results.items(), colors):
        ax.semilogy(rows[0]['history'], label=NAMES[act], color=color)
    ax.axhline(ee, color='black', ls='--', label=f'Ee = {ee:g}')
    ax.set(xlabel='Эпоха', ylabel='Суммарная BCE, Es',
           title=f'Сходимость, seed = {results["sigmoid"][0]["seed"]}')
    ax.legend(); ax.grid(alpha=.25); save(fig, 'convergence.png')
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    pos = np.arange(len(results['sigmoid']))
    for k, (act, rows) in enumerate(results.items()):
        bars = ax.bar(pos + (k-.5)*.36, [r['epochs'] for r in rows],
                      .36, label=NAMES[act], color=colors[k])
        for bar, r in zip(bars, rows):
            if not r['converged']:
                bar.set_hatch('///')
            ax.annotate(str(r['epochs']),
                        (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)
    ax.set_xticks(pos, [str(r['seed']) for r in results['sigmoid']])
    ax.set(xlabel='Seed', ylabel='Выполнено эпох',
           title='Штриховка: порог не достигнут')
    ax.margins(y=.35); ax.legend(); save(fig, 'epochs.png')
    grid = np.linspace(-10, 10, 301)
    aa, bb = np.meshgrid(grid, grid)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout='constrained')
    for ax, (act, model) in zip(axes, models.items()):
        values = decode(model.forward(normalize(np.c_[aa.ravel(), bb.ravel()]))[3])
        values = values.reshape(aa.shape)
        im = ax.pcolormesh(aa, bb, values, cmap='viridis', vmin=-6, vmax=0,
                           shading='auto')
        if values.min() < -3 < values.max():
            ax.contour(aa, bb, values, levels=[-3], colors='white', linewidths=1.4)
        for cls, marker, color in [(0, 'o', 'red'), (-6, 's', 'cyan')]:
            pts = RAW_X[RAW_Y == cls]
            ax.scatter(*pts.T, marker=marker, c=color, s=65,
                       edgecolor='black', label=f'Цель {cls}')
        ax.plot([], [], color='white', label='Граница y = −3')
        ax.set(xlabel='A', ylabel='B', title=NAMES[act],
               xlim=(-10, 10), ylim=(-10, 10), aspect='equal')
        ax.legend(fontsize=8, facecolor='#dddddd', loc='upper right')
    fig.colorbar(im, ax=axes, label='Выход y = −6p', shrink=.8)
    save(fig, 'surfaces.png')
    fig, ax = plt.subplots(figsize=(7, 3.5), layout='constrained')
    bars = ax.bar(list(NAMES.values()),
                  [summarize(rows)['mae_mean'] for rows in results.values()], color=colors)
    ax.bar_label(bars, fmt='%.6f', padding=3)
    ax.set(ylabel='MAE в исходной шкале', title='Среднее по всем запускам')
    ax.margins(y=.2); save(fig, 'mae.png')
    if show:
        plt.show()
    plt.close('all')


def self_test():
    max_error = 0.
    for act in NAMES:
        for target in (0., 1.):
            model = MLP(act, 42)
            # Явно проверяются активная и неактивная ветви ReLU, вдали от 0.
            model.b1[:] = [.8, -1.2]
            x = np.array([.2, .8])
            grads = model.gradients(x, target)
            for param, grad in zip(model.parameters(), grads):
                for idx in np.ndindex(param.shape):
                    old, eps = float(param[idx]), 1e-6
                    param[idx] = old + eps
                    plus = loss_sum(model, x, target)
                    param[idx] = old - eps
                    minus = loss_sum(model, x, target)
                    param[idx] = old
                    max_error = max(max_error, abs((plus-minus)/(2*eps)-grad[idx]))
        _, a = train(act, 42, .5, .01, 10)
        _, b = train(act, 42, .5, .01, 10)
        assert a == b and a['epochs'] == 10
        assert a['Es'] == a['history'][-1]
        _, early = train(act, 42, .5, 100., 10)
        assert early['epochs'] == 0 and early['converged']
    assert max_error < 1e-7, max_error
    assert np.array_equal(decode(normalize(RAW_Y)), RAW_Y)
    assert float(nearest(-3.)) == C0
    model = MLP('relu', 42)
    model.w1[:] = 0.; model.b1[:] = 0.
    assert model.dead_units() == [1, 2]
    assert not np.any(model.gradients(np.ones(2), 1.)[1])
    before = [p.copy() for p in model.parameters()[:3]]
    model.step(np.ones(2), 1., .5)
    assert all(np.array_equal(a, b) for a, b in zip(before, model.parameters()))
    for bad in [(float('nan'), 0), (0, float('inf')), (-10.01, 0)]:
        try:
            predict(model, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Недопустимый ввод принят')
    predict(model, -10., 10.)
    for z in (-1000., 1000.):
        model.w2[:] = 0.; model.b2[...] = z
        for target in (0., 1.):
            assert np.isfinite(loss_sum(model, np.zeros(2), target))
    a, b = MLP('sigmoid', 42), MLP('relu', 42)
    assert all(np.array_equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    # Конструктивная проверка выразительности ReLU 2-2-1 для XOR.
    model.w1[:] = [[1., 1.], [1., 1.]]
    model.b1[:] = [0., -1.]
    model.w2[:] = [20., -40.]; model.b2[...] = -10.
    assert metrics(model)['accuracy'] == 1.
    print(f'Проверки пройдены; ошибка градиента {max_error:.3e}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--lr', type=float, default=.5)
    parser.add_argument('--ee', type=float, default=.01)
    parser.add_argument('--max-epochs', type=int, default=50000)
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).resolve().parent / 'lab3_results')
    parser.add_argument('--no-input', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if (not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.ee)
            or args.ee <= 0 or args.max_epochs < 1 or min(args.seeds) < 0
            or len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 5):
        parser.error('Нужны >=5 разных seed>=0, конечные lr>0, ee>0, max-epochs>=1.')
    args.out.mkdir(parents=True, exist_ok=True)
    results, models = {}, {}
    for act in NAMES:
        results[act] = []
        for seed in args.seeds:
            model, r = train(act, seed, args.lr, args.ee, args.max_epochs)
            results[act].append(r)
            if seed == args.seeds[0]:
                models[act] = model
            print(f'{NAMES[act]:14} seed={seed} эпох={r["epochs"]:5d} '
                  f'Es={r["Es"]:.9f} acc={r["accuracy"]:.0%} MAE={r["mae"]:.6f} '
                  f'порог={r["converged"]} мёртвые={r["final_dead"]}', flush=True)
    stats = {act: summarize(rows) for act, rows in results.items()}
    checks = {act: [predict(models[act], *pair) for pair in CHECK_X] for act in NAMES}
    data = {'parameters': {'seeds': args.seeds, 'lr': args.lr, 'ee': args.ee,
                           'max_epochs': args.max_epochs},
            'numpy_version': np.__version__, 'results': results,
            'summary': stats, 'checks': checks}
    (args.out / 'results.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (args.out / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['activation', *stats['sigmoid']])
        writer.writeheader()
        for act, row in stats.items():
            writer.writerow({'activation': act, **row})
    for act, row in stats.items():
        print(f'\n{NAMES[act]}: порог {row["successes"]}/{row["runs"]}; '
              f'средняя MAE={row["mae_mean"]:.6f}; '
              f'среднее эпох успешных={row["epochs_success_mean"]}')
    for act, rows in checks.items():
        print('\n' + NAMES[act] + f', seed={args.seeds[0]}')
        for r in rows:
            print(f'A={r["A"]:g}, B={r["B"]:g}: p={r["p"]:.6f}, '
                  f'y={r["real"]:.6f}, класс={r["class"]:g}')
    visualize(results, models, args.ee, args.out, not args.no_show)
    print(f'Результаты и графики: {args.out.resolve()}')
    if args.no_input:
        return
    print('Введите A B из [-10; 10], q — выход. Для новых пар эталон не задан.')
    while True:
        try:
            line = input('A B > ').strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if line.lower() in ('q', 'quit', 'exit', 'выход'):
            break
        try:
            vals = list(map(float, line.replace(',', '.').split()))
            if len(vals) != 2:
                raise ValueError('Введите ровно два числа через пробел.')
            for act in NAMES:
                r = predict(models[act], *vals)
                print(f'{NAMES[act]}: p={r["p"]:.6f}, y={r["real"]:.6f}, '
                      f'класс={r["class"]:g}')
        except ValueError as exc:
            print(f'Ошибка ввода: {exc}')


if __name__ == '__main__':
    main()
