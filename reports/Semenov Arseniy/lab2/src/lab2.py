

import argparse
import json
from pathlib import Path

import numpy as np

C0, C1 = 0., -6.
RAW_X = np.array([[0., 0.], [0., -6.], [-6., 0.], [-6., -6.]])
RAW_Y = np.array([0., -6., -6., 0.])
CHECK_X = np.vstack((RAW_X, [[-1., -5.], [-5., -1.], [-1., -1.], [-5., -5.]]))
NAMES = {'MSE': 'А: MSE', 'BCE': 'Б: BCE', 'RAW': 'A_raw: MSE на 0/-6'}


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
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.w1 = rng.uniform(-.5, .5, (2, 2))
        self.b1 = rng.uniform(-.5, .5, 2)
        self.w2 = rng.uniform(-.5, .5, 2)
        self.b2 = np.array(rng.uniform(-.5, .5))

    def forward(self, x):
        h = sigmoid(np.asarray(x) @ self.w1 + self.b1)
        z = h @ self.w2 + self.b2
        return h, z, sigmoid(z)

    def gradients(self, x, target, loss):
        h, _, p = self.forward(x)
        # Отличие BCE: производная сигмоиды сокращается с dL/dp.
        delta = p - target
        if loss != 'BCE':
            delta *= p * (1. - p)
        # Все градиенты вычисляются ДО обновления выходных весов.
        hidden_delta = delta * self.w2 * h * (1. - h)
        return (np.outer(x, hidden_delta), hidden_delta,
                delta * h, np.asarray(delta))

    def parameters(self):
        return self.w1, self.b1, self.w2, self.b2

    def step(self, x, target, loss, lr):
        grads = self.gradients(x, target, loss)
        for param, grad in zip(self.parameters(), grads):
            param -= lr * grad


def loss_sum(model, x, targets, loss):
    _, z, p = model.forward(x)
    if loss == 'BCE':
        # BCE из логитов: без log(0), clipping и переполнения exp.
        return float(np.sum(np.maximum(z, 0) - targets * z
                            + np.log1p(np.exp(-np.abs(z)))))
    return float(.5 * np.sum((p - targets) ** 2))


def metrics(model, loss):
    targets = RAW_Y if loss == 'RAW' else normalize(RAW_Y)
    p = model.forward(normalize(RAW_X))[2]
    real = p if loss == 'RAW' else decode(p)
    return {'Es': loss_sum(model, normalize(RAW_X), targets, loss),
            'accuracy': float(np.mean(nearest(real) == RAW_Y)),
            'mae': float(np.mean(np.abs(real - RAW_Y))),
            'p': p.tolist(), 'real': real.tolist()}


def train(loss, seed, lr, ee, max_epochs):
    model = MLP(seed)
    x = normalize(RAW_X)
    targets = RAW_Y if loss == 'RAW' else normalize(RAW_Y)
    history = [loss_sum(model, x, targets, loss)]
    # Общие диагностические критерии не управляют остановкой.
    first_acc = 0 if metrics(model, loss)['accuracy'] == 1 else None
    first_mae = 0 if metrics(model, loss)['mae'] <= .1 else None
    for epoch in range(1, max_epochs + 1):
        if history[-1] <= ee:
            break
        for row, target in zip(x, targets):
            model.step(row, target, loss, lr)
        m = metrics(model, loss)
        history.append(m['Es'])
        if first_acc is None and m['accuracy'] == 1:
            first_acc = epoch
        if first_mae is None and m['mae'] <= .1:
            first_mae = epoch
    m = metrics(model, loss)
    m.update(seed=seed, epochs=len(history)-1, converged=m['Es'] <= ee,
             history=history, first_accuracy_100=first_acc,
             first_mae_le_01=first_mae,
             weights=[a.tolist() for a in model.parameters()])
    return model, m


def predict(model, a, b):
    pair = np.asarray([a, b], dtype=float)
    if not np.all(np.isfinite(pair)) or np.any(np.abs(pair) > 10):
        raise ValueError('A и B должны быть конечными числами из [-10; 10].')
    p = float(model.forward(normalize(pair))[2])
    real = float(decode(p))
    return {'A': float(a), 'B': float(b), 'p': p,
            'real': real, 'class': float(nearest(real))}


def visualize(results, models, ee, out, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    colors = {'MSE': '#2466a4', 'BCE': '#bf4c24'}
    def save(fig, name):
        fig.savefig(out / name, dpi=180, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 4.3), layout='constrained')
    for loss in ('MSE', 'BCE'):
        h = results[loss][0]['history']
        ax.semilogy(np.arange(len(h)), h, label=NAMES[loss], color=colors[loss])
    ax.axhline(ee, color='black', ls='--', label=f'Ee = {ee:g}')
    ax.set(xlabel='Эпоха', ylabel='Суммарная ошибка Es (логарифмическая шкала)',
           title=f'Сходимость, seed = {results["MSE"][0]["seed"]}')
    ax.grid(alpha=.25); ax.legend(); save(fig, 'convergence.png')

    fig, ax = plt.subplots(figsize=(8, 4.3), layout='constrained')
    pos = np.arange(len(results['MSE']))
    for k, loss in enumerate(('MSE', 'BCE')):
        runs = results[loss]
        bars = ax.bar(pos + (k-.5)*.36, [r['epochs'] for r in runs],
                      .36, label=NAMES[loss], color=colors[loss])
        for bar, r in zip(bars, runs):
            if not r['converged']:
                bar.set_hatch('///')
            ax.annotate(str(r['epochs']), (bar.get_x()+bar.get_width()/2,
                        bar.get_height()), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(pos, [str(r['seed']) for r in results['MSE']])
    ax.set(xlabel='Seed', ylabel='Эпохи до остановки',
           title='Устойчивость обучения (штриховка — порог не достигнут)')
    ax.margins(y=.18); ax.legend(); save(fig, 'epochs.png')

    grid = np.linspace(-10, 10, 241)
    aa, bb = np.meshgrid(grid, grid)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for ax, loss in zip(axes, ('MSE', 'BCE')):
        model = models[loss]
        values = decode(model.forward(normalize(np.c_[aa.ravel(), bb.ravel()]))[2])
        values = values.reshape(aa.shape)
        im = ax.pcolormesh(aa, bb, values, cmap='viridis', vmin=-6, vmax=0,
                           shading='auto')
        if values.min() < -3 < values.max():
            ax.contour(aa, bb, values, levels=[-3], colors='white',
                       linewidths=1.5, linestyles='solid')
        for cls, marker, color in [(0, 'o', 'red'), (-6, 's', 'cyan')]:
            pts = RAW_X[RAW_Y == cls]
            ax.scatter(pts[:, 0], pts[:, 1], marker=marker, c=color, s=65,
                       edgecolor='black', label=f'Истинный класс {cls}')
        ax.plot([], [], color='white', label='Граница y = −3')
        ax.set(xlabel='A', ylabel='B', title=NAMES[loss], xlim=(-10, 10), ylim=(-10, 10))
        ax.set_aspect('equal'); ax.legend(fontsize=8, facecolor='#dddddd')
    fig.colorbar(im, ax=axes, label='Выход в исходной шкале y = −6p', shrink=.85)
    fig.suptitle('Карта выхода сети после скрытого слоя')
    save(fig, 'surfaces.png')

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    for ax, mean in zip(axes, (False, True)):
        vals = [np.mean([r['mae'] for r in results[l]]) if mean
                else results[l][0]['mae'] for l in ('MSE', 'BCE')]
        bars = ax.bar(['А: MSE', 'Б: BCE'], vals, color=list(colors.values()))
        ax.bar_label(bars, fmt='%.5f', padding=3)
        ax.set(ylabel='MAE в исходной шкале', xlabel='Конфигурация',
               title='Среднее по всем seed' if mean else 'Первый seed')
        ax.margins(y=.2)
    save(fig, 'mae.png')
    if show:
        plt.show()
    plt.close('all')


def self_test():
    max_error = 0.
    x = np.array([.2, .8])
    for loss, target in [('MSE', 1.), ('BCE', 1.), ('BCE', 0.), ('RAW', -6.)]:
        model = MLP(42)
        grads = model.gradients(x, target, loss)
        for param, grad in zip(model.parameters(), grads):
            for idx in np.ndindex(param.shape):
                old = float(param[idx]); eps = 1e-6
                param[idx] = old + eps
                plus = loss_sum(model, x, target, loss)
                param[idx] = old - eps
                minus = loss_sum(model, x, target, loss)
                param[idx] = old
                max_error = max(max_error, abs((plus-minus)/(2*eps)-grad[idx]))
    assert max_error < 1e-7, max_error
    assert np.array_equal(decode(normalize(RAW_Y)), RAW_Y)
    assert float(nearest(-3.)) == C0
    model = MLP(42)
    for bad in [(float('nan'), 0), (0, float('inf')), (-10.01, 0)]:
        try:
            predict(model, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Недопустимый ввод принят')
    predict(model, -10, 10)
    for loss in ('MSE', 'BCE', 'RAW'):
        _, a = train(loss, 42, .5, .01, 10)
        _, b = train(loss, 42, .5, .01, 10)
        assert a == b
        assert a['epochs'] == 10 and len(a['history']) == 11
        assert a['Es'] == a['history'][-1]
    # Устойчивость вычисления BCE при больших логитах.
    for z in (-1000., 1000.):
        model.w2[:] = 0; model.b2[...] = z
        for target in (0., 1.):
            assert np.isfinite(loss_sum(model, x, target, 'BCE'))
    print(f'Проверки пройдены; максимальная ошибка градиента {max_error:.3e}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--lr', type=float, default=.5)
    parser.add_argument('--ee', type=float, default=.01)
    parser.add_argument('--max-epochs', type=int, default=50000)
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).resolve().parent / 'lab2_results')
    parser.add_argument('--no-input', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if (not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.ee)
            or args.ee <= 0 or args.max_epochs < 1 or min(args.seeds) < 0
            or len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 5):
        parser.error('Нужны >=5 разных seed >=0, конечные lr>0, ee>0, max-epochs>=1.')
    args.out.mkdir(parents=True, exist_ok=True)
    results, models = {}, {}
    for loss in ('MSE', 'BCE', 'RAW'):
        results[loss] = []
        for seed in args.seeds:
            model, r = train(loss, seed, args.lr, args.ee, args.max_epochs)
            results[loss].append(r)
            if seed == args.seeds[0]:
                models[loss] = model
            print(f'{NAMES[loss]:24} seed={seed} эпох={r["epochs"]:5d} '
                  f'Es={r["Es"]:.9f} acc={r["accuracy"]:.0%} '
                  f'MAE={r["mae"]:.6f} критерий={r["converged"]}', flush=True)
    checks = {loss: [predict(models[loss], *pair) for pair in CHECK_X]
              for loss in ('MSE', 'BCE')}
    data = {'parameters': {'seeds': args.seeds, 'lr': args.lr, 'ee': args.ee,
                           'max_epochs': args.max_epochs},
            'numpy_version': np.__version__, 'results': results, 'checks': checks}
    (args.out / 'results.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    for loss, rows in checks.items():
        print('\n' + NAMES[loss])
        for r in rows:
            print(f'A={r["A"]:g}, B={r["B"]:g}: p={r["p"]:.4f}, '
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
            for loss in ('MSE', 'BCE'):
                r = predict(models[loss], *vals)
                print(f'{NAMES[loss]}: p={r["p"]:.4f}, y={r["real"]:.6f}, '
                      f'ближайший класс={r["class"]:g}')
        except ValueError as exc:
            print(f'Ошибка ввода: {exc}')


if __name__ == '__main__':
    main()
