"""Lab 2 (variant 6). Compare original-scale MSE with BCE and scaled MSE."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from xor_engine import MLP, X,Y,Y_RAW,C0,C1,stats,inference,TEST_PAIRS
SEEDS=[6,7,11,14,18]

def fit_many(name,loss,targets,max_epochs):
    runs=[]
    for seed in SEEDS:
        model=MLP(seed=seed,loss=loss,lr=.5)
        r=model.fit(X,targets,ee=.01,max_epochs=max_epochs)
        s=stats(model,raw_output=(name=='raw MSE'))
        print(name,seed,r['epochs'],r['Es'],s['accuracy'],s['mae'])
        runs.append((model,r,s))
    return runs

def plot_results(results,path):
    plt.figure(figsize=(8,5))
    for key in ('raw MSE','BCE','scaled MSE diagnostic'):
        h=results[key][0][1]['history']
        plt.plot(range(1,len(h)+1),h,label=key)
    plt.axhline(.01,linestyle='--',label='Ee=0.01')
    plt.yscale('log');plt.xlabel('Epoch');plt.ylabel('Total training loss')
    plt.title('Convergence, variant 6');plt.legend();plt.tight_layout()
    plt.savefig(path,dpi=160);plt.close()

def main(max_epochs=20000,interactive=False):
    results={'raw MSE':fit_many('raw MSE','mse',Y_RAW,max_epochs),
             'BCE':fit_many('BCE','bce',Y,max_epochs),
             'scaled MSE diagnostic':fit_many('scaled MSE diagnostic','mse',Y,max_epochs)}
    outdir=Path(__file__).parent/'results';outdir.mkdir(exist_ok=True)
    plot_results(results,outdir/'convergence_runtime.png')
    summary={name:[{'seed':SEEDS[i],'epochs':r['epochs'],'Es':r['Es'],
      'converged':r['converged'],**s} for i,(_,r,s) in enumerate(runs)]
      for name,runs in results.items()}
    with open(outdir/'latest_run.json','w',encoding='utf8') as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    if interactive:
        model=results['BCE'][0][0]
        while True:
            a=input('A (or q): ').strip()
            if a.lower()=='q':break
            try:print(inference(model,float(a),float(input('B: '))))
            except ValueError as e:print(e)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');ap.add_argument('--interactive',action='store_true')
    args=ap.parse_args();main(1000 if args.quick else 20000,args.interactive)
