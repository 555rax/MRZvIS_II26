"""Lab 4 (variant 6). Full 5-seed window/width studies; execution may take time."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ts_engine import series,fit_scale,normalize,denormalize,windows,ForecastMLP,metrics,recursive,inference
SEEDS=[6,7,11,14,18]

def experiment(p,n,seed,epochs=20000):
    raw=series();lo,hi=fit_scale(raw);scaled=normalize(raw,lo,hi)
    xtr,ytr,itr,xte,yte,ite=windows(scaled,p)
    model=ForecastMLP(p,n,seed=seed,lr=.08)
    result=model.fit(xtr,ytr,ee=1e-4,max_epochs=epochs,seed=seed)
    train_metrics=metrics(raw[itr],denormalize(model.predict(xtr),lo,hi))
    test_pred=denormalize(model.predict(xte),lo,hi)
    test_metrics=metrics(raw[ite],test_pred)
    return model,result,train_metrics,test_metrics,ite,test_pred,raw,lo,hi

def main(max_epochs=20000,interactive=False):
    configs=[(p,4) for p in (2,4,6,12)]+[(6,n) for n in (1,2,8)]
    outdir=Path(__file__).parent/'results';outdir.mkdir(exist_ok=True)
    summary={};best=None
    for p,n in configs:
        for seed in SEEDS:
            model,train,tr,te,k,y,raw,lo,hi=experiment(p,n,seed,max_epochs)
            key=f'{p}_{n}_{seed}'
            summary[key]={'p':p,'N':n,'seed':seed,'epochs':train['epochs'],
              'Es':train['Es'],'converged':train['converged'],
              'train':tr,'test':te}
            print(key,summary[key])
            if best is None or te['MAE']<best[0]:best=(te['MAE'],model,raw,lo,hi)
    with open(outdir/'latest_run.json','w',encoding='utf8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    _,model,raw,lo,hi=best
    rec=denormalize(recursive(model,normalize(raw,lo,hi),140,30),lo,hi)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(np.arange(140,170),raw[140:170],label='True')
    ax.plot(np.arange(140,170),rec,label='Recursive')
    ax.set(xlabel='k',ylabel='x(k)',title='30-step forecast');ax.legend()
    fig.tight_layout();fig.savefig(outdir/'recursive_runtime.png',dpi=160);plt.close(fig)
    if interactive:
        while True:
            v=input(f'Enter k or q (p={model.p}): ').strip()
            if v.lower()=='q':break
            try:print(inference(model,raw,lo,hi,k=int(v)))
            except ValueError as e:print(e)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--interactive',action='store_true')
    a=p.parse_args();main(500 if a.quick else 20000,a.interactive)
