"""Lab 3 (variant 6). Compare sigmoid and ReLU hidden activation."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib.pyplot as plt
from xor_engine import MLP,X,Y,X_RAW,Y_RAW,stats,inference
SEEDS=[6,7,11,14,18]

def run(activation,max_epochs):
    runs=[]
    for seed in SEEDS:
        m=MLP(seed=seed,activation=activation,loss='bce',lr=.5)
        r=m.fit(X,Y,ee=.01,max_epochs=max_epochs)
        s=stats(m)
        inactive=int(np.all((X@m.w1+m.b1)<=0,axis=0).sum()) if activation=='relu' else 0
        print(activation,seed,r['epochs'],r['Es'],s['accuracy'],'dead=',inactive)
        runs.append((m,r,s,inactive))
    return runs

def heatmap(model,name,out):
    grid=np.linspace(-10,10,120)
    a,b=np.meshgrid(grid,grid)
    xy=(np.stack([a.ravel(),b.ravel()],axis=-1)+9)/15
    vals=model.predict(xy).reshape(a.shape)
    fig,ax=plt.subplots(figsize=(6,5))
    img=ax.contourf(a,b,vals,levels=25,vmin=0,vmax=1)
    ax.scatter(X_RAW[:,0],X_RAW[:,1],c=(Y_RAW==6).astype(int),edgecolor='black',s=70)
    ax.set(xlabel='A',ylabel='B',title=name);fig.colorbar(img,ax=ax,label='P(c1)')
    fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)

def main(max_epochs=20000,interactive=False):
    res={'sigmoid':run('sigmoid',max_epochs),'relu':run('relu',max_epochs)}
    d=Path(__file__).parent/'results';d.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for name,runs in res.items():ax.plot(runs[0][1]['history'],label=name)
    ax.set_yscale('log');ax.axhline(.01,linestyle='--',label='Ee')
    ax.set(xlabel='Epoch',ylabel='Total BCE',title='Activation comparison')
    ax.legend();fig.tight_layout();fig.savefig(d/'convergence_runtime.png',dpi=150);plt.close(fig)
    for name,runs in res.items():heatmap(runs[0][0],name,d/f'heatmap_{name}_runtime.png')
    summary={name:[{'seed':SEEDS[i],'epochs':r['epochs'],'Es':r['Es'],
      'converged':r['converged'],**s,'dead_neurons':dead}
      for i,(_,r,s,dead) in enumerate(runs)] for name,runs in res.items()}
    with open(d/'latest_run.json','w',encoding='utf8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    if interactive:
        while True:
            a=input('A (or q): ').strip()
            if a.lower()=='q':break
            try:print(inference(res['relu'][0][0],float(a),float(input('B: '))))
            except ValueError as e:print(e)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--interactive',action='store_true')
    a=p.parse_args();main(1000 if a.quick else 20000,a.interactive)
