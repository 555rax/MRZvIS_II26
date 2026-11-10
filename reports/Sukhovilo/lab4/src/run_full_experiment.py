"""Полное воспроизводимое исследование варианта 5: 35 запусков + шум.

Внутри используются только NumPy и стандартная библиотека.
Остальные графики генерируются модулем render_plots.py.
"""
import os,json,csv
from pathlib import Path
import numpy as np
from forecast import *

ROOT=Path(__file__).resolve().parent
RES=ROOT/'results';RES.mkdir(parents=True,exist_ok=True)
raw=generate_series();lo,hi=train_minmax(raw);scaled=normalize(raw,lo,hi)
np.savetxt(ROOT/'series_variant_5.csv',np.column_stack((np.arange(200),raw,scaled)),delimiter=',',header='k,x_real,x_normalized',comments='',fmt=['%d','%.12f','%.12f'])
print('RANGE:', lo, hi,'TOTAL',raw.min(),raw.max(),'TEST normalized min/max',scaled[140:].min(),scaled[140:].max(),flush=True)

configs=[(p,6) for p in (2,4,7,14)]+[(7,n) for n in (1,2,12)]
rows=[];models={};histories={}
for p,n in configs:
 (X,y,k),(Xt,yt,kt)=make_windows(scaled,p)
 for seed in SEEDS:
  model=MLP(p,n,seed,DEFAULT_LR)
  r=model.fit(X,y,seed)
  t=metrics(raw[k],denormalize(model.predict(X),lo,hi))
  v=metrics(raw[kt],denormalize(model.predict(Xt),lo,hi))
  row={'p':p,'N':n,'seed':seed,'epochs':int(r['epochs']), 'Es':float(r['Es']),
       'converged':bool(r['converged']), 'train':t, 'test':v}
  rows.append(row);models[(p,n,seed)]=model;histories[(p,n,seed)]=r['history']
  model.save(RES/f'model_p{p}_N{n}_seed{seed}.npz')
  if seed==10:np.savetxt(RES/f'convergence_p{p}_N{n}.csv',np.column_stack((np.arange(1,r['epochs']+1),r['history'])),delimiter=',',header='epoch,Es',comments='',fmt=['%d','%.12g'])
  print(f'p={p:2d} N={n:2d} seed={seed:2d}, epoch={r["epochs"]:5d}, Es={r["Es"]:.3g}, train MAE={t["MAE"]:.5f}, test MAE={v["MAE"]:.5f}',flush=True)

# Aggregate data; standard deviations sample ddof 1.
def group(p,n):
 data=[r for r in rows if (r['p'],r['N'])==(p,n)]
 def avg(key):return float(np.mean([d[key] for d in data]))
 def sd(key):return float(np.std([d[key] for d in data],ddof=1))
 return {'p':p,'N':n,'n_runs':len(data),'epochs_mean':avg('epochs'),
  'epochs_sd':sd('epochs'),'epochs_min':int(min(d['epochs'] for d in data)),
  'epochs_max':int(max(d['epochs'] for d in data)),
  'MAE_test_mean':float(np.mean([d['test']['MAE'] for d in data])),
  'MAE_test_sd':float(np.std([d['test']['MAE'] for d in data],ddof=1)),
  'n_failed':sum(not d['converged'] for d in data)}
agg=[group(p,n) for p,n in configs]
# Select best mean test MAE; representative run with lowest test MAE for this config.
best=min(agg,key=lambda d:d['MAE_test_mean'])
rep=min([r for r in rows if (r['p'],r['N'])==(best['p'],best['N'])],key=lambda r:r['test']['MAE'])
print('BEST MEAN',best,'BEST SEED',rep['seed'],flush=True)

p,n,s=rep['p'],rep['N'],rep['seed'];model=models[(p,n,s)]
rec=denormalize(forecast_recursive(model,scaled,140,30),lo,hi)
true=raw[140:170]
errs=np.abs(rec-true)
threshold=float(.05*(raw.max()-raw.min()))
first=next((i+1 for i,e in enumerate(errs) if e>threshold),None)
print('RECURSION',[(i+1,round(float(e),5)) for i,e in enumerate(errs)][:10],'threshold',threshold,'first exceed',first,flush=True)
np.savetxt(RES/'recursive_30.csv',np.column_stack((np.arange(140,170),true,rec,errs)),delimiter=',',header='k,true,prediction,absolute_error',comments='',fmt=['%d','%.12f','%.12f','%.12f'])

# baseline inference including outlier
base=models[(7,6,10)]
inf=[]
for k in (24,72,132,140,157,185):
 inf.append(infer(base,raw,lo,hi,k=k))
for label,values in [('custom_constant',[0.0]*7),('custom_outlier',raw[90:97].copy())]:
 if label=='custom_outlier':values=np.array(values);values[3]=hi+1.5*(hi-lo)
 obj=infer(base,raw,lo,hi,values=values)
 obj['label']=label
 inf.append(obj)

# Comparative noise on baseline, 5 random seeds, 0.02 normalized units.
(X,y,k),(Xt,yt,kt)=make_windows(scaled,7)
noise=[]
for seed in SEEDS:
 rng=np.random.RandomState(8900+seed)
 X_noisy=X+rng.normal(loc=0,scale=.02,size=X.shape)
 m=MLP(7,6,seed,DEFAULT_LR)
 r=m.fit(X_noisy,y,seed)
 test=metrics(raw[kt],denormalize(m.predict(Xt),lo,hi))
 base_row=next(d for d in rows if (d['p'],d['N'],d['seed'])==(7,6,seed))
 noise.append({'seed':seed,'noise_sigma':.02,'epochs':r['epochs'],'Es_train_noisy':r['Es'],'converged':r['converged'],
               'test_mae_clean':test['MAE'],'test_mae_baseline':base_row['test']['MAE'],
               'difference':test['MAE']-base_row['test']['MAE']})
 print('NOISE',seed,r['epochs'],test['MAE'],base_row['test']['MAE'],flush=True)

study={
 'source':'PSAI 2.4.docx, variant 5','variant':5,
 'formula':'x(k) = 2.5*cos(0.35*k) + 1.2*sin(0.80*k)',
 'parameters':{'a':2.5,'b':.35,'c':1.2,'d':.80,'p':7,'N':6,'n_points':200,'train_end':140},
 'method':{'lr':DEFAULT_LR,'Ee':EE,'max_epochs':MAX_EPOCHS,'Es':'mean((yhat_norm-y_norm)^2)','loss_per_sample':'0.5 * (yhat-y)^2',
           'weight_initialization':'uniform [-0.1,0.1]','shuffle_seed':'1000 + seed','seeds':list(SEEDS)},
 'scale':{'train_min':lo,'train_max':hi,'test_norm_min':float(scaled[140:].min()),'test_norm_max':float(scaled[140:].max())},
 'counts':{'train':133,'test':60},
 'runs':rows,'aggregates':agg,'selected_config':best,'selected_run':rep,
 'recursive':{'threshold_5pct':threshold,'first_exceed_step':first,'mae_30':float(errs.mean()),'rmse_30':float(np.sqrt(np.mean(errs**2))),
              'predictions':rec.tolist(),'errors':errs.tolist()},
 'inference':inf,'noise':noise,
 'noise_mean_baseline':float(np.mean([d['test_mae_baseline'] for d in noise])),
 'noise_mean_noisy':float(np.mean([d['test_mae_clean'] for d in noise])),
}
(RES/'metrics.json').write_text(json.dumps(study,indent=2,ensure_ascii=False),encoding='utf-8')
with open(RES/'all_35_runs.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['p','N','seed','epochs','Es','converged','train_MAE','train_RMSE','train_sMAPE','test_MAE','test_RMSE','test_sMAPE'])
 for d in rows:w.writerow([d['p'],d['N'],d['seed'],d['epochs'],f"{d['Es']:.12f}",d['converged'],
  *[d['train'][k] for k in ('MAE','RMSE','sMAPE')],*[d['test'][k] for k in ('MAE','RMSE','sMAPE')]])
print('RUNS',len(rows),'noise',len(noise),'saved',RES)
from render_plots import build
build()
