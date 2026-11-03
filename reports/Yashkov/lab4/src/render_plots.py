"""Строит все графики лабораторной №4 из сохранённых метрик и моделей."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecast import *

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':120,
     'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.22,
     'savefig.bbox':'tight'})


def save(fig,name):
    fig.savefig(OUT/name,dpi=175,facecolor='white')
    plt.close(fig)


def build():
 setup()
 data=json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
 raw=generate_series();lo,hi=train_minmax(raw);sc=normalize(raw,lo,hi)
 base=MLP.load(OUT/'model_p8_N5_seed10.npz')
 (X,y,k),(Xt,yt,kt)=make_windows(sc,8)
 train=denormalize(base.predict(X),lo,hi);test=denormalize(base.predict(Xt),lo,hi)
 # 1. Source series and train/test one step overlay.
 fig,ax=plt.subplots(figsize=(10,4.8))
 ax.plot(np.arange(200),raw,'#a5b1c2',lw=1.5,label='Исходный ряд x(k)')
 ax.plot(k,train,'#2878b5',lw=1.15,alpha=.95,label='Одношаговый прогноз: train')
 ax.plot(kt,test,'#d15f3c',lw=1.45,label='Одношаговый прогноз: test')
 ax.axvline(139.5,color='#2b2b2b',ls='--',lw=1.1,label='Граница train/test')
 ax.set(xlabel='k',ylabel='x(k)',title='Вариант 10: исходный ряд и одношаговый прогноз (p=8, N=5, seed=10)')
 ax.legend(loc='lower left',ncol=2,fontsize=8);save(fig,'01_series_one_step.png')
 # 2. Loss (base and 2 alternative windows) log Y
 fig,ax=plt.subplots(figsize=(9.6,4.8))
 for p,c in [(4,'#bd632b'),(8,'#2878b5'),(16,'#40906f')]:
  h=np.loadtxt(OUT/f'convergence_p{p}_N5.csv',skiprows=1,delimiter=',')
  ax.semilogy(h[:,0],h[:,1],color=c,lw=1.8,label=f'p={p}, N=5')
 ax.axhline(EE,color='#464646',ls='--',label=f'Ee={EE:g}')
 ax.set(xlabel='Эпоха',ylabel='Es = средняя квадратичная ошибка (log scale)',title='Сходимость: базовая модель и два альтернативных окна')
 ax.legend();save(fig,'02_convergence_log.png')
 # 3. test scatter and diagonal.
 fig,ax=plt.subplots(figsize=(6.2,5.4))
 ax.scatter(raw[kt],test,c=kt,cmap='viridis',s=27,alpha=.85,label='60 тестовых точек')
 mi=min(np.min(raw[kt]),np.min(test));ma=max(np.max(raw[kt]),np.max(test));ax.plot([mi,ma],[mi,ma],'k--',lw=1,label='Идеально: y=x')
 ax.set(xlabel='Истинное значение x(k)',ylabel='Предсказание (исходная шкала)',title='Одношаговый прогноз: тестовый участок')
 ax.axis('equal');ax.legend();save(fig,'03_scatter_test.png')
 # 4. Recursive forecast and absolute error.
 rep=data['selected_run'];model=MLP.load(OUT/f"model_p{rep['p']}_N{rep['N']}_seed{rep['seed']}.npz")
 rec=denormalize(forecast_recursive(model,sc,140,30),lo,hi)
 idx=np.arange(140,170);error=np.abs(rec-raw[140:170])
 fig,ax=plt.subplots(figsize=(9.4,4.7))
 ax.plot(idx,raw[140:170],marker='.',lw=1.6,color='#334155',label='Эталон')
 ax.plot(idx,rec,marker='o',markersize=3,lw=1.35,color='#b6543a',label='Рекурсивный прогноз')
 ax.set(xlabel='k (140–169)',ylabel='x(k)',title=f"30-шаговый рекурсивный прогноз: p={rep['p']}, N={rep['N']}, seed={rep['seed']}")
 ax.legend();save(fig,'04_recursive_forecast.png')
 fig,ax=plt.subplots(figsize=(9.4,4.2))
 ax.plot(np.arange(1,31),error,'-o',lw=1.5,markersize=3,color='#286eaf',label='Абсолютная ошибка')
 ax.axhline(data['recursive']['threshold_5pct'],color='#ba4343',ls='--',label='5% полного размаха ряда')
 ax.set(xlabel='Шаг рекурсивного прогноза (1–30)',ylabel='|прогноз − эталон|',title='Абсолютная ошибка 30-шагового прогноза')
 ax.legend();save(fig,'05_recursive_error.png')
 # 5. Window study MAE + epochs.
 fig,(ax,ax2)=plt.subplots(1,2,figsize=(10.5,4.6))
 wins=[a for a in data['aggregates'] if a['N']==5]
 xs=np.arange(len(wins));ax.bar(xs,[d['MAE_test_mean'] for d in wins],yerr=[d['MAE_test_sd'] for d in wins],capsize=5,color='#5582b4')
 ax.set_xticks(xs,[str(d['p']) for d in wins]);ax.set(xlabel='Размер окна p',ylabel='Средняя MAE ± SD',title='Тестовая точность')
 ax2.bar(xs,[d['epochs_mean'] for d in wins],color='#73a59a')
 ax2.set_xticks(xs,[str(d['p']) for d in wins]);ax2.set(xlabel='Размер окна p',ylabel='Число эпох (среднее)',title='Скорость и сходимость')
 ax2.text(0,20000*.86,'5/5: не достигли Ee',ha='center',fontsize=8,color='#7d2f2f')
 fig.suptitle('Исследование длины входного окна (N=5, 5 запусков на конфигурацию)',fontsize=12)
 save(fig,'06_window_study.png')
 # 6. Neuron study.
 fig,(ax,ax2)=plt.subplots(1,2,figsize=(10.5,4.6))
 widths=sorted([a for a in data['aggregates'] if a['p']==8],key=lambda x:x['N'])
 xs=np.arange(len(widths));ax.bar(xs,[d['MAE_test_mean'] for d in widths],yerr=[d['MAE_test_sd'] for d in widths],capsize=5,color='#7385ba')
 ax.set_xticks(xs,[str(d['N']) for d in widths]);ax.set(xlabel='Число нейронов N',ylabel='Средняя MAE ± SD',title='Тестовая точность')
 ax2.bar(xs,[d['epochs_mean'] for d in widths],color='#8db1a3')
 ax2.set_xticks(xs,[str(d['N']) for d in widths]);ax2.set(xlabel='Число нейронов N',ylabel='Число эпох (среднее)',title='Сходимость')
 fig.suptitle('Влияние числа скрытых нейронов (p=8, 5 запусков)',fontsize=12)
 save(fig,'07_hidden_study.png')
 # 7. Noise with matched seeds.
 fig,ax=plt.subplots(figsize=(7.9,4.3))
 noise=data['noise'];ix=np.arange(len(noise));bw=.35
 ax.bar(ix-bw/2,[r['test_mae_baseline'] for r in noise],width=bw,color='#3c83ab',label='Без шума')
 ax.bar(ix+bw/2,[r['test_mae_clean'] for r in noise],width=bw,color='#d2855c',label='Шум на входах train, σ=0.02')
 ax.set_xticks(ix,[str(v['seed']) for v in noise]);ax.set(xlabel='Seed',ylabel='MAE на чистом тесте',title='Контролируемый эксперимент с шумом (p=8, N=5)')
 ax.legend();save(fig,'08_noise_comparison.png')
 # 8. schematic pipeline (precise offered blocks; exclude deliberately invalid blocks).
 fig,ax=plt.subplots(figsize=(11.7,5.9));ax.axis('off')
 nodes=[('1. Генерация ряда','k = 0…199'),('2. Train/Test split','140 / 60'),
  ('3. Min-Max train','min/max по k=0…139'),('4. Sliding window','p предыдущих → цель'),
  ('5. MLP p–N–1','sigmoid → linear'),('6. Shuffle every epoch','случайный порядок'),
  ('7. Online backprop','коррекция на каждом примере'),('8. Stop: Es ≤ 10⁻⁴','или 20 000 эпох'),
  ('9. Denormalization','обратная шкала'),('10. MAE, RMSE, sMAPE','train/test'),
  ('11. Recursive forecast','30 шагов от k=140')]
 from matplotlib.patches import FancyBboxPatch
 for i,(label,sub) in enumerate(nodes):
  col=i%4; row=i//4
  xx=.02+col*.245; yy=.72-row*.32
  rect=FancyBboxPatch((xx,yy),.22,.195,boxstyle='round,pad=0.008,rounding_size=0.02',transform=ax.transAxes,facecolor='#eef3f9' if row%2==0 else '#edf8f1',edgecolor='#54677a',lw=1)
  ax.add_patch(rect);ax.text(xx+.11,yy+.13,label,ha='center',va='center',fontsize=8.5,weight='bold',transform=ax.transAxes)
  ax.text(xx+.11,yy+.065,sub,ha='center',va='center',fontsize=7.6,transform=ax.transAxes)
  if i<10:
   if col<3:ax.annotate('',xy=(xx+.245,yy+.098),xytext=(xx+.225,yy+.098),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',lw=1.5,color='#5e6975'))
   else:ax.annotate('',xy=(xx+.11,yy-.12),xytext=(xx+.11,yy-.009),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',lw=1.4,color='#5e6975'))
 ax.text(.02,.018,'Исключены ошибочные блоки: Batch update; Normalize using whole dataset; Sigmoid output; MAPE для ряда, пересекающего ноль.',transform=ax.transAxes,fontsize=8.0,color='#9e3a32')
 ax.set_title('Схема обработки данных для варианта 10 (макет для рукописного листа)',fontsize=12,pad=12)
 save(fig,'09_pipeline_reference.png')
 print('Created all charts in',OUT)

if __name__=='__main__':build()
